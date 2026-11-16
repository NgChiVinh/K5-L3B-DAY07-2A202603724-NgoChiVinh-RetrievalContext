# Copied unchanged from Day 6 repo (NgChiVinh/K5-L3B-DAY06-...-DataIngestion, starter/src/store.py) to reuse the same embedder and Qdrant store.
"""Persistent Vector Store implementations (Qdrant on-disk persistent store & PostgreSQL pgvector)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .models import Chunk

PERMITTED_FILTER_KEYS = {"section", "sub_section", "year", "language", "document_id", "page"}


class QdrantVectorStore:
    """Persistent local Qdrant vector store supporting cosine similarity and payload metadata filtering."""

    def __init__(self, storage_path: str | Path, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("Vector dimensions must be > 0")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.client = QdrantClient(path=str(self.storage_path))

    @staticmethod
    def _collection_name(strategy: str) -> str:
        clean = "".join(c if c.isalnum() or c == "_" else "_" for c in strategy)
        return f"day6_strategy_{clean}"

    @staticmethod
    def _point_uuid(strategy: str, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"day6:{strategy}:{chunk_id}"))

    def create_schema(self, strategy: str) -> None:
        coll = self._collection_name(strategy)
        existing = [c.name for c in self.client.get_collections().collections]
        if coll not in existing:
            self.client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate dimensions, remove obsolete chunks for re-ingested docs, and upsert deterministically."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"Vector dimension {len(v)} != store dimension {self.dimensions}")

        self.create_schema(strategy)
        coll = self._collection_name(strategy)

        # Group incoming chunk point IDs by document_id to remove any obsolete chunks from earlier versions
        doc_to_new_ids: dict[str, set[str]] = {}
        for chunk in chunks:
            pid = self._point_uuid(strategy, chunk.chunk_id)
            doc_to_new_ids.setdefault(chunk.document_id, set()).add(pid)

        for doc_id, new_ids in doc_to_new_ids.items():
            records, _ = self.client.scroll(
                collection_name=coll,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
                ),
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )
            obsolete_ids = [str(r.id) for r in records if str(r.id) not in new_ids]
            if obsolete_ids:
                self.client.delete(
                    collection_name=coll,
                    points_selector=PointIdsList(points=obsolete_ids),
                )

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            pid = self._point_uuid(strategy, chunk.chunk_id)
            payload = dict(chunk.metadata)
            payload.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": chunk.text,
                    "strategy": strategy,
                }
            )
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        if points:
            self.client.upsert(collection_name=coll, points=points)

    def count(self, strategy: str) -> int:
        coll = self._collection_name(strategy)
        return int(self.client.count(collection_name=coll, exact=True).count)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata, rank by cosine similarity, and return top-k hits."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector dim {len(query_vector)} != store dim {self.dimensions}")

        coll = self._collection_name(strategy)
        qdrant_filter: Filter | None = None
        if filters:
            conditions: list[FieldCondition] = []
            for k, v in filters.items():
                if k not in PERMITTED_FILTER_KEYS:
                    raise ValueError(f"Metadata filter key '{k}' is not in permitted keys: {PERMITTED_FILTER_KEYS}")
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            qdrant_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name=coll,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for rank, pt in enumerate(response.points, start=1):
            payload = dict(pt.payload or {})
            text = payload.pop("text", "")
            chunk_id = payload.get("chunk_id", str(pt.id))
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": round(float(pt.score), 6),
                    "text": text,
                    "metadata": payload,
                }
            )
        return hits

    def close(self) -> None:
        self.client.close()


class PgVectorStore:
    """PostgreSQL + pgvector store implementation compatible with starter/docker-compose.yml."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        section TEXT,
                        sub_section TEXT,
                        year INTEGER,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        embedding vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    );
                    """
                )
            conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        import psycopg

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError("vector dimension mismatch")

        doc_ids = sorted({c.document_id for c in chunks})
        keep_ids = [c.chunk_id for c in chunks]

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                for doc_id in doc_ids:
                    cur.execute(
                        """
                        DELETE FROM chunks
                        WHERE strategy = %s AND document_id = %s AND NOT (chunk_id = ANY(%s));
                        """,
                        (strategy, doc_id, keep_ids),
                    )
                for chunk, vec in zip(chunks, vectors):
                    vec_str = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            strategy, chunk_id, document_id, source, page,
                            section, sub_section, year, text, metadata, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            section = EXCLUDED.section,
                            sub_section = EXCLUDED.sub_section,
                            year = EXCLUDED.year,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                        """,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            chunk.metadata.get("section"),
                            chunk.metadata.get("sub_section"),
                            chunk.metadata.get("year"),
                            chunk.text,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            vec_str,
                        ),
                    )
            conn.commit()

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        import psycopg

        vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vector) + "]"
        where_clauses = ["strategy = %s"]
        params: list[Any] = [strategy]

        if filters:
            for k, v in filters.items():
                if k not in PERMITTED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {k}")
                where_clauses.append(f"{k} = %s")
                params.append(v)

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT chunk_id, text, metadata, 1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        full_params = [vec_str] + params + [vec_str, top_k]

        hits: list[dict[str, Any]] = []
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, full_params)
                for rank, (cid, text, meta, sim) in enumerate(cur.fetchall(), start=1):
                    hits.append(
                        {
                            "rank": rank,
                            "chunk_id": cid,
                            "score": round(float(sim), 6),
                            "text": text,
                            "metadata": meta if isinstance(meta, dict) else json.loads(meta),
                        }
                    )
        return hits
