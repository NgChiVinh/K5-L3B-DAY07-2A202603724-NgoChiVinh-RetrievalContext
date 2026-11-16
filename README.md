# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

## Bài làm Day 7

**Học viên:** Ngô Chí Vĩnh (`2A202603724`), lớp L3B · **Báo cáo:** [report/individual/2A202603724_NgoChiVinh.md](report/individual/2A202603724_NgoChiVinh.md)

| Thành phần | Giá trị |
|---|---|
| Corpus | World Bank *Điểm lại 8/2024* (CC BY 3.0 IGO), 47 trang nội dung, 195 chunk Day 6 chiến lược C |
| Embedding / vector store | `gemini-embedding-001` 3072 chiều · Qdrant local `day6_strategy_C` (cosine) |
| Agent LLM | `gemini-3.1-flash-lite` (Gemini free tier), function calling gốc; đổi sang OpenAI bằng `LLM_PROVIDER=openai` |
| Retrieval | dense (Qdrant), BM25 (âm tiết + bigram), hybrid RRF, LLM rerank |
| Benchmark | 24 câu: dev 15 có đáp án + 2 thiếu bằng chứng, test 5 + 2 ([data/day7/questions.json](data/day7/questions.json)) |
| Cấu hình đã chọn | `improve_multitool` (chốt trên dev: [results/selected_config.json](results/selected_config.json)) |

### Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env        # điền GEMINI_API_KEY; không commit .env
```

Store Qdrant và cache embedding của Day 6 không được commit. Copy từ repo Day 6 sau khi đã chạy `python -m src.run` ở `starter/`:

```powershell
xcopy /E /I <DAY6>\starter\qdrant_storage day6_store\qdrant_storage
copy <DAY6>\results\embedding_cache.json results\
```

### Chạy

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\python -m unittest discover -s tests -v                 # test offline (mock), 57 test
.venv\Scripts\python -m src.day7.export_day6                          # Qdrant -> data/day7/chunks.json
.venv\Scripts\python -m src.day7.build_benchmark                      # -> data/day7/questions.json, cases.json
.venv\Scripts\python -m src.day7.eval_retrieval --split dev           # Hit@1, EH@3, MRR của 5 phương pháp
.venv\Scripts\python -m src.day7.run_benchmark --config improve_multitool --split dev
.venv\Scripts\python -m src.day7.run_benchmark --config improve_multitool --split test   # cần selected_config.json
.venv\Scripts\python starter\agent_metrics.py --cases data\day7\cases.json --traces results\agent\test\improve_multitool\traces.json --reviews results\agent\test\improve_multitool\reviews.json --split test --output results\agent\test\improve_multitool\agent_metrics.json
.venv\Scripts\python -m src.day7.report_tables --split dev
```

Cấu hình agent: `baseline_dense`, `agent_bm25`, `agent_hybrid`, `improve_rerank`, `improve_multitool` (phần cơ bản); `improve_label*` (phần nâng cao) ([src/day7/configs.py](src/day7/configs.py)). `PYTHONUTF8=1` cần trên Windows: test có sẵn của starter đọc JSON không chỉ định encoding.

Code: [src/day7/](src/day7/) (agent, retrievers, llm, benchmark, các script chạy) · [src/day6/](src/day6/) (embedder và store copy nguyên từ Day 6) · kết quả và trace: [results/](results/).


### Thử thách nâng cao (+15): **30/30 test đạt (100%)**

Tài liệu: *Việt Nam 2045 – Tăng trưởng Xanh hơn* (World Bank 2025, 160 trang, CC BY-NC 3.0 IGO). Nguồn, thống kê index và phạm vi xử lý ở [data/adv/README.md](data/adv/README.md); phân tích ở mục "Thử thách nâng cao" của báo cáo.

```powershell
# tải PDF theo data/adv/README.md vào adv_store/greener_vi.pdf, rồi:
.venv\Scripts\python -m src.adv.ingest --step render
.venv\Scripts\python -m src.adv.ingest --step vision                         # Gemini vision từng trang
.venv\Scripts\python -m src.adv.ingest --step chart-values --model gpt-5.5   # đọc giá trị biểu đồ thiếu số liệu (200 dpi)
.venv\Scripts\python -m src.adv.ingest --step chunks
.venv\Scripts\python -m src.adv.ingest --step index
.venv\Scripts\python -m src.adv.build_benchmark
.venv\Scripts\python -m src.day7.eval_retrieval --corpus adv --split dev
.venv\Scripts\python -m src.day7.run_benchmark --corpus adv --config improve_label_strict_41_budget --split test
```

Agent của phần nâng cao dùng `ADV_LLM_PROVIDER=openai` và `ADV_LLM_MODEL=gpt-4.1-mini` (cấu hình đã chọn đổi sang `gpt-4.1`), cần `OPENAI_API_KEY` trong `.env`. Kết quả: `results/adv/` (`selected_config.json`, `test_labels.json`, trace từng cấu hình).

---

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
