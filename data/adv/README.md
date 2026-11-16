# Dữ liệu thử thách nâng cao

**Tài liệu:** Ngân hàng Thế giới (2025). *Việt Nam 2045 – Tăng trưởng Xanh hơn: Con đường hướng tới tương lai bền vững* (bản tiếng Việt, phát hành 09/06/2025).

- Nguồn: World Bank Open Knowledge Repository, https://openknowledge.worldbank.org/entities/publication/ff18e434-f1fa-4332-bbf3-c25d9eda49c3. File PDF tiếng Việt là bitstream `36d46ea3-4dc5-4a4f-ae2d-cfb85a3d3981`, tải tại `https://openknowledge.worldbank.org/server/api/core/bitstreams/36d46ea3-4dc5-4a4f-ae2d-cfb85a3d3981/content`.
- SHA-256 của PDF: `7e8cb41a45006a584b1ae688ee8c7661dc385b24c6324a00658400f13a4e4f98`; 2.895.768 byte; **160 trang**.
- Giấy phép: **CC BY-NC 3.0 IGO**, cho phép dùng và chia sẻ phi thương mại nếu ghi nguồn. Repo không lưu file PDF và ảnh trang (`adv_store/`, thư mục bị gitignore). Repo chỉ lưu nội dung trích xuất phục vụ học tập, kèm ghi nguồn.
- Số trang: số trang PDF bắt đầu từ 1. Trang in số N tương ứng với trang PDF N + 32; trang in đánh số La Mã i–xxv tương ứng với trang PDF 9–33.

## Phạm vi index

| Hạng mục | Số lượng |
|---|---|
| Trang được index | 158/160. Bỏ trang 2 (chỉ có số trang) và trang 159 (trang trắng) |
| Chunk text (PyMuPDF, đã bỏ header chạy, gộp block ≤ 1000 ký tự) | 587 |
| Chunk hình/bảng (Gemini vision `gemini-3.1-flash-lite`, mỗi phần tử trực quan một chunk) | 114 |
| Trang gọi vision / lỗi | 158 / 0 |

Mỗi chunk hình/bảng lưu các trường sau trong `metadata`:
- `modality` (table, chart, diagram, map, image);
- `label` (ví dụ "Hình 1.6");
- `title`;
- `region` (vùng trên trang);
- `page_image` (ảnh trang gốc để đối chiếu).

Bảng được chép lại dạng markdown. Biểu đồ được mô tả đủ trục, đơn vị, chú giải, nhãn số liệu, kèm dòng "Giá trị ước lượng" (≈) khi biểu đồ không ghi nhãn số liệu.

**Phần chưa xử lý tốt:**
- Ảnh bìa (trang 1, 160) mang tính trang trí nên không index.
- Giá trị ước lượng (≈) đọc từ biểu đồ không có nhãn số liệu có sai số bằng mắt, khoảng ±0,5 đơn vị trục.
- Tài liệu hầu như không có ảnh chụp. Nhóm "ảnh, sơ đồ" dùng sơ đồ dòng thời gian, infographic, hình phương pháp luận và sơ đồ có chú thích mũi tên.

## File

- `vision_pages.json`: output vision từng trang (phần tử, token, latency).
- `chunks.json`: 701 chunk dùng cho BM25 và Qdrant (`adv_store/qdrant`, collection `day6_strategy_ADV`, `gemini-embedding-001`, 3072 chiều).
- `index_stats.json`: thống kê ingest.
- `questions.json`, `cases.json`: benchmark (dev 15+3, test 24+6). Câu hỏi được viết bằng cách đọc ảnh trang gốc, không dựa vào output của model.

Tái tạo:

```powershell
python -m src.adv.ingest --step render
python -m src.adv.ingest --step vision
python -m src.adv.ingest --step chunks
python -m src.adv.ingest --step index
python -m src.adv.build_benchmark
```
