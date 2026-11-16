# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Ngô Chí Vĩnh · **MSSV:** `2A202603724` · **Lớp:** `L3B`
- **Day 6 tái sử dụng:** https://github.com/NgChiVinh/K5-L3B-DAY06-NgoChiVinh-2A202603724-DataIngestion
- Có dùng AI (Claude Code) hỗ trợ lập trình, chạy thí nghiệm và chấm nháp; học viên chịu trách nhiệm kiểm tra lại code, nhãn và kết quả (RULES.md).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Báo cáo *Điểm lại tháng 8/2024: Vươn tới tầm cao mới trên thị trường vốn* của Ngân hàng Thế giới (`document_id = wb-diem-lai-2024-08`), giấy phép **CC BY 3.0 IGO**. PDF có 64 trang, trong đó **47 trang nội dung hợp lệ** (13–18, 20–37, 39–61; quy tắc lọc trang theo Day 6). Dùng lại **195 chunk** của chiến lược C ở Day 6 (`by_paragraph` ≤ 1000 ký tự, metadata `section`). Chunk được xuất từ Qdrant ra [data/day7/chunks.json](../../data/day7/chunks.json) bằng `python -m src.day7.export_day6`. ID chunk giữ nguyên Day 6, ví dụ `wb-diem-lai-2024-08__C__p49__c002`.
- **Vector store, embedding model, phiên bản và số chiều:** **Qdrant local** (`qdrant-client==1.20.0`), collection `day6_strategy_C`, metric cosine, tìm kiếm exact. Embedding **`gemini-embedding-001`**, **3072 chiều**; câu hỏi nhúng với `task_type=RETRIEVAL_QUERY`, giống Day 6. Code embedder và store được copy nguyên từ Day 6 vào [src/day6/](../../src/day6/). Thư mục Qdrant và cache embedding được copy từ Day 6 sang `day6_store/` và `results/embedding_cache.json` (không commit; cách tái tạo ở README). Kiểm tra kết nối: dense trên dev với nhãn gốc Day 6 cho Hit@1 12/15 và Evidence Hit@3 15/15, **trùng khớp kết quả cấu hình C của Day 6**.
- **Agent model, cách chạy và phân chia dev/test:**
  - Model: **`gemini-3.1-flash-lite`** qua Gemini API **free tier**, `temperature=0`, `thinking_level=low`, dùng function calling gốc (native).
  - Lý do đổi model: ban đầu dùng `gemini-3.5-flash`, nhưng free tier của model này chỉ có **20 request/ngày** và đã hết quota trong lúc chạy rerank. Mọi kết quả trong báo cáo đều chạy bằng flash-lite.
  - Lệnh chạy: xem [README](../../README.md#bài-làm-day-7).
  - Chia dev/test: giữ đúng phân chia của Day 6 (q01–q15 dev, q16–q20 test) và thêm 4 câu thiếu bằng chứng. **Dev có 17 câu (15 có đáp án, 2 thiếu bằng chứng), test có 7 câu (5 + 2)**, tổng 24 câu. File: [data/day7/questions.json](../../data/day7/questions.json), tạo bằng `python -m src.day7.build_benchmark`.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* gold được lưu dạng `gold_groups`. Mỗi nhóm ứng với một ý mà câu hỏi bắt buộc phải có; trong một nhóm, chỉ cần một trong các evidence thay thế. Mỗi evidence phải nằm trọn trong một chunk đúng `source_id` và trang. So khớp sau khi chuẩn hóa NFC, chữ thường và gộp khoảng trắng. Một ranking được tính là hit khi top-k **phủ đủ mọi nhóm**. Bộ build tự kiểm tra rằng mọi evidence đều nằm trong chunk.
  - *Câu cần nhiều chunk:* q05, q08, q09, q11, q14 (dev) và q19 (test). Với các câu này, Hit@1 gần như không thể đạt (top-1 chỉ có một chunk), nên số Hit@1 thấp là do định nghĩa metric, không hẳn do ranking kém.
  - *Sửa nhãn (label QA):* nhãn được sửa sau khi đọc đối chiếu `expected_answer` với chunk, **trước khi chạy test**:
    - Ở Day 6, quote chỉ phủ một phần đáp án của q05 (thiếu 166 điểm cơ bản), q08 (thiếu 7% GDP), q09 (chỉ có I#2) và q14 (thiếu 86%). Mình bổ sung nhóm gold cho các ý còn thiếu.
    - q11 tách thành 3 nhóm (i)/(ii)/(iii) ở trang 50; trang 60 có câu tóm tắt cả ba nên được dùng làm evidence thay thế.
    - q19 tách thành 2 nhóm: 37 ngàn tỷ và 3 đợt IPO nằm ở hai chunk khác nhau.
    - q20 (test) chỉ giữ chunk trang 61 chứa tổng 87 tỷ USD, vì quote ở trang 60 chỉ phủ phần 20 tỷ.
    - Việc phát hiện nhãn thiếu bắt đầu từ việc đọc câu trả lời dev ở q14, nên metric dev được tính lại trên nhãn mới. Code và lý do ghi ở [src/day7/build_benchmark.py](../../src/day7/build_benchmark.py).
  - *Câu thiếu bằng chứng:* chọn câu gần chủ đề và kiểm tra bằng cách tìm từ khóa trên toàn bộ 195 chunk (trường `absence_check`):
    - u01 (dev): dự báo GDP năm 2027. Báo cáo chỉ dự báo đến 2026, không chunk nào chứa "2027".
    - u02 (dev): trái phiếu xanh năm 2023. Không chunk nào chứa "trái phiếu xanh".
    - u03 (test): % tăng giá vàng miếng. Báo cáo chỉ có nhu cầu mua vàng +28,6% nên đây là câu bẫy.
    - u04 (test): khối lượng giao dịch hợp đồng tương lai VN30. Từ "phái sinh" chỉ xuất hiện ở phần định nghĩa trang 42–43.
  - *Chấm câu trả lời:* thang 0/1/2 theo BENCHMARK.md. Mỗi citation được đọc đối chiếu với nội dung chunk. Abstention được tính đúng khi agent từ chối và không đưa ra khẳng định thiếu căn cứ.

## Retrieval và thử nghiệm

Hit@1 và Evidence Hit@3 được đo bằng **câu hỏi gốc cộng filter `section` do người dùng cung cấp** (có ở 6 câu, lấy từ trường `filter` của Day 6), top-10 cho mọi phương pháp. Context Evidence Rate đo trên **context cuối mà agent thật gửi cho model**: là tập hợp các chunk đã hiển thị trong kết quả tool, sau khi loại trùng và giới hạn bởi budget tính bằng **ký tự**. Chi phí thực tế là **0 USD** (Gemini free tier); số token được ghi để quy đổi nếu cần. Nguồn số liệu: `results/retrieval/<split>/*.json` và `results/agent/<split>/*/summary.json`; bảng sinh bằng `python -m src.day7.report_tables --split dev|test`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`baseline_dense`) | Dense Qdrant Day 6; agent 1 tool, top-k 3, filter user, không rerank, budget 3000 ký tự, tối đa 3 search | Mốc so sánh | dev | 10/15 (66,7%) | 14/15 (93,3%) | 12/15 (80,0%) | retrieval 0,19 s/câu; agent 7,7 s/câu, 33,7k token vào; 0 USD |
| Phương pháp 2 (`agent_bm25`) | BM25 (âm tiết + bigram âm tiết), cùng top-k/budget | Từ khóa giúp câu có số liệu, tên riêng | dev | 9/15 (60,0%) | 12/15 (80,0%) | 11/15 (73,3%) | 0,005 s/câu; agent 8,8 s; 33,8k token; 0 USD |
| Phương pháp 3 (`agent_hybrid`) | RRF(dense, BM25), k=60, 20 candidate mỗi nhánh | Gộp hai tín hiệu bù cho nhau | dev | 10/15 (66,7%) | 11/15 (73,3%) | 11/15 (73,3%) | 0,02 s/câu; agent 9,2 s; 34,2k token; 0 USD |
| Cải thiện 1 (`improve_rerank`) | Hybrid lấy 15 candidate, flash-lite chấm 0–10 rồi xếp lại; agent top-k 5, budget 5000 | Rerank đưa chunk đúng lên đầu, top-k 5 đủ ý cho câu nhiều ý | dev | 11/15 (73,3%) | 14/15 (93,3%) | 14/15 (93,3%) | rerank +5,8 s/câu (lần chạy đầu, chưa cache); agent 11,9 s; 42,9k token; 0 USD |
| Cải thiện 2 (`improve_multitool`) — **đã chọn** | Agent tự chọn `search_dense` hoặc `search_keyword`; top-k 5, budget 6000, tối đa 4 search, **thêm chunk lân cận ±1 cùng trang** | Lỗi dev đều do evidence nằm ở chunk kề nhau; thêm lân cận và cho phép tìm từ khóa sẽ giải quyết | dev | (dense 10/15, BM25 9/15) | (14/15, 12/15) | **15/15 (100%)** | agent 6,8 s; 59,6k token (≈1,8 lần baseline); 0 USD |
| Cấu hình đã chọn (`improve_multitool`) | Như trên, đã khóa trong `results/selected_config.json` | Đánh giá độc lập | test | (dense 3/5, BM25 2/5) | (4/5, 3/5) | 4/5 (80,0%) | agent 6,5 s; 23,6k token; 0 USD |
| Baseline (`baseline_dense`) | Như dòng đầu | Đánh giá độc lập | test | 3/5 (60,0%) | 4/5 (80,0%) | **5/5 (100%)** | agent 8,8 s; 13,8k token; 0 USD |

Retrieval thuần (câu hỏi gốc), bổ sung cho bảng trên:

| Phương pháp | dev Hit@1 | dev EH@3 | dev MRR@10 | test Hit@1 | test EH@3 | Latency / câu | Token LLM (dev, vào/ra) |
|---|---|---|---|---|---|---|---|
| dense | 10/15 | 14/15 | 0,789 | 3/5 | 4/5 | 0,19 s (embedding đã cache) – 3,3 s (test, query mới) | 0 |
| bm25 | 9/15 | 12/15 | 0,698 | 2/5 | 3/5 | 0,005 s | 0 |
| hybrid (RRF) | 10/15 | 11/15 | 0,736 | 3/5 | 4/5 | 0,02 s | 0 |
| dense + LLM rerank | 11/15 | 14/15 | 0,836 | 4/5 | 5/5 | 8,3 s dev (lần đầu) / 2,4 s test | 62,5k / 9,3k |
| hybrid + LLM rerank | 11/15 | 14/15 | 0,836 | 4/5 | 5/5 | 5,8 s dev (lần đầu) / 2,7 s test | 61,5k / 8,2k |

*Ghi chú:* hai phương pháp rerank được tính là một phương pháp riêng vì cách xếp hạng khác hẳn (model chấm từng đoạn). Đổi top-k không được tính là phương pháp mới. Latency rerank trên dev lấy từ lần chạy đầu tiên; lần tính lại sau khi sửa nhãn đọc từ cache nên không phản ánh thời gian thật.

**Lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:**

1. **q09, q14, q05: thiếu evidence trong context của baseline.**
   - *Triệu chứng:* q09 nêu I#2–I#4 chỉ dựa vào chú thích Hình 2.5, không có định nghĩa. q14 tự báo "không đủ bằng chứng" cho tỷ trọng 86%. q05 thiếu nguyên nhân chênh lệch lãi suất.
   - *Nguyên nhân:* đáp án trải trên các chunk kề nhau cùng trang (`p42__c001`/`c002`, `p51__c001`/`c002`, `p14__c003`/`c004`), trong khi top-3 dense đã bị các chunk trang khác chiếm chỗ, ví dụ q14 bị `p45` chiếm.
   - *Giả thuyết và cách thử:* lấy top-k 5 và thêm chunk lân cận ±1, budget 6000, ở cấu hình `improve_multitool`.
   - *Kết quả:* Context Evidence Rate trên dev tăng từ 12/15 lên 15/15; câu trả lời trung bình tăng từ 1,80 lên 2,00 (q05, q09, q14 cùng lên 2 điểm).
2. **q11: agent BM25 và hybrid từ chối nhầm.**
   - *Triệu chứng:* lượt tìm đầu chỉ lấy được `p50__c001`, chunk này kết thúc ở "(i) Ký quỹ…". Ba chunk đầu đã dùng khoảng 2700/3000 ký tự budget, nên lượt tìm thứ hai không thêm được đoạn nào (`added_ids: []`). Model kết luận "không liệt kê ba rào cản" và từ chối.
   - *Giả thuyết:* budget nhỏ chặn việc tìm nhiều lượt; nếu thêm chunk lân cận thì `p50__c002…c004` sẽ vào context.
   - *Kết quả:* `improve_multitool` trả lời đúng q11 (2 điểm); `improve_rerank` cũng đúng nhờ trang 60 được xếp hạng cao.
3. **Quan sát phụ:**
   - Với rerank, model hay chèn tên báo cáo vào truy vấn ("Điểm lại tháng 8/2024…"), làm truy vấn nhiễu thêm.
   - Rerank cộng thêm 1 lượt gọi LLM cho mỗi lần tìm, khiến agent chậm nhất (11,9 s/câu).
   - Khi thử nhanh với `gemini-3.5-flash-lite`, model **bỏ dấu tiếng Việt** trong truy vấn ("ty le no xau"). Đây là lý do prompt yêu cầu viết truy vấn có dấu và chọn `3.1-flash-lite`, vì model này giữ dấu.

**Chọn cấu hình** (tiêu chí xếp theo thứ tự ưu tiên, ghi trong [results/selected_config.json](../../results/selected_config.json), commit trước khi chạy test):

1. Context Evidence Rate trên dev.
2. Điểm câu trả lời trung bình trên dev ([results/dev_answer_grades.json](../../results/dev_answer_grades.json)).
3. Abstention: từ chối đúng câu thiếu bằng chứng và ít từ chối nhầm nhất.
4. Nếu hòa thì chọn cấu hình có latency và token thấp hơn.

`improve_multitool` dẫn đầu cả ba tiêu chí đầu: 15/15; 2,00; từ chối đúng 2/2 và không từ chối nhầm câu nào. Đổi lại, cấu hình này tốn ≈1,8 lần token đầu vào so với baseline.

## Agent và đánh giá

- **Tool:** có 4 tool: `search_dense` (Qdrant), `search_keyword` (BM25), `search_hybrid` (RRF) và `search_hybrid_rerank`. Mỗi cấu hình bật một tập con. Model tự viết truy vấn và tự quyết định có tìm tiếp hay không.
- **Số lượt gọi và điều kiện dừng:**
  - Mỗi câu được tìm tối đa `max_searches` lượt (3 hoặc 4).
  - Lượt tìm cuối có ghi chú "đây là lượt cuối". Nếu model vẫn gọi tool sau khi hết lượt, agent dừng với `search_limit` và từ chối.
  - Tối đa `max_searches + 2` lượt gọi LLM (`turn_limit`).
- **Cách chọn context:** loại trùng theo ID, giữ theo thứ tự hiển thị. Tổng độ dài text không được vượt budget (tính bằng ký tự); chunk vượt budget bị bỏ và ghi vào `skipped_over_budget`. Có tùy chọn thêm chunk lân cận ±1 cùng trang. Nội dung tài liệu được đánh dấu là dữ liệu, không phải chỉ dẫn.
- **Citation:** câu trả lời chỉ được chấp nhận khi có `[chunk-id]` và **mọi** ID đều nằm trong context. Nếu không, agent trả `abstain` với event `invalid_or_missing_citations`. Câu bắt đầu bằng "KHÔNG ĐỦ BẰNG CHỨNG" được tính là `abstain`.
- **Xử lý lỗi:**
  - Lỗi của tool được trả về cho model và ghi event `tool_error`.
  - Với lỗi 429 hoặc 5xx, code chờ rồi thử lại tối đa 6 lần. Riêng khi server yêu cầu chờ quá 300 s (hết quota ngày), code dừng ngay và ghi `status: "error"`.
  - HTTP timeout 90 s.
  - Không có fallback sang kết quả giả. Trên toàn bộ dev và test, số lỗi provider là **0**.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| baseline_dense / dev | 17/17 | 1 search: 15 câu, 2 search: 2 câu; 34 lượt gọi LLM | 0/3/12, TB 1,80 | không chấm riêng ở dev | 15/15 | 2/2 | 0/15 |
| improve_rerank / dev | 17/17 | 1: 16 câu, 2: 1 câu; 35 lượt gọi LLM agent + rerank | 0/1/14, TB 1,93 | không chấm riêng ở dev | 15/15 | 2/2 | 0/15 |
| improve_multitool / dev | 17/17 | 1: 14 câu, 2: 3 câu; 34 lượt gọi LLM | 0/0/15, TB 2,00 | không chấm riêng ở dev | 15/15 | 2/2 | 0/15 |
| **baseline_dense / test** | 7/7 | 1: 5 câu, 2: 2 câu; 14 lượt gọi LLM | **0/0/5, TB 2,00** | **8/8** | 5/5 | 2/2 | 0/5 |
| **improve_multitool / test** | 7/7 | 1: 5 câu, 2: 2 câu; 14 lượt gọi LLM | **0/1/4, TB 1,80** | **7/8** | 5/5 | 2/2 | 0/5 |

Agent BM25 và hybrid trên dev không được chấm câu trả lời thủ công, vì đã bị loại ở tiêu chí 1 và tiêu chí 3 (từ chối nhầm q11). Số liệu test được tổng hợp bằng `starter/agent_metrics.py`, xem `results/agent/test/<config>/agent_metrics.json`.

**Chấm context trên test** (0 = không có evidence, 1 = có nhưng thiếu điều kiện hoặc nguồn, 2 = đủ):

| Cấu hình | q16 | q17 | q18 | q19 | q20 | Số câu mỗi mức (0/1/2) |
|---|---|---|---|---|---|---|
| baseline_dense | 2 | 2 | 2 | 2 | 2 | 0/0/5 |
| improve_multitool | 2 | 2 | 2 | **1** (có 37 ngàn tỷ, thiếu chunk "ba đợt IPO") | 2 | 0/1/4 |

**Chi tiết từng câu test của cấu hình đã chọn (`improve_multitool`):**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 | trực tiếp | `search_dense` "ước tính giải ngân đầu tư công tăng 1 điểm phần trăm GDP…" → `p37__c002` hạng 1 | GDP tăng thêm 0,1% [`p37__c002`] | 2; citation đúng (câu ước tính nằm nguyên văn trong chunk) |
| q17 | điều kiện | `search_dense` (nguyên nhân chênh lệch) → `p46__c001`; `search_keyword` "Bộ Tài chính từ chối toàn bộ giá thầu…" → `p46__c002` | Nhà đầu tư "mua và nắm giữ"/BHXH bị hạn chế pháp lý; Bộ Tài chính hủy phát hành; 20% [`p46__c001`, `p46__c002`] | 2; hai citation đều hỗ trợ |
| q18 | diễn đạt lại | `search_keyword` "lưu ký bù trừ ngân hàng thanh toán…" → `p57__c001` hạng 1 | VSDC; NHNN cho trái phiếu chính phủ; NHTM nhà nước cho chứng khoán doanh nghiệp [`p57__c001`] | 2 |
| q19 | filter `chuong_2` | `search_keyword` (37 ngàn tỷ) → `p40__c001`, sau đó thêm lân cận, context đạt 5983/6000 ký tự; `search_keyword` "số đợt IPO năm 2023" → **`p49__c002` hạng 1 nhưng bị loại vì hết budget** | 37 ngàn tỷ [`p40__c001`] đúng; khẳng định sai "không có số đợt IPO" [`p49__c005`] | **1**; citation thứ hai không hỗ trợ (lỗi trên test, xem phần giới hạn) |
| q20 | filter `chuong_2` | `search_dense` (kịch bản 2030) → `p60__c003`, `p60__c004`, `p61__c002`… | 87 tỷ USD [`p61__c002`]; 20/15/14/5 tỷ [`p60__c003`] | 2 |
| u03 | thiếu bằng chứng | `search_keyword` "giá vàng miếng…" → `p14__c003` (nhu cầu mua vàng) | KHÔNG ĐỦ BẰNG CHỨNG; phân biệt rõ nhu cầu mua vàng +28,6% với giá vàng | Từ chối đúng |
| u04 | thiếu bằng chứng | `search_keyword` "hợp đồng tương lai VN30…" → các chunk không liên quan | KHÔNG ĐỦ BẰNG CHỨNG | Từ chối đúng |

Ví dụ trình bày:
- **Câu có bằng chứng:** q18.
- **Câu thiếu bằng chứng:** u03. Đây là câu bẫy: agent không lấy số 28,6% làm % tăng giá vàng.
- **Một lỗi:** q19 của cấu hình đã chọn.

Baseline trả lời đúng q19 với citation `p49__c001` và `p49__c002`.

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace: `results/agent/<split>/<config>/traces.json` lưu từng tool call, query, filter, kết quả kèm score/rank, `added_ids`, `skipped_over_budget`, context cuối, câu trả lời, citation, usage và latency.
  - Retrieval: `results/retrieval/<split>/<method>.json`.
  - Nhãn: `results/agent/test/<config>/reviews.json` và `results/dev_answer_grades.json`.
  - Log: `results/logs/`.
  - Tổng cộng 201 lượt gọi LLM của agent (241,7k token vào, 21,7k token ra, 27,3k token suy luận), cộng 48 lượt gọi rerank khi đánh giá retrieval. Chi phí **0 USD**.
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - *Trên dev*, cấu hình cải thiện 2 tăng Context Evidence Rate từ 12/15 lên 15/15 và câu trả lời từ 1,80 lên 2,00, với latency tương đương (6,8 so với 7,7 s/câu) nhưng tốn token gấp ≈1,8 lần. Cải thiện 1 (rerank) tăng MRR retrieval (0,789 → 0,836) và Context Evidence Rate (14/15), nhưng chậm nhất và token tăng 27%.
  - *Trên test, cấu hình đã chọn không tốt hơn baseline* (câu trả lời 1,8 so với 2,0; citation 7/8 so với 8/8). Nguyên nhân là ở q19: chunk lân cận của lượt tìm đầu (nhiều chunk chỉ là số liệu trục biểu đồ, như `p49__c003`/`c005`) lấp đầy budget, khiến chunk đúng của lượt tìm thứ hai bị loại. Đây đúng là rủi ro mà README của starter đã cảnh báo.
  - *Hướng sửa đề xuất:* chia budget cho từng lượt tìm, chỉ thêm lân cận khi chunk gốc có score cao, hoặc ưu tiên kết quả mới hơn chunk lân cận. Mình **không áp dụng** hướng sửa này, vì lỗi được phát hiện trên test; muốn đánh giá nó cần một bộ test mới (RULES).
  - *Giới hạn:*
    - Test chỉ có 7 câu, nên chênh lệch 1 câu chưa có ý nghĩa thống kê.
    - Nhãn và các câu thiếu bằng chứng do AI chấm nháp và cần học viên kiểm lại.
    - Nhãn Day 6 đã được sửa nhiều, nên số liệu không so trực tiếp được với báo cáo Day 6.
    - LLM có thể cho kết quả khác giữa các lần chạy dù `temperature=0`; cache `results/llm_cache/` giúp tái lập đúng lần chạy này trên máy học viên. Cache không được commit; người chấm chạy lại sẽ gọi model mới, nhưng toàn bộ trace của lần chạy này đã được lưu trong `results/`.
    - Free tier giới hạn quota theo ngày; `gemini-3.5-flash` chỉ có 20 request/ngày.
    - BM25 dùng âm tiết và bigram, chưa tách từ tiếng Việt thật.
    - Dữ liệu dạng bảng và biểu đồ bị `pypdf` trải phẳng từ Day 6 (ví dụ `p49__c005`), vừa làm nhiễu retrieval vừa tốn budget.

## Thử thách nâng cao (nếu thực hiện)

**Kết quả: cấu hình đã chốt đạt 30/30 câu test (100%), ngưỡng yêu cầu là 27/30.** Nhãn từng câu ở [results/adv/test_labels.json](../../results/adv/test_labels.json).

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - Ngân hàng Thế giới (2025), *Việt Nam 2045 – Tăng trưởng Xanh hơn: Con đường hướng tới tương lai bền vững*, bản tiếng Việt phát hành 09/06/2025.
  - **160 trang**, giấy phép **CC BY-NC 3.0 IGO**. Nguồn tải từ World Bank Open Knowledge Repository; SHA-256 của PDF ghi trong [data/adv/README.md](../../data/adv/README.md).
  - Repo không chứa PDF và ảnh trang; chỉ chứa nội dung trích xuất kèm ghi nguồn. Tài liệu có khoảng 50 hình (biểu đồ đường, cột, tròn, phân tán, hộp, sơ đồ dòng thời gian, infographic) và khoảng 15 bảng chính cùng nhiều bảng phụ lục.
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý** (code ở [src/adv/ingest.py](../../src/adv/ingest.py)):
  1. *Text:* PyMuPDF đọc theo block, bỏ header chạy, gộp thành chunk ≤ 1000 ký tự. Được **587 chunk**.
  2. *Bảng, biểu đồ, sơ đồ:* chuyển mỗi trang thành ảnh 100 dpi, rồi Gemini vision (`gemini-3.1-flash-lite`) liệt kê mọi phần tử trực quan dưới dạng JSON (loại, số hiệu, tiêu đề, vùng trên trang, nội dung). Bảng được chép lại dạng markdown; biểu đồ được mô tả trục, đơn vị, chú giải, nhãn số liệu, kèm dòng "Giá trị ước lượng (≈)". Được **114 chunk hình/bảng**. Mỗi chunk lưu số trang, số hiệu, vùng trên trang và đường dẫn ảnh trang gốc để kiểm chứng.
  3. *Lượt đọc giá trị (cải thiện D):* 11/55 biểu đồ thiếu dòng giá trị, như Hình 1.1 và A2.3.1. Lượt bổ sung đọc ảnh 200 dpi bằng `gpt-5.5`. Bản thử với `gpt-4.1` đọc **sai** (ví dụ Hình A2.3.1 ghi Hàn Quốc thấp nhất, trong khi gốc là In-đô-nê-xi-a ≈0,8), nên mình đã hoàn tác và không dùng.
  4. *Index:* `gemini-embedding-001` 3072 chiều, Qdrant local tại `adv_store/qdrant` (701 point), cùng BM25 trên 701 chunk.
  5. *Phạm vi:* index 158/160 trang; bỏ trang 2 (chỉ có số trang) và trang 159 (trắng). Ảnh bìa mang tính trang trí nên không đặt câu hỏi.
  6. *Đối chiếu với bản gốc:* các số liệu then chốt của 17 hình/bảng dùng trong test đều được kiểm tra trong chunk so với ảnh trang gốc, ví dụ Bảng 3.6 đúng hàng/cột và ghi chú, Hình 1.1 RCP 8.5 năm 2100 ≈28,3°C. Giá trị "≈" của biểu đồ không ghi nhãn có sai số đọc bằng mắt khoảng ±0,5 đơn vị trục.
  7. *Giới hạn:* tài liệu hầu như không có ảnh chụp. Nhóm "ảnh, sơ đồ" dùng sơ đồ dòng thời gian (Hình A2.4.1), infographic (Hình 3.6), hình phương pháp luận có khung chú giải (Hình 1.12), sơ đồ mục tiêu có mũi tên chú thích (Hình 0.5/2.3) và sơ đồ dải chi phí có nhãn (Hình 2.19b).
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:**
  - *Benchmark* ([data/adv/questions.json](../../data/adv/questions.json)) được chia trước khi thử nghiệm:
    - dev 18 câu: 15 có đáp án (bảng 3, biểu đồ 4, ảnh/sơ đồ 2, multi-hop 3, text 3) và 3 thiếu bằng chứng;
    - test 30 câu: 24 có đáp án (bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4) và 6 thiếu bằng chứng.
  - *Cách soạn câu hỏi:* câu hỏi và đáp án được viết bằng cách đọc ảnh trang gốc, không dựa vào output của model.
  - *Schema evidence:* mỗi câu có `group`, `evidence_note` (trang, bảng/hình, hàng/cột/ô hoặc vùng, phép tính), `reason` và `gold_groups`. Mỗi nhóm gold cho phép `{page, label}` (chunk của hình/bảng đó) hoặc `{page, evidence}` (chuỗi con trong text). Câu thiếu bằng chứng có `absence_reason` và `absence_check`.
  - *Thí nghiệm trên dev:* xem bảng dưới. Model agent là `gpt-4.1-mini` cho các cấu hình đầu và `gpt-4.1` cho C/E (OpenAI, `temperature=0`). Quota free tier Gemini (`generate_content`) đã hết trong ngày, nên phần nâng cao chuyển sang OpenAI; phần cơ bản vẫn giữ Gemini.

| Cấu hình (dev, 18 câu) | Thay đổi / giả thuyết | Context Evidence | Từ chối đúng | Đạt (chấm tay) |
|---|---|---|---|---|
| `baseline_dense` (4.1-mini) | dense, top-k 3, budget 3000 | 13/15 | 2/3 | 14/18 (trước cải thiện D) |
| `agent_hybrid` (4.1-mini) | RRF dense+BM25 | 13/15 | 2/3 | – |
| `improve_rerank` (4.1-mini) | hybrid + rerank LLM | 13/15 | 1/3 (du1 bịa giá tín chỉ từ mức thuế đề xuất) | – |
| `improve_multitool` (4.1-mini) | dense+BM25, neighbours | 11/15 (model chỉ chọn BM25) | 2/3 | – |
| A `improve_label` | Câu hỏi ghi "Bảng/Hình X.Y" thì chunk đó lên đầu; Hit@1 retrieval 10→12/15, MRR 0,776→0,865 | 13/15 | 2/3 | – |
| A+B `improve_label_strict` | + quy tắc prompt: marker từ chối, không thay chỉ tiêu gần giống | 13/15 | 2/3 (mini bỏ qua quy tắc) | – |
| A+B+C `improve_label_strict_41` | + model `gpt-4.1` | 14/15 | 3/3 | 15/18 (d04, d06 do thiếu giá trị biểu đồ; d10 do budget) |
| **A+B+C+D+E `improve_label_strict_41_budget`** | + lượt đọc giá trị biểu đồ (D) + budget 10.000 và tối đa 4.000 ký tự mỗi lượt tìm (E) | **15/15** | **3/3** | **18/18** |

Các lỗi trên dev dẫn tới từng cải thiện:
- d10: Bảng 2.3 được xếp hạng 1 ở lượt tìm thứ hai nhưng bị loại vì lượt đầu đã dùng 5160/6000 ký tự. Đây cũng là lỗi đã gặp ở test phần cơ bản, và dẫn tới cải thiện E.
- d06 và d04: biểu đồ không có dòng giá trị, dẫn tới cải thiện D.
- du1/du2: bịa số hoặc thiếu marker từ chối, dẫn tới cải thiện B và C.

Cấu hình và tiêu chí được khóa trong [results/adv/selected_config.json](../../results/adv/selected_config.json) (commit `76855ef`) trước khi chạy test.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | 6/6 | 100% | Tra đúng hàng/cột; t01, t02 có phép tính (0,46 điểm; gấp 2 lần); t04 đúng chiều tác động theo ghi chú |
| Ảnh, sơ đồ | 5/5 | 100% | t10 trả lời ≈1.550 triệu tấn, nằm trong khoảng chấp nhận 1.500–1.550 |
| Biểu đồ | 5/5 | 100% | Đều dựa trên nhãn số in trên hình (4,0/6,4; −3,0/−11,0; 29,4; 45%/18%) |
| Multi-hop | 4/4 | 100% | t17: bảng trang 36 + sơ đồ trang 139; t18: văn bản + Bảng 1.1, phép tính 0,26 điểm |
| Text | 4/4 | 100% | – |
| Thiếu bằng chứng | 6/6 | 100% | Nhận ra ô trống (Sing-ga-po), năm không có (2025), tỉnh không có trên hình (Đà Nẵng) |
| **Toàn bộ** | **30/30** | **100%** | 0 lỗi hệ thống, 0 timeout |

- **Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:**
  - Trace test: `results/adv/agent/test/improve_label_strict_41_budget/traces.json`, ghi tool call, query, kết quả kèm rank/score/method (`label`/`hybrid`), `added_ids`, `skipped_over_budget`, context, câu trả lời, citation và token.
  - Nhãn test: [results/adv/test_labels.json](../../results/adv/test_labels.json), có lý do và cách đối chiếu cho từng câu. Mọi citation, kể cả citation phụ (ví dụ t17 cite thêm trang 27/34/77/90), đều đã được đọc và đều hỗ trợ câu trả lời.
  - Retrieval trên test (câu hỏi gốc): dense Hit@1 19/24, EH@3 22/24; hybrid_label Hit@1 19/24, EH@3 23/24, MRR 0,880.
  - So sánh với baseline (`baseline_dense`, `gpt-4.1-mini`) trên test: chỉ chạy được 20/30 câu (t01–t20) trước khi hết quota embedding Gemini; 10 câu còn lại không có kết quả và không được tính. Trên 20 câu đó baseline đạt **16/20**. Bốn câu không đạt là t03 (đọc sai hàng/cột Bảng 2.3), t17 và t19 (thiếu một ý của câu multi-hop) và t20 (1,2% thay vì 1%). Cấu hình đã chọn đạt 20/20 trên cùng 20 câu. Nhãn ở `results/adv/agent/test/baseline_dense/partial_labels.json`.
  - Thời gian: agent trên test trung bình 4,8 s/câu (61 lượt gọi `gpt-4.1`, 89.546 token vào / 4.376 token ra). Vision Gemini: 158 trang, trung bình 4,3 s/trang, 251.904/59.432 token (free tier, 0 USD). Lượt đọc giá trị `gpt-5.5`: 11 lượt gọi, 51.171/15.167 token.
  - Chi phí OpenAI (dev + test): theo số token ghi trong các `summary.json`; số tiền thực tế cần xem trên trang billing của OpenAI.
  - Lỗi hệ thống trên test của cấu hình đã chọn: 0. Hết quota Gemini free tier trước đó được ghi là `status: error` trong các lần chạy dev bị thay thế, không thay bằng kết quả giả.
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - Câu hỏi và nhãn do AI hỗ trợ soạn và chấm, có đối chiếu ảnh trang gốc; **học viên cần kiểm tra lại**.
  - Test 30 câu là mẫu nhỏ; kết quả 100% không bảo đảm cho câu hỏi ngoài bộ này, nhất là với biểu đồ không có nhãn số liệu.
  - Retriever theo số hiệu chỉ hiệu quả khi người hỏi nêu số hình/bảng (24/30 câu test có nêu, kể cả câu thiếu bằng chứng). Benchmark vì vậy thiên về câu hỏi có số hiệu, và kết quả với câu hỏi tự nhiên không nêu số hình có thể thấp hơn.
  - Phần nâng cao dùng model khác phần cơ bản (OpenAI thay Gemini) do quota; agent và retriever dùng chung code.
  - Tái lập: xem [data/adv/README.md](../../data/adv/README.md) và README (lệnh `--corpus adv`).
