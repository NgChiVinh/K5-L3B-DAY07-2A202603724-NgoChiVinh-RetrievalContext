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

Chưa thực hiện. Tài liệu Day 6 chỉ có 64 trang (cần từ 100 trang); kế hoạch là dùng lại agent và các retriever này với một tài liệu mới.
