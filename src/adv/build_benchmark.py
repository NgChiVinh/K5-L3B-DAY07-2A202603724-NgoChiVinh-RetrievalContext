"""Advanced benchmark for "Việt Nam 2045 – Tăng trưởng Xanh hơn" (160 pages).

Run: python -m src.adv.build_benchmark  ->  data/adv/questions.json, data/adv/cases.json
Questions and answers were written by reading the rendered ORIGINAL pages (adv_store/pages/pNNN.png),
not model output. Split fixed before any experiment: dev 15 answerable + 3 unanswerable,
test 24 answerable (bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4) + 6 unanswerable.
Gold alternatives: {'page', 'label'} = the visual chunk of that figure/table on that page;
{'page', 'evidence'} = substring of a PyMuPDF text chunk. Every group is validated against chunks.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from src.day7.benchmark import _alt_hit

from .ingest import CHUNKS_PATH, DATA, DOC_ID


def V(page: int, label: str) -> dict[str, Any]:
    return {'source_id': DOC_ID, 'page': page, 'label': label}


def T(page: int, evidence: str) -> dict[str, Any]:
    return {'source_id': DOC_ID, 'page': page, 'evidence': evidence}


def Q(qid, split, group, question, answer, groups, evidence_note, reason):
    return {'id': qid, 'split': split, 'group': group, 'question': question, 'answerable': True,
            'expected_behavior': 'search_and_answer', 'expected_answer': answer, 'filter': None,
            'gold_groups': groups, 'evidence_note': evidence_note, 'reason': reason}


def U(qid, split, question, absence_reason, absence_check):
    return {'id': qid, 'split': split, 'group': 'thieu_bang_chung', 'question': question, 'answerable': False,
            'expected_behavior': 'search_and_abstain', 'absence_reason': absence_reason,
            'absence_check': absence_check, 'filter': None}


QUESTIONS = [
    # ------------------------------------------------------------------ DEV (15 + 3)
    Q('d01', 'dev', 'bang', 'Theo Bảng 2.1 về mục tiêu giảm phát thải trong CLQG-BĐKH, lĩnh vực nông nghiệp phải giảm bao nhiêu phần trăm vào năm 2030 và lĩnh vực chất thải phải giảm bao nhiêu phần trăm vào năm 2050?',
      'Nông nghiệp giảm 43,0% vào năm 2030; chất thải giảm 90,7% vào năm 2050.',
      [[V(67, 'Bảng 2.1')]], 'Trang PDF 67 (tr.35), Bảng 2.1: hàng "Nông nghiệp" cột 2030 "% thay đổi" = 43,0; hàng "Chất thải" cột 2050 "% thay đổi" = 90,7.', 'Tra đúng hàng/cột của bảng.'),
    Q('d02', 'dev', 'bang', 'Theo Bảng 3.3, năm 2020 ngành sản xuất chế biến thực phẩm (chế biến và bảo quản thủy hải sản) có bao nhiêu lao động, và số lao động ngành đánh bắt và nuôi trồng thủy sản năm 2020 là bao nhiêu?',
      'Sản xuất chế biến thực phẩm: 536.390 lao động; đánh bắt và nuôi trồng thủy sản: 32.461 lao động (năm 2020).',
      [[V(100, 'Bảng 3.3')]], 'Trang PDF 100 (tr.68), Bảng 3.3: hàng 6 và hàng 1, cột 2020.', 'Tra hai ô cùng cột.'),
    Q('d03', 'dev', 'bang', 'Theo Phụ lục 1.7, ví dụ quốc tế nào được nêu cho hành động "Đầu tư hạ tầng thích ứng với biến đổi khí hậu" trong lĩnh vực giảm thiểu rủi ro thiên tai?',
      'Các dự án quản lý lũ lụt của Thái Lan: xây dựng rào chắn lũ, hồ chứa và hệ thống thoát nước để giảm tác động của mưa lớn và ngăn lũ lụt ở các khu vực đô thị như Bangkok.',
      [[V(129, 'Phụ lục 1.7'), T(129, 'Các dự án quản lý lũ lụt của Thái Lan')]], 'Trang PDF 129 (tr.97), bảng Phụ lục 1.7, hàng "Đầu tư hạ tầng thích ứng với biến đổi khí hậu", cột "Ví dụ".', 'Tra ô theo hàng hành động.'),
    Q('d04', 'dev', 'bieu_do', 'Theo Hình 1.1, đến khoảng năm 2100 kịch bản phát thải nào cho nhiệt độ trung bình năm của Việt Nam cao nhất và ở mức khoảng bao nhiêu độ C?',
      'Kịch bản RCP 8.5, khoảng 28–28,5°C vào cuối thế kỷ (gần 2100).',
      [[V(40, 'Hình 1.1')]], 'Trang PDF 40 (tr.8), Hình 1.1 (biểu đồ đường, nửa trên trái): đường đỏ RCP 8.5 cao nhất, kết thúc ≈28,3°C.', 'Đọc chú giải và trục tung; chấp nhận 28–28,5°C.'),
    Q('d05', 'dev', 'bieu_do', 'Theo Hình 0.1, đến năm 2050 GDP theo giá so sánh giảm khoảng bao nhiêu phần trăm so với kịch bản cơ sở trong kịch bản tích cực và trong kịch bản tiêu cực?',
      'Năm 2050: kịch bản tích cực khoảng −11% (≈ −11,2%); kịch bản tiêu cực khoảng −15,5% đến −16% (≈ −15,7%).',
      [[V(16, 'Hình 0.1')]], 'Trang PDF 16 (tr.viii), Hình 0.1, điểm cuối năm 2050 của đường xanh lá (Tích cực) và đường xám (Tiêu cực).', 'Biểu đồ không có nhãn số: chấp nhận sai số ±0,5 điểm.'),
    Q('d06', 'dev', 'bieu_do', 'Theo Hình A2.3.1 về tiêu thụ năng lượng sơ cấp theo % GDP (kWh/$), quốc gia nào có mức thấp nhất và Việt Nam ở mức khoảng bao nhiêu?',
      'In-đô-nê-xi-a thấp nhất (khoảng 0,8 kWh/$); Việt Nam khoảng 1,5 kWh/$ (cột màu đỏ, ≈1,48).',
      [[V(137, 'Hình A2.3.1')]], 'Trang PDF 137 (tr.105), Hình A2.3.1 (biểu đồ cột, bên trái).', 'Đọc cột thấp nhất và cột Việt Nam; chấp nhận 1,4–1,5.'),
    Q('d07', 'dev', 'bieu_do', 'Theo Hình 1.3 về tình trạng đường quốc lộ, vùng nào có tỷ lệ đường ở tình trạng "Tốt" cao nhất và vùng nào thấp nhất?',
      'Đồng bằng sông Hồng cao nhất (khoảng 46%); Đồng bằng sông Cửu Long thấp nhất (khoảng 15%).',
      [[V(43, 'Hình 1.3')]], 'Trang PDF 43 (tr.11), Hình 1.3 (biểu đồ thanh ngang xếp chồng), đoạn màu xanh "Tốt".', 'So sánh độ dài đoạn "Tốt" giữa các vùng.'),
    Q('d08', 'dev', 'anh_so_do', 'Theo sơ đồ Hình A2.4.1 về kế hoạch phát triển thị trường các-bon, giai đoạn 2023–2024 tập trung vào việc gì?',
      'Xây dựng hệ thống quy định về quản lý tín chỉ các-bon, trao đổi hạn ngạch và hoạt động của nền tảng giao dịch tín chỉ các-bon.',
      [[V(139, 'Hình A2.4.1')]], 'Trang PDF 139 (tr.107), sơ đồ dòng thời gian, ô bên trái dưới mốc 2023–2024.', 'Nhận diện nội dung ô sơ đồ.'),
    Q('d09', 'dev', 'anh_so_do', 'Trong sơ đồ Hình 0.5 về mục tiêu giảm nhẹ, mũi tên ghi "-27%" ứng với chỉ tiêu nào và mũi tên "-43.5%" ứng với chỉ tiêu nào?',
      '"-27%" ứng với Chỉ tiêu NDC 2020; "-43.5%" ứng với Chỉ tiêu NDC 2022 (đều cho năm 2030, so với baseline).',
      [[V(22, 'Hình 0.5')]], 'Trang PDF 22 (tr.xiv), Hình 0.5, hai cột năm 2030 với mũi tên và nhãn.', 'Đọc nhãn chú thích trên sơ đồ.'),
    Q('d10', 'dev', 'multi_hop', 'Khi thuế BVMT đối với than được cải biên thành thuế các-bon 7,5 US$/tấn thì mức thuế là bao nhiêu VND/tấn (Bảng 2.2), và chính sách thuế các-bon làm tỷ lệ nghèo theo chuẩn 6,85 US$ tăng bao nhiêu điểm phần trăm (Bảng 2.3)?',
      '500.000 VND/tấn than; thuế các-bon làm tỷ lệ nghèo (chuẩn 6,85 US$) tăng 0,5 điểm phần trăm (−0,5 trong bảng, dấu âm = tăng nghèo).',
      [[V(78, 'Bảng 2.2')], [V(82, 'Bảng 2.3')]], 'Trang 78 Bảng 2.2 hàng Than cột cuối; trang 82 Bảng 2.3 hàng "Thuế các-bon" cột 6,85 US$.', 'Kết hợp hai bảng ở hai trang.'),
    Q('d11', 'dev', 'multi_hop', 'Theo Hình 2.19a, tỷ lệ công suất điện mặt trời của Việt Nam đạt đỉnh khoảng bao nhiêu phần trăm và vào năm nào; QHĐ8 sửa đổi đặt chỉ tiêu điện mặt trời bao nhiêu GW vào năm 2030?',
      'Tỷ lệ điện mặt trời đạt đỉnh khoảng 11% vào năm 2021 (≈10,7% năm 2022); QHĐ8 sửa đổi đặt chỉ tiêu 73 GW điện mặt trời vào năm 2030 (tăng hơn 5 lần so với 12,8 GW trước đó).',
      [[V(88, 'Hình 2.19')], [T(22, 'chỉ tiêu 73 GW điện mặt trời vào năm 2030'), T(67, '73 GW điện mặt trời vào năm 2030')]],
      'Trang 88 Hình 2.19a (đường vàng); trang 22 hoặc 67 văn bản về QHĐ8 sửa đổi.', 'Kết hợp biểu đồ và văn bản.'),
    Q('d12', 'dev', 'multi_hop', 'Rừng ngập mặn có giá trị bình quân bao nhiêu USD quốc tế/ha/năm theo Hình 3.6, và diện tích rừng ngập mặn của Việt Nam năm 2020 so với năm 1990 thay đổi thế nào?',
      '45.271 USD quốc tế/ha/năm; diện tích rừng ngập mặn giảm từ khoảng 250.000 ha (1990) xuống khoảng 150.000 ha (2020).',
      [[V(109, 'Hình 3.6')], [T(31, 'khoảng 150.000 héc-ta, giảm so với khoảng 250.000 héc-ta')]],
      'Trang 109 bảng trong Hình 3.6; trang 31 văn bản Tổng quan.', 'Kết hợp hình và văn bản.'),
    Q('d13', 'dev', 'text', 'Hệ số Gini của Việt Nam năm 2022 là bao nhiêu, dự báo tăng lên bao nhiêu vào năm 2050, và điều đó khiến thêm bao nhiêu người rơi xuống dưới ngưỡng nghèo?',
      'Gini 36,1 năm 2022, tăng 5% lên 37,9 vào năm 2050, khiến thêm hơn 400.000 người rơi xuống dưới ngưỡng nghèo.',
      [[T(37, 'đang ở mức 36,1 vào năm 2022')]], 'Trang PDF 37 (tr.5), Thông điệp chính Chương 1.', 'Tổng hợp ba số liệu trong một đoạn.'),
    Q('d14', 'dev', 'text', 'Chi tiêu cho an sinh xã hội của Việt Nam ở mức bao nhiêu % GDP, và phạm vi bao phủ của hỗ trợ bằng tiền trên toàn quốc và ở nhóm nghèo nhất là bao nhiêu?',
      'Dưới 1% GDP; phạm vi bao phủ 13% trên toàn quốc và 25% người dân ở nhóm nghèo nhất.',
      [[T(82, 'Chi tiêu cho an sinh xã hội, ở mức dưới 1% GDP')]], 'Trang PDF 82 (tr.50), đoạn sau Bảng 2.3.', 'Tìm điều kiện trong văn bản.'),
    Q('d15', 'dev', 'text', 'Năm 2020 Việt Nam có bao nhiêu biện pháp phi thuế quan áp dụng cho bao nhiêu mặt hàng thân thiện môi trường, và nhóm sản phẩm nào chịu nhiều biện pháp nhất?',
      '199 biện pháp phi thuế quan cho 54 mặt hàng; sản phẩm năng lượng tái tạo chịu nhiều nhất (62 biện pháp).',
      [[T(88, '199 biện pháp phi thuế quan áp dụng cho 54 mặt hàng')]], 'Trang PDF 88 (tr.56), đoạn đầu trang.', 'Tìm số liệu trong văn bản.'),
    U('du1', 'dev', 'Giá tín chỉ các-bon giao dịch trên sàn thí điểm năm 2025 là bao nhiêu US$/tấn CO2?',
      'Báo cáo chỉ nêu lộ trình thí điểm (Hình A2.4.1, tr.139) và giá các-bon theo kịch bản mô phỏng (Hình 2.12a), không có giá giao dịch thực tế trên sàn thí điểm.',
      "Không trang nào chứa 'giá tín chỉ'; 'tín chỉ các-bon' chỉ ở tr.77 và tr.139 (khung pháp lý, lộ trình)."),
    U('du2', 'dev', 'Theo Hình 1.6, tỷ lệ thiệt hại do căng thẳng nhiệt độ cao vào năm 2060 là bao nhiêu?',
      'Hình 1.6 (tr.46) chỉ có các năm 2025–2050.', 'Trục hoành Hình 1.6 dừng ở 2050; tr.46 không chứa "2060".'),
    U('du3', 'dev', 'Năm 2024 Việt Nam có bao nhiêu ô tô điện được đăng ký mới?',
      'Báo cáo chỉ nhắc chính sách/lộ trình xe điện, không có số lượng xe đăng ký.', "Không trang nào chứa 'ô tô điện'; 'xe điện' chỉ xuất hiện trong ngữ cảnh chính sách và xuất khẩu."),
    # ------------------------------------------------------------------ TEST (24 + 6)
    Q('t01', 'test', 'bang', 'Theo Bảng 1.1, tốc độ tăng trưởng GDP dài hạn theo giá so sánh giai đoạn 2040–50 trong kịch bản tiêu cực (RCP8.5) là bao nhiêu phần trăm và thấp hơn kịch bản cơ sở bao nhiêu điểm phần trăm?',
      '4,10%; thấp hơn kịch bản cơ sở (4,56%) 0,46 điểm phần trăm.',
      [[V(44, 'Bảng 1.1')]], 'Trang PDF 44 (tr.12), Bảng 1.1: hàng "Tiêu cực (RCP8.5)" và "Kịch bản cơ sở", cột 2040–50.', 'Tra ô và tính hiệu.'),
    Q('t02', 'test', 'bang', 'Theo Bảng 2.2, với xăng, tỷ lệ tiêu dùng cuối là bao nhiêu phần trăm, thuế suất thuế BVMT tối đa (EPT+) là bao nhiêu VND/lít, và gấp mấy lần thuế suất hiện nay?',
      'Tỷ lệ tiêu dùng cuối 8%; thuế suất tối đa (EPT+) 4.000 VND/lít, gấp 2 lần thuế suất hiện nay (2.000 VND/lít).',
      [[V(78, 'Bảng 2.2')]], 'Trang PDF 78 (tr.46), Bảng 2.2, hàng "Xăng".', 'Tra ô và so sánh.'),
    Q('t03', 'test', 'bang', 'Theo Bảng 2.3 (dấu âm nghĩa là tăng tỷ lệ nghèo), kịch bản chính sách nào làm tăng tỷ lệ nghèo theo chuẩn 6,85 US$ nhiều nhất và tăng bao nhiêu điểm phần trăm?',
      'Kịch bản "Tham vọng": tăng 1,7 điểm phần trăm (giá trị −1,7).',
      [[V(82, 'Bảng 2.3')]], 'Trang PDF 82 (tr.50), Bảng 2.3, cột 6,85 US$, hàng "Tham vọng"; ghi chú dấu âm.', 'Tra cột và hiểu ghi chú đơn vị/dấu.'),
    Q('t04', 'test', 'bang', 'Trong Bảng 3.6 về tương tác giữa các ngành biển, ô ở hàng "Môi trường và đa dạng sinh học" và cột "Dầu khí" có điểm bao nhiêu, và theo ghi chú điểm đó thể hiện tác động của ngành nào lên ngành nào?',
      'Điểm −2: tác động (tiêu cực) của ngành Dầu khí (cột) đối với Môi trường và đa dạng sinh học (hàng).',
      [[V(114, 'Bảng 3.6')]], 'Trang PDF 114 (tr.82), Bảng 3.6, hàng cuối, cột 2; ghi chú dưới bảng.', 'Tra ô ma trận và đọc ghi chú chiều tác động.'),
    Q('t05', 'test', 'bang', 'Theo Bảng 3.2, tổng giá trị gia tăng của điện gió ngoài khơi năm 2020 là bao nhiêu tỷ VND và dự báo năm 2030 là bao nhiêu tỷ VND (giá năm 2010)?',
      '10.054 tỷ VND năm 2020; dự báo 178.800 tỷ VND năm 2030.',
      [[V(99, 'Bảng 3.2')]], 'Trang PDF 99 (tr.67), Bảng 3.2, hàng 7 "Điện gió ngoài khơi", cột 2020 và 2030 (dự báo).', 'Tra hai ô cùng hàng.'),
    Q('t06', 'test', 'bang', 'Theo Bảng 3.4, quốc gia nào có tỷ lệ % GDP hoặc GVA từ các ngành dựa vào đại dương cao nhất và tỷ lệ đó là bao nhiêu; Trung Quốc có tỷ lệ việc làm trên tổng việc làm là bao nhiêu?',
      'Ai-xơ-len cao nhất với 26%; Trung Quốc có tỷ lệ việc làm 40,2%.',
      [[V(101, 'Bảng 3.4')]], 'Trang PDF 101 (tr.69), Bảng 3.4, cột "% GDP hoặc GVA" và hàng Trung Quốc.', 'So sánh trong cột và tra ô.'),
    Q('t07', 'test', 'anh_so_do', 'Theo sơ đồ Hình A2.4.1 về kế hoạch phát triển thị trường các-bon, giai đoạn 2025–2027 là hoạt động gì?',
      'Thí điểm trao đổi tín chỉ các-bon.',
      [[V(139, 'Hình A2.4.1')]], 'Trang PDF 139 (tr.107), ô giữa dưới mốc 2025–2027.', 'Nhận diện nội dung ô sơ đồ.'),
    Q('t08', 'test', 'anh_so_do', 'Trong biểu đồ tròn của Hình 3.6, phần nào gần như không nhìn thấy (lát rất mỏng) và giá trị bình quân tương ứng của nó là bao nhiêu USD quốc tế/ha/năm?',
      'Các hệ sinh thái ven biển, trừ rừng ngập mặn: 400 USD quốc tế/ha/năm.',
      [[V(109, 'Hình 3.6')]], 'Trang PDF 109 (tr.77), infographic Hình 3.6: nhãn ở đỉnh hình tròn + bảng bên trái.', 'Nhận diện thành phần trên hình và đối chiếu bảng kèm theo.'),
    Q('t09', 'test', 'anh_so_do', 'Theo khung chú giải của Hình 1.12 (phương pháp luận lợi ích gấp ba), "lợi ích thứ ba" bao gồm những gì và tỷ lệ lợi ích/chi phí là bao nhiêu?',
      'Giảm khí thải hiệu ứng nhà kính, dịch vụ hệ sinh thái, quản lý nước; tỷ lệ lợi ích/chi phí = 1,38.',
      [[V(58, 'Hình 1.12')]], 'Trang PDF 58 (tr.26), khung chú giải bên phải Hình 1.12.', 'Đọc chú giải của hình.'),
    Q('t10', 'test', 'anh_so_do', 'Trong Hình 2.3, mũi tên ở cột năm 2050 đi từ mức nào xuống mục tiêu gì?',
      'Từ kịch bản cơ sở (khoảng 1.500 triệu tấn CO2 tương đương) xuống mục tiêu của CLQG-BĐKH – phát thải ròng bằng không.',
      [[V(66, 'Hình 2.3')]], 'Trang PDF 66 (tr.34), Hình 2.3 (nửa dưới), cột 2050, nhãn "Kịch bản cơ sở" và "Mục tiêu của CLQG-BĐKH - Phát thải ròng bằng không".', 'Đọc nhãn và mũi tên trên sơ đồ; chấp nhận 1.500–1.550.'),
    Q('t11', 'test', 'anh_so_do', 'Trong Hình 2.19b, chi phí điện quy dẫn (LCOE) của điện mặt trời cộng pin được thể hiện theo những mức công suất lưu trữ nào, và được so sánh với những loại nhà máy điện mới nào?',
      'Theo các mức 25%, 50% và 100% công suất; so sánh với nhà máy điện than và CCGT (tua-bin khí chu trình kết hợp).',
      [[V(88, 'Hình 2.19')]], 'Trang PDF 88 (tr.56), Hình 2.19b (bên phải), các nhãn trên dải.', 'Nhận diện nhãn trên sơ đồ dải chi phí.'),
    Q('t12', 'test', 'bieu_do', 'Theo Hình 1.6, tỷ lệ thiệt hại do căng thẳng nhiệt độ cao (RCP4.5 so với kịch bản cơ sở) năm 2025 và năm 2035 là bao nhiêu?',
      '4,0% năm 2025 và 6,4% năm 2035.',
      [[V(46, 'Hình 1.6')]], 'Trang PDF 46 (tr.14), Hình 1.6, nhãn số trên cột xanh đậm 2025 và 2035.', 'Đọc nhãn số liệu trên cột.'),
    Q('t13', 'test', 'bieu_do', 'Trong Hình 3.5, tỉnh nào có nhiều khu công nghiệp và khu chế xuất đang hoạt động nhất, bao nhiêu khu, và trong đó bao nhiêu khu có nhà máy xử lý nước thải tập trung đạt chuẩn?',
      'Thái Bình: 6 khu đang hoạt động, trong đó 4 khu có nhà máy xử lý nước thải tập trung đạt chuẩn.',
      [[V(105, 'Hình 3.5')]], 'Trang PDF 105 (tr.73), Hình 3.5, cặp cột Thái Bình.', 'Đọc chiều cao cột theo trục tung.'),
    Q('t14', 'test', 'bieu_do', 'Theo Hình 2.17 (kịch bản giảm 30% xuất khẩu sang OECD), lĩnh vực nào chịu tổn thất GDP lớn nhất và mức thay đổi là bao nhiêu theo % tổng GDP và % GDP ngành?',
      'Chế biến chế tạo: −3,0% tổng GDP và −11,0% GDP ngành.',
      [[V(86, 'Hình 2.17')]], 'Trang PDF 86 (tr.54), Hình 2.17, cặp cột đầu tiên.', 'Đọc nhãn hai chuỗi.'),
    Q('t15', 'test', 'bieu_do', 'Theo Hình 2.15b, năm 2024 Hoa Kỳ chiếm bao nhiêu phần trăm tổng xuất khẩu của Việt Nam, và hai thị trường OECD tiếp theo là nước nào với tỷ lệ bao nhiêu?',
      'Hoa Kỳ 29,4%; tiếp theo là Hàn Quốc 6,3% và Nhật Bản 6,1%.',
      [[V(85, 'Hình 2.15')]], 'Trang PDF 85 (tr.53), Hình 2.15b (bên phải), nhãn trên cột.', 'Đọc nhãn và thứ tự cột.'),
    Q('t16', 'test', 'bieu_do', 'Theo Hình 3.2, năm 2019 ngành kinh tế biển nào đóng góp tỷ trọng lớn nhất vào tổng giá trị gia tăng và bao nhiêu phần trăm; ngành chế biến than cốc và các sản phẩm dầu mỏ tinh chế chiếm bao nhiêu?',
      'Du lịch biển lớn nhất với 45%; chế biến than cốc và các sản phẩm dầu mỏ tinh chế chiếm 18%.',
      [[V(99, 'Hình 3.2')]], 'Trang PDF 99 (tr.67), Hình 3.2 (biểu đồ tròn, trên phải).', 'Đọc chú giải màu và nhãn %.'),
    Q('t17', 'test', 'multi_hop', 'Theo bảng tiến độ năm gói chính sách của CCDR (phần Giới thiệu), hệ thống giao dịch phát thải (ETS) dự kiến thí điểm từ năm nào và triển khai đầy đủ năm nào; theo Hình A2.4.1, năm nào sàn giao dịch tín chỉ các-bon chính thức ra mắt?',
      'ETS thí điểm từ năm 2025 và triển khai đầy đủ vào năm 2028; theo Hình A2.4.1, sàn giao dịch tín chỉ các-bon chính thức ra mắt năm 2028.',
      [[T(36, 'Thí điểm dự kiến từ năm 2025 và triển khai đầy đủ vào năm 2028')], [V(139, 'Hình A2.4.1')]],
      'Trang 36 (tr.4) bảng gói chính sách, hàng "Đẩy nhanh chuyển đổi năng lượng sạch"; trang 139 sơ đồ mốc 2028.', 'Kết hợp bảng và sơ đồ ở hai trang.'),
    Q('t18', 'test', 'multi_hop', 'Theo kịch bản có khả năng xảy ra nhất (RCP4.5), thiệt hại GDP do biến đổi khí hậu vào năm 2050 là bao nhiêu % so với kịch bản cơ sở, và tốc độ tăng trưởng GDP giai đoạn 2040–50 của kịch bản này thấp hơn kịch bản cơ sở bao nhiêu điểm phần trăm (Bảng 1.1)?',
      'Thiệt hại 12,5% GDP vào năm 2050; tăng trưởng 2040–50 là 4,30% so với 4,56% → thấp hơn 0,26 điểm phần trăm.',
      [[T(37, 'tăng lên đến 12,5% vào năm 2050'), T(44, '12,5% vào năm 2050')], [V(44, 'Bảng 1.1')]],
      'Trang 37 (Thông điệp chính) hoặc trang 44 văn bản; Bảng 1.1 trang 44 hàng RCP4.5.', 'Kết hợp văn bản và bảng, có phép tính.'),
    Q('t19', 'test', 'multi_hop', 'Theo Bảng 3.2, tỷ lệ các ngành/lĩnh vực biển trong tổng GVA dự báo năm 2030 là bao nhiêu, và theo Bảng 3.3 tỷ lệ việc làm của các ngành này trên tổng việc làm năm 2018 là bao nhiêu?',
      '9,650% GVA năm 2030 (dự báo); 8,8% tổng việc làm năm 2018.',
      [[V(99, 'Bảng 3.2')], [V(100, 'Bảng 3.3'), T(100, 'Việt Nam chỉ chiếm 8,8%')]],
      'Trang 99 Bảng 3.2 hàng cuối cột 2030; trang 100 Bảng 3.3 hàng cuối cột 2018.', 'Kết hợp hai bảng.'),
    Q('t20', 'test', 'multi_hop', 'Theo RCP4.5, tỷ lệ thiệt hại do căng thẳng nhiệt độ cao vào năm 2050 là bao nhiêu (Hình 1.6), và các hiện tượng thời tiết cực đoan làm giảm GDP bình quân bao nhiêu mỗi năm và có thể lên đến bao nhiêu (phân tích Hình 1.7)?',
      'Căng thẳng nhiệt: 10,0% năm 2050; thời tiết cực đoan làm giảm GDP bình quân 1% mỗi năm, có thể lên đến 2,5%.',
      [[V(46, 'Hình 1.6')], [T(47, 'bình quân 1% mỗi năm'), V(47, 'Hình 1.7')]],
      'Trang 46 nhãn 10.0% cột 2050; trang 47 văn bản trên Hình 1.7.', 'Kết hợp hai trang.'),
    Q('t21', 'test', 'text', 'Trong giai đoạn 1998–2015, thành phố Đà Nẵng đã trải qua bao nhiêu cơn bão, áp thấp nhiệt đới và trận lụt?',
      '26 cơn bão, 13 áp thấp nhiệt đới và 46 trận lụt.',
      [[T(46, 'Đà Nẵng đã trải qua 26 cơn bão, 13 áp thấp nhiệt đới và 46 trận lụt')]], 'Trang PDF 46 (tr.14), đoạn đầu.', 'Tìm số liệu có điều kiện thời gian.'),
    Q('t22', 'test', 'text', 'Theo Hộp 1.2, cách tiếp cận đầu tư phối hợp vào Thủ Đức có thể đóng góp bao nhiêu cho sản lượng TP. Hồ Chí Minh mỗi năm vào năm 2040, và riêng đầu tư chống ngập có thể nâng sản lượng thêm bao nhiêu?',
      'Tới 35 tỷ US$ mỗi năm vào năm 2040 (so với kịch bản cơ sở); riêng chống ngập (bảo vệ khu tây bắc Thủ Đức) nâng sản lượng thêm 10 tỷ US$ vào năm 2040.',
      [[T(55, 'có thể lên tới 35 tỷ US$ mỗi năm vào năm 2040')]], 'Trang PDF 55 (tr.23), Hộp 1.2.', 'Tổng hợp hai số liệu có điều kiện.'),
    Q('t23', 'test', 'text', 'Theo QHĐ8 điều chỉnh, năng lượng hạt nhân dự kiến cung cấp phụ tải nền ở mức bao nhiêu MW và chiếm bao nhiêu phần trăm toàn hệ thống?',
      '10.500–14.000 MW, chiếm 1,4–1,7% toàn hệ thống.',
      [[T(122, 'Năng lượng hạt nhân dự kiến sẽ cung cấp phụ tải nền ổn định')]], 'Trang PDF 122 (tr.90), Phụ lục 1.0.', 'Tìm thông tin trong phụ lục.'),
    Q('t24', 'test', 'text', 'Siêu bão Yagi (tháng 9/2024) đã làm hư hại hoặc phá hủy bao nhiêu héc-ta lúa, rau màu và cây ăn quả, và tổn thất ước tính riêng trong ngành nông nghiệp là bao nhiêu?',
      'Khoảng 262.000 ha; tổn thất ước tính 2,44 tỷ US$ riêng trong ngành nông nghiệp.',
      [[T(41, 'khoảng 262.000 ha lúa, rau màu và cây ăn quả')]], 'Trang PDF 41 (tr.9).', 'Tìm số liệu trong văn bản.'),
    U('tu1', 'test', 'Mức thuế bảo vệ môi trường hiện hành đối với túi ni-lông là bao nhiêu VND/kg?',
      'Bảng 2.2 (tr.78) chỉ liệt kê thuế BVMT với nhiên liệu hóa thạch.', "Không trang nào chứa 'ni-lông' hoặc 'túi'."),
    U('tu2', 'test', 'Theo Bảng 3.2, tổng giá trị gia tăng của ngành đánh bắt và nuôi trồng thủy sản năm 2025 là bao nhiêu tỷ VND?',
      'Bảng 3.2 (tr.99) chỉ có các năm 2010–2020 và dự báo 2030, không có 2025.', 'Đối chiếu tiêu đề cột Bảng 3.2.'),
    U('tu3', 'test', 'Theo Hình 3.5, thành phố Đà Nẵng có bao nhiêu khu công nghiệp có nhà máy xử lý nước thải tập trung đạt chuẩn?',
      'Hình 3.5 (tr.105) chỉ gồm 7 tỉnh: Hưng Yên, Thái Bình, Ninh Bình, Thanh Hoá, Quảng Bình, Thừa Thiên Huế, Lâm Đồng; Đà Nẵng chỉ được nhắc ở văn bản về kế hoạch bảo vệ môi trường.', 'Đối chiếu trục hoành Hình 3.5.'),
    U('tu4', 'test', 'Tỷ lệ che phủ rừng (tính theo % diện tích đất) của Việt Nam năm 2023 là bao nhiêu?',
      'Báo cáo chỉ nêu diện tích rừng ngập mặn (150.000 ha năm 2020, tr.31), không có tỷ lệ che phủ rừng toàn quốc.', "'che phủ rừng' chỉ xuất hiện ở tr.31 trong cụm 'che phủ rừng ngập mặn'."),
    U('tu5', 'test', 'Theo Bảng 3.4, Sing-ga-po có bao nhiêu việc làm trong các ngành dựa vào đại dương?',
      'Bảng 3.4 (tr.101) chỉ có % GDP của Sing-ga-po (7,0); ô việc làm để trống.', 'Đối chiếu hàng Sing-ga-po trong Bảng 3.4.'),
    U('tu6', 'test', 'Giá điện bán lẻ bình quân của Việt Nam năm 2024 là bao nhiêu VND/kWh?',
      'Báo cáo không có giá điện bán lẻ (chỉ có LCOE theo $/MWh ở Hình 2.19b và kịch bản giảm giá điện).', "Không trang nào chứa 'VND/kWh' hay 'đồng/kWh'; 'bán lẻ' chỉ là tên ngành ở Hình 2.17."),
]


def validate(questions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[str]:
    problems = []
    for q in questions:
        for i, group in enumerate(q.get('gold_groups') or []):
            ok = [a for a in group if any(_alt_hit(c, a) for c in chunks)]
            if not ok:
                problems.append(f"{q['id']} group {i}: no chunk matches {group}")
            group[:] = ok or group
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    chunks = json.loads(CHUNKS_PATH.read_text(encoding='utf-8'))
    problems = validate(QUESTIONS, chunks)
    for q in QUESTIONS:
        if q['answerable']:
            q['gold'] = [a for g in q['gold_groups'] for a in g]
    (DATA / 'questions.json').write_text(json.dumps(QUESTIONS, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    (DATA / 'cases.json').write_text(json.dumps([{'id': q['id'], 'split': q['split'], 'answerable': q['answerable'],
                                                    'group': q['group']} for q in QUESTIONS], ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    from collections import Counter
    for split in ('dev', 'test'):
        print(split, dict(Counter(q['group'] for q in QUESTIONS if q['split'] == split)))
    print('validation problems:', len(problems))
    for p in problems:
        print('  ', p)


if __name__ == '__main__':
    main()
