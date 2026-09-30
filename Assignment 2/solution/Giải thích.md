Assignment 2 yêu cầu bạn **xây dựng và đánh giá một hệ thống chọn sản phẩm thời trang để đề xuất cho người dùng, đồng thời quyết định sản phẩm nằm ở vị trí nào**. Mục tiêu chính là tăng tỷ lệ click, nhưng bài còn yêu cầu phân tích tốc độ cập nhật, tham số, khác biệt giữa người dùng, fairness và thay đổi theo thời gian.

Tôi đã triển khai một quy trình Python giải quyết từng phần, chạy trên dữ liệu được cung cấp và tạo báo cáo LaTeX. Dưới đây là cách nối **yêu cầu của đề → phương pháp đã làm → kết quả và ý nghĩa**.

**Trước hết, cần hiểu bài toán và dữ liệu.** Theo [đề Assignment 2](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/2026_Assignment.pdf>), website Zozo hiển thị ba sản phẩm ở ba vị trí: trái, giữa và phải.

Mỗi dòng dữ liệu ghi lại một sản phẩm đã hiển thị:

| Biến | Ý nghĩa | Vai trò trong lời giải |
|---|---|---|
| `timestamp` | Thời điểm hiển thị | Chia dữ liệu theo thời gian, kiểm tra tính dừng |
| `item_id` | Sản phẩm được hiển thị | Hành động mà thuật toán lựa chọn |
| `position` | Vị trí 1, 2 hoặc 3 | Học hiệu quả của sản phẩm tại từng vị trí |
| `click` | Có click hay không: 1 hoặc 0 | Phần thưởng |
| `propensity_score` | Xác suất chính sách thu thập dữ liệu chọn sản phẩm tại vị trí đó | Đánh giá chính sách mới từ dữ liệu cũ |
| `user_feature_0–3` | Các đặc trưng người dùng đã mã hóa | Phân nhóm người dùng và kiểm tra fairness |

Đây là bài toán **multi-armed bandit** vì hệ thống phải cân bằng hai việc:

- **Exploitation:** ưu tiên sản phẩm đã có bằng chứng mang lại nhiều click.
- **Exploration:** thử sản phẩm còn ít thông tin, vì chúng có thể tốt hơn.

Ví dụ, sản phẩm A có CTR ước lượng cao sau nhiều lần hiển thị. Sản phẩm B mới được hiển thị ít lần nên chưa biết rõ. Chỉ chọn A có thể bỏ lỡ B; thử B quá nhiều có thể làm mất click hiện tại.

Trong bài này còn có yếu tố vị trí: sản phẩm tốt ở giữa chưa chắc tốt ở bên trái.

**Đề chia thành bảy yêu cầu, tổng cộng 10 điểm.**

| Câu | Đề yêu cầu | Điểm |
|---|---|---:|
| (a) | Triển khai hai thuật toán từ TS/UCB; mô tả và so sánh CTR với nhau và với chính sách sinh dữ liệu | 3 |
| (b) | Kiểm tra độ nhạy với batch size | 1 |
| (c) | Kiểm tra độ nhạy với tham số | 1 |
| (d) | Kiểm tra tác động của aggregation/heterogeneity | 1 |
| (e) | Đánh giá fairness với người dùng và sản phẩm; đề xuất, triển khai mở rộng | 2 |
| (f) | Dùng thực nghiệm kiểm tra dữ liệu có dừng hay không | 1 |
| (g) | Nêu thay đổi mô hình khi môi trường không dừng | 1 |

---

**Trước khi triển khai thuật toán, tôi đã kiểm tra dữ liệu và thiết kế cách đánh giá.**

File chính có:

- **1.356.670 dòng** và **4.720 click**.
- Ba vị trí, bảy ngày từ 24–30/11/2019.
- CTR toàn bộ khoảng **0,348%**.
- **79 item được quan sát**, mặc dù tên file ghi “80items”.
- Tất cả propensity score đều bằng **1/80**.

CTR ở đây được tính trên **một lượt hiển thị sản phẩm tại một vị trí**, không phải tỷ lệ người dùng click ít nhất một sản phẩm trong cả trang.

Chênh lệch 79/80 item cần xử lý rõ. Tôi giới hạn thuật toán vào 79 item có dữ liệu và đưa ra giả định:

> File được tạo bằng cách giữ lại một tập item từ log uniform 80 item, không lọc thêm theo click hoặc đặc điểm người dùng.

Nếu giả định này đúng, xác suất chọn mỗi item **sau khi điều kiện hóa trên tập giữ lại** là 1/79. Mã sử dụng giá trị này trong đánh giá chính, đồng thời xuất thêm IPS dùng propensity gốc 1/80 để đối chiếu.

**Đây vẫn là giả định cần xác minh về cách tạo dữ liệu.** Kết quả hiện tại không chứng minh hiệu quả trên toàn bộ 80 item ban đầu.

Tôi cũng chia dữ liệu theo thời gian:

| Tập | Ngày | Số dòng | Mục đích |
|---|---|---:|---|
| Train | 24–26/11 | 542.955 | Học thống kê ban đầu |
| Validation | 27–28/11 | 435.182 | Chọn tham số và làm các thí nghiệm độ nhạy |
| Test | 29–30/11 | 378.533 | Đánh giá cuối |

Sau khi chọn cấu hình bằng validation, mô hình được huấn luyện lại trên năm ngày đầu và đánh giá trên hai ngày cuối.

Cách chia này tránh việc dùng thông tin tương lai để quyết định mô hình trong quá khứ.

---

**Với câu (a), tôi triển khai Position Thompson Sampling và Position UCB.**

Điểm chung của hai phương pháp là học thống kê riêng cho **từng cặp sản phẩm–vị trí**.

Ví dụ, thay vì chỉ học một CTR cho item 35, mô hình học riêng:

- Item 35 tại vị trí trái.
- Item 35 tại vị trí giữa.
- Item 35 tại vị trí phải.

Với 79 item và ba vị trí, mô hình có **237 cặp** cần học.

Đối với **Thompson Sampling**, mỗi cặp có một phân phối mô tả xác suất click còn chưa chắc chắn:

\[
\theta_{a,p}
\sim
\operatorname{Beta}
\left(
S_{a,p}+\kappa m,\;
N_{a,p}-S_{a,p}+\kappa(1-m)
\right)
\]

Trong đó:

- \(a\): item.
- \(p\): vị trí.
- \(N_{a,p}\): số lần hiển thị.
- \(S_{a,p}\): số click.
- \(m\): CTR nền ước lượng từ dữ liệu huấn luyện.
- \(\kappa\): mức độ làm trơn về CTR nền.

Thuật toán lấy mẫu từ các phân phối này, rồi chọn cách bố trí có tổng score cao nhất.

Ý nghĩa trực quan: item đã có nhiều bằng chứng tốt thường được chọn; item chưa rõ chất lượng vẫn có cơ hội được thử do độ bất định lớn.

Tôi dùng prior dựa trên CTR nền vì dữ liệu có CTR rất thấp. Một prior có trung bình 50% sẽ không phản ánh tốt bối cảnh này. Đây là cách làm *empirical Bayes*: thông tin nền được ước lượng từ dữ liệu huấn luyện.

Đối với **UCB**, thuật toán dùng:

\[
U_{a,p}
=
\frac{S_{a,p}+\kappa m}{N_{a,p}+\kappa}
+
c\sqrt{
\frac{\log(2+\sum_{a,p}N_{a,p})}
{N_{a,p}+1}
}
\]

Score gồm hai thành phần:

1. CTR đã được làm trơn.
2. Phần thưởng khám phá cho cặp item–vị trí còn ít quan sát.

\(c\) càng lớn thì thuật toán càng coi trọng khám phá.

Sau khi có score, cả hai phương pháp đều giải bài toán:

\[
\max_{a_1,a_2,a_3\text{ khác nhau}}
\sum_{p=1}^{3}\operatorname{score}(a_p,p)
\]

Điều kiện “khác nhau” ngăn một sản phẩm xuất hiện ở cả ba vị trí. Phần này nằm trong hàm `assign_slate` của [bandits.py](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/bandits.py:26>).

Ví dụ, slate tối ưu theo **posterior mean** của Position TS sau khi học năm ngày đầu là:

| Vị trí | Item |
|---|---:|
| Trái | 35 |
| Giữa | 60 |
| Phải | 58 |

Đây là bố trí minh họa theo giá trị trung bình. Chính sách TS được đánh giá vẫn ngẫu nhiên, nên không luôn chọn đúng ba item này.

Cả hai phương pháp còn có 5% xác suất chọn một slate ngẫu nhiên để duy trì khám phá.

**Phần khó của câu (a) là đánh giá CTR từ log.** Một dòng chỉ cho biết click của item đã hiển thị. Nếu thuật toán mới muốn chọn item khác, dữ liệu không cho biết người dùng có click item đó hay không.

Vì vậy, tôi không gán reward giả cho những lựa chọn chưa quan sát. Tôi triển khai ba thước đo:

| Thước đo | Cách hiểu |
|---|---|
| IPS | Điều chỉnh reward bằng tỷ lệ xác suất chọn action giữa policy mới và logger |
| SNIPS | Chuẩn hóa tổng trọng số của IPS |
| DR | Kết hợp dự đoán reward với phần hiệu chỉnh bằng propensity |

Ví dụ, IPS dùng trọng số:

\[
w_i=
\frac{\pi(a_i\mid x_i,p_i)}
{b(a_i\mid x_i,p_i)}
\]

và tính:

\[
\widehat{CTR}_{IPS}
=
\frac{1}{n}\sum_i w_i r_i
\]

Tôi dùng **DR làm thước đo chính**, còn IPS và SNIPS để kiểm tra chéo.

Kết quả trên test:

| Chính sách | CTR ước lượng bằng DR |
|---|---:|
| Logger trên cùng dữ liệu test | 0,361% |
| TS gộp theo item | 0,677% |
| Position TS | 0,585% |
| Position UCB | 0,488% |
| Fair TS | 0,504% |

Có hai điểm cần diễn giải đúng:

- Position TS có bằng chứng cải thiện so với logger theo khoảng tin cậy đã tính.
- Chênh lệch Position TS–Position UCB là **0,097 điểm phần trăm**, nhưng khoảng tin cậy 95% là **[-0,026; 0,207] điểm phần trăm**, nên chưa đủ bằng chứng khẳng định TS tốt hơn UCB.

Khoảng tin cậy được tính bằng bootstrap theo block một giờ, giữ nguyên mô hình đã huấn luyện. Nó chưa bao gồm toàn bộ bất định do huấn luyện lại hoặc tương quan giữa các lượt truy cập của cùng người dùng.

Ngoài ra, `Pooled TS` là baseline tôi triển khai, **không phải bản phục dựng chính xác TS production của Zozo**. So sánh với phương pháp sinh file chính được thực hiện bằng CTR logger trên cùng test.

Kết quả câu (a) nằm ở **Tables 3–4 và Figure 1**.

---

**Với câu (b), tôi kiểm tra thuật toán thay đổi thế nào khi cập nhật theo các batch khác nhau.**

Batch size trả lời câu hỏi:

> Thu thập bao nhiêu quan sát rồi mới cập nhật kiến thức và quyết định của thuật toán?

Batch nhỏ cho phép cập nhật thường xuyên hơn. Batch lớn làm mô hình sử dụng thông tin cũ lâu hơn.

Tôi triển khai replay theo thứ tự thời gian:

1. Tạo policy từ thông tin đã có trước batch.
2. Duyệt các dòng log trong batch.
3. Chỉ nhận phản hồi khi action policy khớp action đã ghi trong log.
4. Cập nhật mô hình ở cuối batch.

Với log uniform trên 79 item, trung bình chỉ khoảng 1/79 số dòng được nhận làm phản hồi học.

Tôi thử ba batch size và năm seed:

| Batch size, tính bằng dòng log | TS: IPS CTR trung bình | UCB: IPS CTR trung bình |
|---:|---:|---:|
| 500 | 0,420% | 0,310% |
| 5.000 | 0,396% | 0,414% |
| 50.000 | 0,403% | 0,310% |

**Batch ở đây là số dòng log, không phải số phản hồi được chấp nhận hay số người dùng.** Batch 5.000 dòng tương ứng trung bình khoảng 63 phản hồi khớp nếu giả định uniform đúng.

Kết quả không cho thấy “batch càng nhỏ luôn càng tốt”. Click hiếm và phản hồi được nhận ít khiến kết quả có biến động đáng kể.

Thí nghiệm này nằm ở **Table 5, Figures 2 và 7**. Các thanh sai số biểu diễn độ lệch chuẩn giữa seed, không phải khoảng tin cậy 95%.

Một phân biệt quan trọng: **câu (a) đánh giá policy cố định trên test; câu (b) đánh giá quá trình học thích nghi bằng replay trên validation**.

---

**Với câu (c), tôi kiểm tra độ nhạy với tham số thay vì chọn một giá trị tùy ý.**

Đối với TS, tham số được khảo sát là prior strength:

\[
\kappa\in\{10,100,1000\}
\]

\(\kappa\) lớn khiến các CTR ít dữ liệu được kéo mạnh hơn về mức nền. Điều này có thể hữu ích khi mỗi cặp chỉ có ít click.

Kết quả DR trên validation:

- \(\kappa=10\): khoảng **0,382%**.
- \(\kappa=100\): khoảng **0,388%**.
- \(\kappa=1000\): khoảng **0,406%**.

Do đó, cấu hình được chọn là **\(\kappa=1000\)**.

Đối với UCB, tôi thử:

\[
c\in\{0{,}001,\;0{,}01,\;0{,}1,\;1\}
\]

Giá trị được chọn là **\(c=1\)** vì có DR validation cao nhất trong lưới đã thử, khoảng **0,402%**.

Điều này chỉ có nghĩa là cấu hình tốt nhất **trong các giá trị đã khảo sát và giai đoạn validation này**. Nó không chứng minh đây là giá trị tối ưu cho mọi dữ liệu.

Kết quả nằm ở **Table 6 và Figure 3**. Test không được dùng để chọn các tham số này.

---

**Với câu (d), tôi kiểm tra xem chia mô hình chi tiết hơn có thực sự giúp ích hay không.**

“Aggregation/heterogeneity” có thể hiểu là chọn mức độ gộp dữ liệu:

| Cấu trúc | Điều mô hình phân biệt |
|---|---|
| Pooled | Chỉ phân biệt item |
| Position | Phân biệt item và vị trí |
| Segment | Phân biệt item, vị trí và nhóm người dùng |

Ví dụ, mô hình pooled giả định một item có chung chất lượng ở ba vị trí. Mô hình position cho phép item đó tốt hơn ở giữa. Mô hình segment còn cho phép các nhóm người dùng thích những item khác nhau.

Tôi dùng `user_feature_0` cho phép so sánh segment ở phần này. Những nhóm có ít dữ liệu được làm trơn về ước lượng global item–vị trí.

Kết quả TS trên validation:

| Cấu trúc | DR CTR |
|---|---:|
| Item only | 0,481% |
| Item–position | 0,406% |
| Item–position–group | 0,365% |

Trong lần chạy này, mô hình chi tiết hơn **không cải thiện CTR ước lượng**.

Một cách giải thích hợp lý là: chia dữ liệu thành nhiều ô làm mỗi ô có ít click hơn, tăng độ bất định. Đây là diễn giải phù hợp với kết quả, chưa phải chứng minh nguyên nhân duy nhất.

Tôi cũng chạy cùng phép so sánh cho UCB; mô hình pooled có biến động đáng kể giữa seed, một phần liên quan đến cách phá hòa khi score không phân biệt vị trí.

Kết quả nằm ở **Table 7 và Figure 4**.

---

**Với câu (e), tôi tách fairness đối với người dùng và fairness đối với sản phẩm.**

Đối với **người dùng**, câu hỏi là:

> Chính sách có giúp một số nhóm nhưng làm nhóm khác nhận đề xuất kém hơn không?

Tôi kiểm tra riêng từng nhóm của cả bốn `user_feature`, vì đề cho phép coi cả bốn là biến nhạy cảm khi chưa biết chính xác biến nào được bảo vệ.

Mỗi nhóm có:

- CTR logger.
- CTR ước lượng bằng DR và SNIPS.
- Effective sample size, tức mức thông tin hiệu dụng sau weighting.
- Uplift so với logger của chính nhóm đó.

Uplift được tính:

\[
uplift_g=
\frac{\widehat{CTR}_{policy,g}}
{CTR_{logger,g}}-1
\]

So với logger trong cùng nhóm giúp tránh đánh đồng khác biệt về mức quan tâm nền với tác động của policy.

Để nhóm quá nhỏ không chi phối tóm tắt, tôi dùng ngưỡng ít nhất **7.900 dòng và 10 click logger**. Cùng một tập nhóm đủ điều kiện được dùng cho mọi policy; nhóm nhỏ vẫn được xuất đầy đủ trong Table 9.

Đối với **sản phẩm**, câu hỏi là:

> Thuật toán có tập trung gần như toàn bộ lượt hiển thị vào vài item, khiến các item khác hầu như không có cơ hội xuất hiện không?

Tôi đo:

- Gini của exposure.
- Entropy chuẩn hóa.
- Exposure share thấp nhất và cao nhất.
- Tỷ lệ item có xác suất hiển thị dương.

Exposure được tính từ xác suất policy trên toàn bộ context đánh giá, không chỉ từ những dòng replay khớp.

**Mở rộng đã triển khai là Fair TS**, kết hợp TS phân nhóm với nhánh chọn ngẫu nhiên:

\[
\pi_{fair}
=
(1-\epsilon)\pi_{segment}
+
\epsilon\pi_{uniform}
\]

Tôi thử bốn biến phân nhóm và:

\[
\epsilon\in\{0{,}10,\;0{,}25,\;0{,}50\}
\]

Trên validation, trước hết giữ những cấu hình có DR đạt ít nhất 95% Position TS, sau đó chọn cấu hình có uplift của nhóm kém nhất cao nhất.

Cấu hình được chọn dùng:

- `user_feature_1`.
- \(\epsilon=0,50\).

Kết quả test:

| Thước đo | Position TS | Fair TS |
|---|---:|---:|
| DR CTR tổng thể | 0,585% | 0,504% |
| Gini exposure | 0,870 | 0,419 |
| Exposure share thấp nhất | 0,0633% | 0,6330% |

Fair TS phân bổ cơ hội hiển thị đều hơn, đồng thời có CTR tổng thể thấp hơn ở point estimate.

Nhánh uniform bảo đảm mỗi item có xác suất tối thiểu \(\epsilon/79\) tại mỗi vị trí. **Bảo đảm này áp dụng cho exposure của item; nó không bảo đảm mọi nhóm người dùng đều được hưởng lợi.**

Các kết quả theo nhóm hiện còn nhiễu. Một số DR ước lượng âm dẫn đến uplift dưới -100%; đó là biểu hiện bất ổn của estimator trong mẫu hữu hạn, không phải CTR thật có thể âm. Vì vậy cần đọc cùng SNIPS và ESS, và chưa thể kết luận hệ thống “công bằng với mọi người dùng”.

Phần này nằm ở **Tables 8–9, 16–17 và Figure 5**.

---

**Với câu (f), tôi dùng kiểm định để tìm bằng chứng về thay đổi theo thời gian.**

“Tính dừng” trong phạm vi phân tích này liên quan đến câu hỏi:

> Xác suất click có giữ nguyên qua các ngày không, sau khi xét các yếu tố đang kiểm soát?

Chỉ nhìn biểu đồ CTR lên xuống chưa đủ, vì biến động có thể do ngẫu nhiên hoặc do thành phần người dùng thay đổi.

Tôi thực hiện:

1. Thống kê CTR theo ngày.
2. Kiểm định CTR toàn bộ có thay đổi giữa các ngày.
3. Kiểm định sau khi giữ cố định item và position.
4. Kiểm định sau khi xét thêm `user_feature_0`.
5. Kiểm tra phân phối của từng user feature có thay đổi theo ngày.

Ba kiểm định reward dùng parametric bootstrap với 1.000 mẫu để xử lý tình trạng nhiều ô có ít click. Tôi điều chỉnh Benjamini–Hochberg cho bảy kiểm định.

Kết quả đều bác bỏ giả thuyết ổn định được kiểm tra ở mức 5%; các kiểm định reward có p-value khoảng 0,001, là giới hạn phân giải của số lần bootstrap đã dùng.

Diễn giải phù hợp là:

> Có bằng chứng chống lại giả thuyết reward và thành phần người dùng ổn định trong tuần quan sát, dưới các giả định của kiểm định.

Không nên diễn giải thành “đã chứng minh mọi khía cạnh của dữ liệu đều không dừng”. Thiếu user ID khiến tương quan giữa các lượt truy cập chưa được xử lý đầy đủ.

Kết quả nằm ở **Tables 10–11 và Figure 6**.

---

**Với câu (g), đề chỉ yêu cầu nêu thay đổi mô hình; tôi đã triển khai thêm một phương án để minh họa.**

Khi sở thích thay đổi, dữ liệu cũ có thể không còn phản ánh hiện tại. Tôi thêm cơ chế giảm trọng số lịch sử:

\[
N\leftarrow\gamma N+N_{batch}
\]

\[
S\leftarrow\gamma S+S_{batch}
\]

Trong đó:

- \(\gamma=1\): giữ nguyên ảnh hưởng lịch sử.
- \(\gamma<1\): giảm dần ảnh hưởng dữ liệu cũ.
- \(\gamma\) càng nhỏ: quên nhanh hơn.

Tôi thử \(\gamma=1,\;0,99,\;0,95\), giữ batch size ở 5.000 dòng log.

Với TS, IPS CTR trung bình trên validation lần lượt khoảng:

- **0,396%** khi không discount.
- **0,424%** với \(\gamma=0,99\).
- **0,438%** với \(\gamma=0,95\).

Tuy nhiên, UCB không cải thiện tương tự trong thí nghiệm này. Do đó, không thể kết luận cứ phát hiện drift thì discount sẽ tốt hơn.

Tôi cũng trình bày các hướng khác trong báo cáo: sliding window, phát hiện điểm thay đổi, thêm đặc trưng thời gian và đánh giá cuốn chiếu theo thời gian.

Kết quả triển khai discount nằm ở **Table 15**.

---

**Về yêu cầu nộp bài, tôi đã chuẩn bị mã và tài liệu có thể tái lập.**

[Guidelines](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/Guidelines for coding and submission of Assignment 2.pdf>) yêu cầu dùng một ngôn ngữ lập trình, kiểm soát randomness, đánh số bảng/hình và nộp báo cáo PDF tạo bằng LaTeX.

Bộ lời giải hiện có:

| Thành phần | Nội dung |
|---|---|
| [bandits.py](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/bandits.py>) | Thuật toán, chọn slate, OPE, replay |
| [diagnostics.py](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/diagnostics.py>) | Kiểm tra dữ liệu, fairness, stationarity |
| [run_assignment.py](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/run_assignment.py>) | Chạy toàn bộ thí nghiệm, xuất bảng và hình |
| [Hướng dẫn tiếng Việt](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/HUONG_DAN.md>) | Công thức, giải thích và hướng dẫn chạy |
| [Báo cáo PDF](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/Assignment_2_STUDENTNUMBER.pdf>) | Bản nháp 10 trang từ kết quả thực chạy |
| [report.tex](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/report.tex>) | Nguồn LaTeX để bạn sửa báo cáo |

Tôi đã chạy 10 kiểm tra tính đúng đắn và hai lần chạy toàn bộ dữ liệu. **17 bảng CSV, 14 đoạn LaTeX và 7 hình PNG giống nhau từng byte giữa hai lần chạy.** Bảng được in với identifier; hình được lưu PNG/PDF và có tùy chọn `--show` để mở xem.

Bạn nên đọc **hướng dẫn tiếng Việt song song với báo cáo PDF**, rồi mở hàm tương ứng trong mã cho từng câu. Trước khi nộp, cần thay mã sinh viên, kiểm tra giả định 79/80 item và chỉnh phần diễn giải theo cách hiểu của bạn.





**Chưa nên nộp nguyên trạng bản hiện tại.** Mã đã chạy được, kết quả đã kiểm tra và báo cáo đã có đủ các phần (a)–(g), nhưng PDF vẫn ghi **`STUDENTNUMBER`** và **“Working draft”**. Bạn cần hoàn thiện bản nộp cuối.

**Những việc cần sửa trước khi nộp:**

1. **Điền mã sinh viên trong [report.tex](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/report.tex:10>).**

   Thay:
   ```latex
   \newcommand{\studentnumber}{STUDENTNUMBER}
   ```
   bằng mã thật, ví dụ:
   ```latex
   \newcommand{\studentnumber}{123456}
   ```

   Sau đó biên dịch lại thành `Assignment_2_123456.pdf`. **Chỉ đổi tên PDF chưa đủ**, vì mã sinh viên bên trong vẫn chưa thay.

2. **Xóa lời nhắc dành cho bản nháp.**

   Trong `report.tex`, xóa câu:
   ```latex
   \textbf{Working draft.} Replace the student number and review the interpretation before submission.
   ```

   Có thể đổi tiêu đề `Automatically generated findings` thành `Results`. Giữ phần giải thích rằng bảng và hình được sinh từ mã để tái lập.

3. **Đọc và hoàn thiện phần diễn giải bằng cách hiểu của bạn.**

   Đặc biệt, báo cáo nên nói rõ:

   - TS có CTR ước lượng cao hơn UCB, nhưng **chưa đủ bằng chứng thống kê để khẳng định TS tốt hơn UCB**.
   - Mô hình phân nhóm chi tiết hơn không cải thiện CTR trong thí nghiệm hiện tại.
   - Fair TS cải thiện độ đồng đều của exposure, nhưng **không bảo đảm mọi nhóm người dùng đều hưởng lợi**.
   - Batch nhỏ hơn và discount mạnh hơn không phải lúc nào cũng tốt hơn.

   Các bảng đã có số liệu; phần bạn cần chú trọng là giải thích **kết quả đó có nghĩa gì và giới hạn ở đâu**.

4. **Giữ nguyên phần giải thích về 79/80 item.**

   File chỉ có 79 item nhưng propensity là 1/80. Báo cáo hiện đã công khai giả định xử lý. Bạn nên xác minh cách lọc dữ liệu với giảng viên nếu có thể; không nên xóa đoạn này hoặc trình bày giả định như sự thật đã được xác nhận.

5. **Kiểm tra yêu cầu trên Canvas rồi biên dịch PDF cuối.**

   Hai PDF hướng dẫn bạn cung cấp yêu cầu báo cáo LaTeX, tên file đúng, code tái lập và bảng/hình có identifier. Chúng **không nêu rõ định dạng gói code, giới hạn trang hay các yêu cầu khai báo công cụ hỗ trợ**; những điểm đó cần đối chiếu Canvas.

**Về file cần nộp, tôi đề xuất bộ sau.** Tên PDF là yêu cầu rõ trong hướng dẫn; tên ZIP code dưới đây là cách đóng gói đề xuất.

| File | Nội dung |
|---|---|
| `Assignment_2_MASINHVIEN.pdf` | Báo cáo cuối sau khi sửa và biên dịch |
| `Assignment_2_MASINHVIEN_code.zip` | Mã Python, dependencies và hướng dẫn chạy |
| Nguồn LaTeX | Nộp thêm nếu Canvas/giảng viên yêu cầu |

Trong ZIP code nên có:

```text
bandits.py
diagnostics.py
run_assignment.py
test_assignment.py
requirements.txt
requirements-lock.txt
README_RUN.txt
outputs/
    tables/
    figures/
    console.txt
    manifest.json
verification.txt
```

`README_RUN.txt` nên ghi vị trí dữ liệu, phiên bản Python và lệnh chạy, chẳng hạn:

```powershell
python -m pip install -r requirements.txt
python run_assignment.py --data "DUONG_DAN/zozo_Context_80items.csv" --show
```

Khi đóng gói riêng, cần thử chạy từ thư mục đã giải nén để chắc chắn đường dẫn hoạt động. Hiện mã đã được kiểm tra trong cấu trúc repository, **chưa có ZIP code nộp bài được đóng gói và kiểm tra riêng**.

**Không nên nộp cả thư mục `solution` mà chưa lọc file.** Một số file không cần đưa vào bộ nộp:

- `.aux`, `.log` của LaTeX và `report_preview.png`: file phục vụ biên dịch/kiểm tra.
- `HUONG_DAN.md`, `Giải thích.md`: tài liệu học của bạn, trừ khi muốn bổ sung.
- `zozo_Context_80items.csv`: file khoảng 269 MB; chỉ kèm nếu giảng viên yêu cầu nộp lại dữ liệu.
- Các PDF đề bài và hướng dẫn gốc.


**Có. Tôi đã chạy output thực tế và vừa kiểm tra lại độ khớp với đề. Kết quả hiện có đủ các phần (a)–(g), nhưng cần phân biệt “phép tính đã kiểm tra đúng” với “mọi kết luận đều chắc chắn”.**

Trước đó, tôi đã chạy toàn bộ dữ liệu **hai lần**: 17 bảng CSV, 14 đoạn LaTeX và 7 hình PNG giống nhau từng byte. Lần kiểm tra này, tôi tiếp tục:

- Chạy lại **10 kiểm tra tự động**: tất cả đạt.
- Tính lại IPS, SNIPS, DR và ESS bằng phép tính riêng, không gọi hàm đánh giá ban đầu: **khớp Table 3**.
- Kiểm tra tổng số dòng/click theo ngày và theo tập train–validation–test.
- Kiểm tra mỗi slate có ba item khác nhau, xác suất hợp lệ và tổng exposure bằng 1.
- Kiểm tra bảng/hình được dẫn trong báo cáo đều có file output tương ứng.

**Đối chiếu từng yêu cầu của đề:**

| Yêu cầu | Output hiện có | Đánh giá |
|---|---|---|
| **(a)** Hai thuật toán TS/UCB, chọn item và vị trí, so sánh CTR | Tables 3–4, 12; Figure 1 | Đã triển khai và có so sánh với logger |
| **(b)** Độ nhạy batch size | Table 5; Figures 2, 7 | Đã thử 500, 5.000, 50.000 dòng log |
| **(c)** Độ nhạy tham số | Table 6; Figure 3 | Đã thử nhiều giá trị và chọn bằng validation |
| **(d)** Aggregation/heterogeneity | Table 7; Figure 4 | Đã so sánh item-only, item–position và phân nhóm |
| **(e)** Fairness người dùng/item và mở rộng | Tables 8–9, 16–17; Figure 5 | Đã đo và triển khai Fair TS; kết luận về người dùng còn hạn chế |
| **(f)** Kiểm tra tính dừng | Tables 10–11; Figure 6 | Có kiểm định thực nghiệm, không chỉ nhìn biểu đồ |
| **(g)** Thay đổi khi không dừng | Table 15 và phần giải thích | Có đề xuất, đồng thời triển khai discount để minh họa |

**Có bốn điểm cần giữ rõ trong bản nộp:**

1. **Giả định 79/80 item chưa được xác minh.**  
   File chứa 79 item nhưng propensity là 1/80. Phép đánh giá chính dùng 1/79 dựa trên giả định lọc theo item từ log uniform. Các phép tính đã khớp, nhưng tính hợp lệ của giả định phụ thuộc cách tạo file.

2. **Hai loại thí nghiệm có ý nghĩa khác nhau.**  
   So sánh chính trên test dùng policy đã huấn luyện và giữ cố định. Thí nghiệm batch dùng replay thích nghi trên validation. Không nên trình bày chúng như cùng một thí nghiệm online.

3. **Fairness với item được cải thiện; fairness với mọi người dùng chưa được chứng minh.**  
   Gini exposure giảm từ khoảng **0,870 xuống 0,419**. Tuy nhiên, một số ước lượng DR theo nhóm rất nhiễu, thậm chí âm. Đây không phải xác suất click thật âm và không được dùng để khẳng định chắc chắn mức thiệt hại của nhóm.

4. **Mặc định chương trình lưu hình, chưa tự mở hình.**  
   Bảng được in ra console; hình được lưu PNG/PDF và in identifier. Để thể hiện rõ yêu cầu bảng/hình khi chạy, dùng:
   ```powershell
   python "assignments/Assignment 2/solution/run_assignment.py" --show
   ```

**Đánh giá của tôi: phần triển khai đã bao phủ đề và các phép tính chính đã qua kiểm tra; bản PDF vẫn cần hoàn thiện trước khi nộp.** Ngoài thay mã sinh viên và xóa “Working draft”, cần giữ đúng các giới hạn trên và diễn giải kết quả bằng lời của bạn. Tôi không thể bảo đảm điểm số chỉ từ việc chương trình chạy thành công.

Bạn có thể đối chiếu trực tiếp với [báo cáo PDF](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/Assignment_2_STUDENTNUMBER.pdf>) và [toàn bộ bảng output](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/outputs/console.txt>).

Lưu ý **[Assignment_2_Overleaf.zip](<D:/Learning-from-Big-Data-main/Learning-from-Big-Data-main/assignments/Assignment 2/solution/Assignment_2_Overleaf.zip>) chỉ chứa nguồn báo cáo cùng bảng/hình để biên dịch trên Overleaf; nó không thay thế ZIP mã Python.**

Nếu chỉ sửa mã sinh viên và câu chữ, bạn không cần chạy lại toàn bộ thí nghiệm; chỉ cần biên dịch lại PDF. Nếu thay thuật toán, tham số hoặc dữ liệu, cần chạy lại để báo cáo và kết quả khớp nhau.
