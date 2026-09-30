# Hướng dẫn triển khai Assignment 2 về Multi Armed Bandits

Bộ mã này giúp bạn làm đủ các câu (a)–(g) của đề năm 2026 bằng Python, đọc kết quả và viết báo cáo LaTeX. Hãy bắt đầu bằng việc chạy mã, đọc phần phương pháp bên dưới, rồi chỉnh phần thảo luận trong `report.tex` bằng cách diễn đạt của bạn. Báo cáo hiện là một bản nháp học tập có số liệu được sinh từ mã, chưa có mã sinh viên của bạn.

Bạn có thể xem ngay [báo cáo PDF đã biên dịch](Assignment_2_STUDENTNUMBER.pdf) hoặc tải [gói LaTeX cho Overleaf](Assignment_2_Overleaf.zip). Kết quả đã được kiểm tra bằng hai lần chạy toàn bộ dữ liệu; 17 bảng CSV, 14 đoạn LaTeX và 7 hình PNG khớp nhau từng byte. Xem `verification.txt`.

## Chạy chương trình

Mở PowerShell ở thư mục gốc của repository:

```powershell
python -m pip install -r "assignments/Assignment 2/solution/requirements.txt"
python "assignments/Assignment 2/solution/run_assignment.py"
```

Chương trình đọc trực tiếp file ZIP nên không cần giải nén. Đường dẫn mặc định được tính từ vị trí file mã, không phụ thuộc working directory. Bạn cũng có thể chỉ định `--data "đường/dẫn/file.csv.zip"` và `--output "thư/mục/kết/quả"`. Loader sẽ từ chối log có propensity thay đổi vì replay hiện tại chỉ hỗ trợ log uniform.

Muốn chạy thử nhanh hoặc xem hình ngay:

```powershell
python "assignments/Assignment 2/solution/run_assignment.py" --quick
python "assignments/Assignment 2/solution/run_assignment.py" --show
python -m unittest discover -s "assignments/Assignment 2/solution" -v
```

`--quick` lấy ngẫu nhiên 20.000 dòng mỗi ngày, giữ thứ tự thời gian sau lấy mẫu và ghi vào `outputs_quick`. Không dùng kết quả quick trong báo cáo cuối. Lệnh mặc định dùng toàn bộ dữ liệu và ghi vào `outputs`.

Các thư viện đã dùng trong lần chạy được lưu ở `outputs/manifest.json`; `requirements-lock.txt` ghi phiên bản của môi trường đã kiểm tra. Chương trình dùng `random.seed(1)`, `np.random.seed(1)` và các RNG cục bộ có seed xác định. Lặp lại trong cùng môi trường sẽ tái tạo các bảng. `--seed` cho phép kiểm tra độ ổn định dưới seed khác.

## Các file cần đọc

| File | Mục đích |
|---|---|
| `bandits.py` | TS, UCB, chọn slate, IPS/SNIPS/DR, replay và bootstrap |
| `diagnostics.py` | Kiểm tra dữ liệu, fairness, kiểm định tính dừng |
| `run_assignment.py` | Toàn bộ thí nghiệm, bảng, hình và số liệu cho LaTeX |
| `test_assignment.py` | Kiểm tra các tính chất quan trọng của thuật toán |
| `outputs/console.txt` | Toàn bộ bảng được in với identifier |
| `outputs/tables/Table_XX.csv` | Số liệu đầy đủ, chính xác, có thể kiểm tra lại |
| `outputs/figures/Figure_XX.pdf` | Hình vector để chèn vào báo cáo |
| `outputs/results_summary.txt` | Tóm tắt được sinh từ kết quả thực chạy |
| `report.tex` | Báo cáo tiếng Anh theo các câu hỏi của đề |

## Đọc dữ liệu đúng trước khi làm bandit

Nguồn chính là `data/assignment 2/zozo_Context_80items.zip`. Trong bản dữ liệu đã kiểm tra:

- Có **1.356.670 dòng**, **4.720 click**, **79 item** được quan sát, với ID từ 1 đến 79.
- Có ba vị trí và bảy ngày, từ 24 đến 30 tháng 11 năm 2019, theo UTC.
- Propensity score của mọi dòng là **0,0125 = 1/80**.
- CTR toàn bộ file khoảng **0,3479%**. Đây là CTR trên một slot hiển thị, không phải xác suất người dùng click ít nhất một item trong cả trang.
- Có 162 dòng trùng toàn bộ trường sau khi bỏ cột số thứ tự. Mã giữ lại vì không có impression ID/user ID để chứng minh chúng là bản ghi lỗi. Cần nêu hạn chế này.
- Các `user_feature_0` đến `user_feature_3` là biến phân loại đã hash. Không được diễn giải thứ tự hash thành thứ tự tuổi hay suy đoán danh tính/giới tính thật.

Tên file “80items” không có nghĩa là file chứa đủ 80 item. Mã không tạo thêm dữ liệu cho item vắng mặt.

### Giả định về propensity

Phân tích chính giới hạn candidate set ở 79 item quan sát được. **Giả định** file được tạo bằng cách lọc theo item từ log uniform 80 item, không lọc thêm theo reward, context hay thời điểm. Khi đó:

\[
P(A=a\mid A\text{ thuộc tập giữ lại},x,p)=1/79.
\]

Đây là giả định cần xác minh với giảng viên hoặc quy trình tạo file, không phải điều đã được chứng minh từ CSV. Nếu việc lọc khác giả định này, kết quả OPE có thể sai. Table 3 vẫn in `IPS_original_pscore` dùng 1/80 để bạn thấy độ nhạy: con số này bằng IPS điều kiện nhân 80/79. SNIPS không thay đổi khi mọi trọng số cùng nhân một hằng số. Việc SNIPS ổn định không tự chứng minh phép lọc là hợp lệ.

Không thay `propensity_score` bằng tần suất item quan sát được mà không có lập luận. Không dùng file BTS 10/20 item để đối chiếu CTR thô với file 79 item: khác tập hành động và cơ chế chọn mẫu. Đối thủ đúng cho câu (a) là logging policy của **chính file đang đánh giá**, cùng các dòng test.

## Bài toán cần tối ưu

Với mỗi context người dùng, chọn một slate gồm ba item khác nhau, gán vào ba vị trí. Gọi \(q(a,p,x)\) là xác suất click của item \(a\) ở vị trí \(p\). Theo giả định phần thưởng từng slot không phụ thuộc các item cạnh nó:

\[
\max_{a_1,a_2,a_3\;\text{khác nhau}}\sum_{p=1}^{3}q(a_p,p,x).
\]

`assign_slate` dùng bài toán gán tuyến tính của SciPy để tối ưu tổng score và không lặp item. Tối ưu tổng click của ba slot cũng tối ưu CTR trung bình khi ba slot được coi trọng như nhau. Chỉ lấy item tốt nhất cho từng vị trí độc lập có thể hiển thị trùng item nên không đủ.

Log không có khóa đáng tin cậy để khôi phục từng page view. Phân tích đánh giá xác suất chọn item tại vị trí đã ghi nhận, không ghép ba dòng gần nhau thành một impression. Khi triển khai, policy vẫn sinh ra slate hợp lệ. Nếu có tương tác giữa các item hoặc người dùng, đánh giá marginal từng slot chưa đủ; cần log slate đầy đủ và phương pháp slate OPE.

## Câu a Hai thuật toán và cách đánh giá CTR

### Position Thompson Sampling

Mỗi cặp item–vị trí có số click \(S_{a,p}\), số lần hiển thị \(N_{a,p}\). Ước lượng CTR toàn bộ warm-start bằng \(m=(S+1)/(N+2)\). Chọn prior strength \(\kappa\):

\[
\theta_{a,p}\mid D\sim\operatorname{Beta}
(\kappa m+S_{a,p},\;\kappa(1-m)+N_{a,p}-S_{a,p}).
\]

Lấy mẫu một ma trận \(\theta\), chọn slate tối đa tổng mẫu. Prior được căn theo CTR thấp của dữ liệu, thay vì mặc định Beta(1,1) với trung bình 50%. \(\kappa\) càng lớn thì càng co ước lượng về trung bình chung. Đây là empirical Bayes, nên không diễn giải nó như prior độc lập với dữ liệu.

Thêm hỗn hợp khám phá \(\epsilon=0,05\): với xác suất \(\epsilon\), hiển thị một slate uniform gồm ba item khác nhau. Với xác suất còn lại, dùng slate do thuật toán chọn.

Trong test tĩnh, mã sinh 256 slate cho mỗi seed, với 5 seed. Policy test được **định nghĩa** là hỗn hợp đều của 1.280 slate này và nhánh uniform. Do đó xác suất đánh giá được biết chính xác cho policy hữu hạn đã xây dựng; nó chỉ xấp xỉ policy TS lý tưởng tích phân trên toàn bộ posterior.

### Position UCB

Giữ thống kê riêng cho item–vị trí, dùng score:

\[
U_{a,p}=\frac{S_{a,p}+\kappa m}{N_{a,p}+\kappa}
+c\sqrt{\frac{\log(2+\sum_{a,p}N_{a,p})}{N_{a,p}+1}}.
\]

Sau đó giải cùng bài toán gán slate và thêm nhánh uniform như TS. \(c\) điều chỉnh khám phá. Đây là biến thể UCB có làm trơn; không tuyên bố nó giữ nguyên mọi bảo đảm regret của UCB1 cổ điển.

### Chia thời gian

| Phần | Ngày UTC | Cách dùng |
|---|---|---|
| Train | 24–26/11 | Fit policy, prior và reward model để đánh giá validation |
| Validation | 27–28/11 | Chọn tham số, khảo sát batch, cấu trúc và fairness |
| Test | 29–30/11 | Đánh giá cuối sau khi khóa cấu hình |

Sau chọn tham số, fit lại trên 24–28/11 rồi giữ policy cố định trong test. Không chọn mô hình theo Table 3 sau khi đã xem test; nếu làm vậy cần một test mới. Batch replay là thí nghiệm học thích nghi riêng ở validation.

### Vì sao không lấy click của bất kỳ item nào mình muốn

Một dòng log chỉ cho biết reward của item thực sự đã được hiển thị. Nếu policy muốn item khác, reward đó chưa được quan sát. Không được gán 0 cho action khác, cũng không được lấy CTR toàn bộ bảy ngày làm môi trường rồi gọi kết quả là đánh giá độc lập.

Với \(\pi_i=\pi(a_i\mid x_i,p_i)\), \(b_i\) là propensity của logger và \(w_i=\pi_i/b_i\):

\[
\widehat V_{IPS}=\frac1n\sum_iw_ir_i,
\qquad
\widehat V_{SNIPS}=\frac{\sum_iw_ir_i}{\sum_iw_i}.
\]

DR dùng reward model \(\hat q(a,p)\) được fit hoàn toàn trước test:

\[
\widehat V_{DR}=\frac1n\sum_i\left[
\sum_a\pi(a\mid x_i,p_i)\hat q(a,p_i)
+w_i(r_i-\hat q(a_i,p_i))\right].
\]

Report DR là thước đo chính, IPS và SNIPS làm đối chiếu. Không clip trọng số hay kết quả âm để che nhiễu. DR/IPS có thể vượt [0,1] trong mẫu hữu hạn; SNIPS nằm trong [0,1] với reward nhị phân và trọng số dương. Báo cáo thêm:

\[
ESS=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

Một triệu dòng log có thể chỉ tương đương vài nghìn quan sát hữu ích nếu policy rất tập trung.

So sánh với `Logger` là CTR trực tiếp của cùng test. `Pooled TS` là baseline học không phân biệt vị trí, **không phải** phục dựng chính xác Bernoulli TS production của Zozo. Bài báo Zozo mô tả cả Random và BTS; file chính chỉ có propensity uniform, không có nhãn cho phép tách hai policy production.

Table 4 lấy chênh lệch giữa hai policy trên cùng các mẫu bootstrap theo block một giờ. Dùng 2.000 lần bootstrap, giữ nguyên policy đã fit; khoảng tin cậy là xấp xỉ, điều kiện trên training/policy đã fit. Nó không bao gồm bất định do fit lại training, không phải CI của online learning và không xử lý hoàn hảo tương quan người dùng lặp lại vì thiếu user ID. Chỉ có hai ngày test; không khái quát kết quả thành hiệu quả dài hạn. Các so sánh pairwise chưa điều chỉnh đa kiểm định.

## Câu b Độ nhạy với batch size

`replay` duyệt validation theo thời gian. Mỗi batch:

1. Từ lịch sử đã có, sinh xác suất policy trước khi đọc reward trong batch.
2. Chỉ nhận phản hồi khi action policy khớp action log. Mã nhận một dòng với xác suất \(\pi(a_i\mid x_i,p_i)\), tương đương tích phân phép lấy mẫu action rồi kiểm tra khớp.
3. Cộng số click và số lần hiển thị của những dòng nhận được vào model tại cuối batch.

Khi logger uniform trên \(K\) item, xác suất nhận trung bình là \(1/K\), không phụ thuộc context. Chỉ khoảng 1/79 log được dùng làm phản hồi học. IPS tích lũy dùng mọi dòng và xác suất được xác định trước reward, không dùng reward của action chưa hiển thị.

Thử batch **500, 5.000, 50.000 dòng log**, 5 seed. **Đơn vị batch là số slot log đã duyệt**, không phải số page view, không phải số phản hồi được chấp nhận và không phải số giây. Ví dụ B=5.000 tương ứng trung bình khoảng 63 phản hồi khớp. Đây là delayed replay trên lịch log, không phải tái hiện đầy đủ tốc độ học khi triển khai thật.

Table 5/Figure 2 báo cáo trung bình và SD giữa seed, không gọi SD là CI. Figure 7 minh họa một seed, không dùng để tuyên bố khác biệt có ý nghĩa thống kê. Batch nhỏ thường cập nhật nhanh hơn nhưng không nhất thiết có CTR cao hơn trong mẫu hiếm click. Đọc kết quả thực tế thay vì viết kết luận định sẵn.

## Câu c Độ nhạy với tham số

- TS: \(\kappa\in\{10,100,1000\}\).
- UCB: \(c\in\{0{,}001,0{,}01,0{,}1,1\}\), giữ \(\kappa=100\).
- Chọn theo DR trung bình trên validation, không dùng test.

Table 6/Figure 3 cho thấy trade-off làm trơn/khám phá. Với UCB pooled, thứ tự gán ba item có thể hòa score giữa các vị trí; seed có thể thay đổi cách phá hòa. Với UCB position không có hòa đáng kể, SD giữa seed có thể bằng 0; điều này không có nghĩa là không có bất định thống kê.

## Câu d Aggregation và heterogeneity

So sánh trên cùng validation và cùng tham số đã chọn:

1. **Pooled**: mỗi item một phân phối reward, gộp ba vị trí.
2. **Position**: mỗi item–vị trí một phân phối.
3. **Segment**: thêm nhóm theo `user_feature_0`, dùng prior co về ước lượng global item–vị trí từ train.

Biến thể segment dùng empirical Bayes: group posterior cộng prior \(\kappa m_{a,p}\), trong đó \(m_{a,p}\) fit từ toàn bộ train. Nhóm chưa thấy quay về model global. Parent chứa cả dữ liệu của từng nhóm, nên đây là partial pooling gần đúng, không phải suy luận Bayesian phân cấp chính xác.

Tăng độ chi tiết giúp biểu diễn khác biệt sở thích nhưng giảm số click mỗi ô; do đó personalization có thể làm CTR xấu đi. Phần (d) giữ `user_feature_0` cố định để so sánh minh bạch. Phần fairness khảo sát cả bốn biến như đề cho phép. Không mã hóa hash thành một số liên tục.

## Câu e Fairness cho người dùng và item

### Người dùng

Với từng level của từng feature, Table 9 in CTR logger, DR, SNIPS, ESS và uplift so với logger trong chính nhóm đó:

\[
uplift_g=\widehat V_g/CTR_{logger,g}-1.
\]

Không so CTR tuyệt đối rồi mặc định đó là phân biệt đối xử: nhóm có thể khác mức quan tâm nền. Báo cáo cả chênh lệch CTR tuyệt đối giữa các nhóm trong cùng feature và worst-group uplift.

Để tránh nhóm vài click chi phối kết luận, nhóm dùng trong tóm tắt phải có ít nhất \(100K=7.900\) dòng test và 10 click logger. **Cùng một tập nhóm đủ điều kiện được dùng cho mọi policy**. Tất cả nhóm còn lại vẫn xuất trong Table 9, không bị xóa. `low_realized_ESS` cảnh báo ESS thực tế dưới 100. Ngưỡng là lựa chọn phân tích phải nêu rõ; nó không biến các group point estimate thành kết luận chắc chắn. Chưa kiểm tra fairness trên mọi giao của bốn feature.

### Item

Tính expected exposure share từ xác suất policy trên toàn bộ context test; không tính exposure chỉ trên những dòng replay khớp. Giữ cả item có exposure bằng 0 trong mẫu số.

- Gini thấp hơn: exposure đều hơn.
- Entropy chuẩn hóa cao hơn: ít tập trung hơn.
- Minimum share: mức hiển thị thấp nhất.
- Coverage: tỷ lệ item có xác suất hiển thị dương. Khi có epsilon, coverage luôn 100%, nên cần xem cả Gini và minimum share.

Đây là fairness về **cơ hội được hiển thị**, không phải fairness theo chất lượng item hay lợi ích nhà bán hàng. Chưa có thông tin seller để đo fairness theo seller.

### Mở rộng đã triển khai

`Fair TS` kết hợp TS phân nhóm với uniform slate:

\[
\pi_{fair}(a\mid x,p)=(1-\epsilon)\pi_{segment}(a\mid x,p)+\epsilon/K.
\]

Thử bốn feature và \(\epsilon\in\{0{,}10,0{,}25,0{,}50\}\) trên validation. Giữ các cấu hình có DR ít nhất 95% DR của Position TS trên validation, rồi chọn cấu hình có worst-group uplift cao nhất. Nếu không có cấu hình qua ngưỡng, code chọn phương án best effort và ghi `fairness_utility_floor_feasible=false` trong manifest. Tính khả thi phải được báo cáo.

Nhánh uniform bảo đảm xác suất tối thiểu \(\epsilon/K\) tại mỗi slot cho mỗi item trong candidate set. Nó **không bảo đảm** utility của mọi nhóm người dùng tăng, không bảo đảm CTR test giữ ngưỡng 95%, và không thiết lập demographic parity. Phải đọc Table 8/9 để xem kết quả test có thật sự tốt hơn với thước đo đã chọn hay không. Kết quả người dùng hiện là point estimates; không viết “cải thiện có ý nghĩa thống kê” cho chúng.

## Câu f Tính dừng

Figure 6 và Table 10 mô tả CTR theo ngày. Một đường lên xuống không đủ chứng minh reward không dừng.

Table 11 dùng ba kiểm định likelihood-ratio nhị thức, so xác suất click không đổi qua bảy ngày với xác suất riêng từng ngày:

1. CTR toàn bộ dữ liệu.
2. CTR có điều kiện trên item và position.
3. CTR có điều kiện thêm `user_feature_0`.

Với các ô ít click, thay xấp xỉ chi-square bằng parametric bootstrap: giữ số lần hiển thị từng ô/ngày, sinh click dưới xác suất pooled của ô, fit lại null trong mỗi mẫu và tính statistic. Full run dùng 1.000 mẫu; p nhỏ nhất có thể báo là 1/1001, không được hiểu p bằng 0.

Kiểm tra thêm phân phối từng context feature qua các ngày bằng chi-square. Các feature có level hiếm được ghi là exploratory. Điều chỉnh Benjamini–Hochberg trên cả bảy kiểm định. Đây vẫn là kiểm định dựa vào giả định độc lập Bernoulli có điều kiện; tương quan thời gian/người dùng có thể làm p-value lạc quan.

Phân biệt **reward drift** với **context drift**. Có thể reward có điều kiện ổn định nhưng mix người dùng thay đổi. “Không bác bỏ H0” chỉ là chưa tìm đủ bằng chứng chống lại giả thuyết, không chứng minh môi trường dừng. Dữ liệu bảy ngày cũng không thể chứng minh tính dừng dài hạn.

## Câu g Thay đổi model khi không dừng

Mã đã có tùy chọn discount trong `replay`:

\[
N\leftarrow\gamma N+N_{batch},\qquad S\leftarrow\gamma S+S_{batch}.
\]

Prior giữ nguyên; discount áp dụng cả lịch sử warm-start. Table 15 thử \(\gamma=1,0{,}99,0{,}95\) sau mỗi 5.000 dòng log trên validation. Đơn vị discount phải giữ cố định để so sánh. Half-life lịch sử là \(\log(0{,}5)/\log(\gamma)\) batch với \(0<\gamma<1\). Đây là minh họa discount, chưa phải tối ưu chung gamma và batch.

Trong báo cáo, trình bày thêm các lựa chọn chưa triển khai:

- Sliding window: bỏ thống kê quá cũ; chọn độ dài theo validation theo thời gian.
- Change detection: theo dõi residual/reward, reset hoặc tăng khám phá khi phát hiện đổi chế độ.
- Time context: thêm giờ/ngày nhưng tránh quá nhiều nhóm hiếm click.
- Walk-forward evaluation: fit quá khứ, đánh giá tương lai nhiều lần, cần nhiều thời gian dữ liệu hơn.

Không tự động kết luận discount tốt hơn chỉ vì đã phát hiện drift; so Table 15 và nêu đánh đổi giữa khả năng thích nghi với nhiễu.

## Bản đồ câu hỏi và kết quả

| Câu | Bảng và hình chính | Hàm liên quan |
|---|---|---|
| a | Tables 2–4, 12, 14; Figure 1 | `Bandit`, `assign_slate`, `evaluate`, `bootstrap_means` |
| b | Table 5; Figures 2, 7 | `replay` |
| c | Table 6; Figure 3 | vòng lặp `tuning_rows` |
| d | Table 7; Figure 4 | `Config.structure` |
| e | Tables 8, 9, 16, 17; Figure 5 | `fairness`, `exposures`, `gini` |
| f | Tables 10, 11; Figure 6 | `stationarity`, `deviance` |
| g | Table 15 | `replay(..., discount=...)` |

Table 1 là audit nguồn; Table 13 cung cấp thống kê lịch sử item–vị trí. Table 12 là slate tối ưu theo **posterior mean**, giúp minh họa item nào vào vị trí nào. Nó không đồng nhất với stochastic policy dùng trong Table 3: TS còn lấy mẫu và có uniform exploration.

## Viết và nộp báo cáo LaTeX

Trong thư mục `solution`, biên dịch:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Assignment_2_STUDENTNUMBER report.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Assignment_2_STUDENTNUMBER report.tex
```

Thay `STUDENTNUMBER` bằng mã thật, cả ở lệnh trên lẫn `\studentnumber` trong `report.tex`. Có thể dùng Overleaf: tải `report.tex` cùng `outputs/tex` và `outputs/figures`, giữ nguyên cấu trúc thư mục, chọn pdfLaTeX. Báo cáo dùng tiếng Anh để phù hợp đề; tài liệu giải thích này dùng tiếng Việt.

`report.tex` nạp bảng và `findings.tex` do chương trình sinh ra, nên tránh chép tay số khác lần chạy. Mọi bảng đều được in ra console với identifier; mọi hình được lưu và thông báo identifier, có `--show` để mở xem. Hình PDF và PNG thể hiện cùng dữ liệu.

Trước nộp: thay mã sinh viên, đọc lại từng công thức, viết diễn giải của bạn, kiểm tra giới hạn và giả định 79/80 item, xem tất cả hình, đối chiếu bảng với `console.txt`, và kiểm tra quy định môn học về khai báo công cụ hỗ trợ. Hai PDF được cung cấp không nêu page limit hay ngày nộp; không tự thêm những yêu cầu này. Chỉ nộp những thành phần giảng viên/Canvas yêu cầu.

## Tài liệu nguồn

- [Đề Assignment 2026](../2026_Assignment.pdf): định nghĩa dữ liệu và câu (a)–(g).
- [Guidelines](../Guidelines%20for%20coding%20and%20submission%20of%20Assignment%202.pdf): seed, identifier, một ngôn ngữ và báo cáo LaTeX.
- [Bài báo Zozo được cung cấp](../zozo%20paper.pdf), Saito và cộng sự (2021), đặc biệt mục 2–3: logging policy, propensity theo vị trí và OPE.
- [Open Bandit Pipeline của tác giả](https://github.com/st-tech/zr-obp): nguồn chính về dữ liệu và triển khai tham khảo.
- [Li và cộng sự về offline replay](https://arxiv.org/abs/1003.5956): cơ sở replay từ random logging. Các bảo đảm lý thuyết không tự động áp dụng nguyên vẹn cho mọi biến thể lịch batch, drift hay lọc dữ liệu.
