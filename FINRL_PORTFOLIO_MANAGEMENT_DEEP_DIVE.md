# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: THIẾT KẾ BÀI TOÁN TỐI ƯU HÓA DANH MỤC (PORTFOLIO MANAGEMENT) TRONG FRAMEWORK FINRL

---

## 1. TỔNG QUAN BÀI TOÁN TỐI ƯU HÓA DANH MỤC (PORTFOLIO MANAGEMENT)

Trong tài chính định lượng, bài toán **Quản lý / Tối ưu hóa danh mục đầu tư (Portfolio Management - PM)** là quá trình phân bổ liên tục nguồn vốn vào một tập hợp gồm $N$ tài sản rủi ro (cổ phiếu, tiền mã hóa, hàng hóa) và 1 tài sản phi rủi ro (tiền mặt - Cash) qua các chu kỳ giao dịch $t = 1, 2, \dots, T$.

Khác biệt căn bản với bài toán giao dịch đơn lẻ (Single-stock trading - mua/bán theo số lượng cổ phiếu $k$ cổ phiếu), bài toán tối ưu danh mục giải quyết việc tìm kiếm vector tỷ trọng phân bổ tối ưu $\mathbf{w}_t$:
$$\mathbf{w}_t = [w_{t, 0}, w_{t, 1}, w_{t, 2}, \dots, w_{t, N}]^\top \in \mathbb{R}^{N+1}$$
thỏa mãn **ràng buộc mặt phẳng đơn hình (Probability Simplex)**:
$$\sum_{i=0}^N w_{t, i} = 1 \quad \text{và} \quad w_{t, i} \ge 0 \quad \forall i \in \{0, 1, \dots, N\}$$
trong đó $w_{t, 0}$ là tỷ trọng tiền mặt, $w_{t, 1 \dots N}$ là tỷ trọng của $N$ tài sản rủi ro.

---

## 2. THIẾT KẾ MÔI TRƯỜNG TRONG FINRL: HAI TRƯỜNG PHÁI KHÁC BIỆT

Trong repository FinRL, tồn tại **hai môi trường độc lập** phục vụ bài toán danh mục với triết lý thiết kế và công thức toán học hoàn toàn khác nhau:

| Đặc tính | `PortfolioOptimizationEnv` (Chuyên biệt hiện đại) | `StockPortfolioEnv` (Legacy Baseline) |
| :--- | :--- | :--- |
| **Vị trí file** | `finrl/meta/env_portfolio_optimization/env_portfolio_optimization.py` | `finrl/meta/env_portfolio_allocation/env_portfolio.py` |
| **Công trình gốc** | Costa & Costa (2023) & Jiang et al. (2017) | Jiang et al. (biến thể rút gọn của FinRL) |
| **Observation Space** | Tensor 3D $(F \times N \times M)$ hoặc Dict `{state, last_action}` | Ma trận hiệp phương sai + chỉ báo kỹ thuật $(N + K) \times N$ |
| **Action Space** | Box $(N + 1)$ (Bao gồm Tiền mặt tại index 0) | Box $(N)$ (Chỉ gồm các cổ phiếu) |
| **Ràng buộc Action** | Softmax Simplex: $\sum w_i = 1, w_i \ge 0$ | Softmax normalization cơ bản |
| **Hiện tượng trôi dạt (Price Drift)** | **Có** ($\mathbf{w}'_{t-1} = \frac{\mathbf{y}_t \odot \mathbf{w}_{t-1}}{\mathbf{y}_t^\top \mathbf{w}_{t-1}}$) | Không mô phỏng rõ rệt trôi dạt trọng số |
| **Phí giao dịch (Cost Model)** | **TRF (Transaction Remainder Factor)** & **WVM** | Đơn giản hóa, không trừ phí giao dịch bước |
| **Hàm Phần thưởng (Reward)** | **Logarithmic Return**: $r_t = \ln(V_t / V_{t-1})$ | Tổng giá trị tài sản tức thời $V_t$ hoặc return |
| **Agent tối ưu** | `finrl/agents/portfolio_optimization` (EIIE, EI3, GPM) | `stablebaselines3` (PPO, A2C, DDPG, SAC) |

---

## 3. PHÂN TÍCH TOÁN HỌC CHI TIẾT TRONG `PortfolioOptimizationEnv`

### 3.1. Vector Biến Động Giá Tương Đối (Relative Price Vector)
Tại mỗi bước thời gian $t$, giá các tài sản thay đổi so với bước trước đó $t-1$. Vector biến động giá tương đối được định nghĩa:
$$\mathbf{y}_t = \left[ 1.0, \frac{p_{t, 1}}{p_{t-1, 1}}, \frac{p_{t, 2}}{p_{t-1, 2}}, \dots, \frac{p_{t, N}}{p_{t-1, N}} \right]^\top \in \mathbb{R}^{N+1}$$
- Phần tử đầu tiên $y_{t, 0} = 1.0$ là tài sản tiền mặt (Cash), giả định sức mua của tiền mặt danh nghĩa không đổi trong chu kỳ bước ngắn (không chịu rủi ro biến động giá cổ phiếu).
- Các phần tử $y_{t, i} = \frac{p_{t, i}}{p_{t-1, i}}$ ($i \ge 1$) biểu diễn tỷ số tăng/giảm giá của cổ phiếu $i$.

### 3.2. Hiện Tượng Trôi Dạt Trọng Số (Price Drift Effect)
Vào cuối phiên giao dịch $t-1$, danh mục được phân bổ theo tỷ trọng $\mathbf{w}_{t-1}$. Tuy nhiên, qua đêm và trong suốt phiên $t$, biến động thị trường làm thay đổi giá trị tương đối của các vị thế:
- Cổ phiếu tăng giá sẽ chiếm tỷ trọng lớn hơn trong danh mục.
- Cổ phiếu giảm giá sẽ chiếm tỷ trọng nhỏ hơn.

Trọng số thực tế **trước khi tái cân bằng** tại bước $t$ bị "trôi dạt" (drift) thành $\mathbf{w}'_{t-1}$:
$$\mathbf{w}'_{t-1} = \frac{\mathbf{y}_t \odot \mathbf{w}_{t-1}}{\mathbf{y}_t^\top \mathbf{w}_{t-1}} = \frac{[w_{t-1, 0}, w_{t-1, 1} y_{t, 1}, \dots, w_{t-1, N} y_{t, N}]^\top}{\sum_{j=0}^N w_{t-1, j} y_{t, j}}$$
Nếu agent quyết định giữ nguyên trọng số $\mathbf{w}_t = \mathbf{w}'_{t-1}$, agent **không phải trả bất kỳ khoản phí giao dịch nào**. Nếu $\mathbf{w}_t \neq \mathbf{w}'_{t-1}$, agent buộc phải thực hiện mua/bán và chịu phí tái cân bằng.

### 3.3. Các Mô Hình Phí Giao Dịch (Commission Fee Models)

#### A. Mô hình TRF (Transaction Remainder Factor - Jiang et al., 2017)
Giả sử phí hoa hồng mua là $c_p$ và phí hoa hồng bán là $c_s$ (trong FinRL giả định $c_p = c_s = c$). Sau khi tái cân bằng, tổng giá trị danh mục giảm đi một hệ số $\mu_t \in (0, 1]$. Phương trình bảo toàn dòng tiền:
$$c \sum_{i=1}^N (w'_{t-1, i} - \mu_t w_{t, i})^+ + c \sum_{i=1}^N (\mu_t w_{t, i} - w'_{t-1, i})^+ = (1 - \mu_t)$$
Giải phương trình này dẫn đến điểm bất động (Fixed-point iteration):
$$\mu_t^{(k+1)} = \frac{1 - c_p w'_{t-1, 0} - (c_s + c_p - c_s c_p) \sum_{i=1}^N \max(w'_{t-1, i} - \mu_t^{(k)} w_{t, i}, 0)}{1 - c_p w_{t, 0}}$$
Trong code FinRL (`env_portfolio_optimization.py` dòng 321-334), vòng lặp `while abs(mu - last_mu) > 1e-10` thực thi hội tụ thuật toán này trong chưa đầy 5-10 vòng lặp.

#### B. Mô hình WVM (Weights Vector Modifier) & Linear Turnover
- **WVM**: Trừ thẳng chi phí giao dịch phát sinh vào số dư tiền mặt:
  $$\text{Fee}_t = \sum_{i=1}^N |w_{t, i} - w'_{t-1, i}| \cdot V_{t-1}$$
  Nếu số dư tiền mặt không đủ thanh toán phí, giao dịch bị từ chối và trọng số giữ nguyên $\mathbf{w}_t = \mathbf{w}'_{t-1}$.
- **Linear Turnover Model**:
  $$\mu_t = 1.0 - c_{\text{fee}} \sum_{i=1}^N |w_{t, i} - w'_{t-1, i}|$$

### 3.4. Cập Nhật Giá Trị Tài Sản & Hàm Phần Thưởng (Log Return)
Sau khi trả phí $\mu_t$ và đón nhận biến động giá $\mathbf{y}_t$, giá trị tài sản mới tại bước $t$ là:
$$V_t = \mu_t \cdot V_{t-1} \cdot (\mathbf{y}_t^\top \mathbf{w}_t)$$

Phần thưởng tức thời của môi trường là **Tỷ suất sinh lời logarit (Logarithmic Return)**:
$$r_t = \ln\left( \frac{V_t}{V_{t-1}} \right) \times \text{reward\_scaling} = \left[ \ln(\mu_t) + \ln(\mathbf{y}_t^\top \mathbf{w}_t) \right] \times \text{reward\_scaling}$$

#### Ý nghĩa kinh tế sâu sắc của hàm phần thưởng Log Return:
1. **Tính cộng gộp thời gian (Time-additivity)**:
   $$\sum_{t=1}^T r_t = \sum_{t=1}^T \ln\left(\frac{V_t}{V_{t-1}}\right) = \ln\left(\frac{V_T}{V_0}\right)$$
   Tổng phần thưởng tích lũy trong Reinforcement Learning chính là **tỷ suất sinh lời toàn kỳ (Total log growth rate)**.
2. **Nguyên lý Kelly (Kelly Criterion)**: Tối đa hóa kỳ vọng của Log Return tương đương với tối đa hóa **tỷ suất sinh lời nhân tố hình học dài hạn** (Geometric Growth Rate), bảo vệ danh mục khỏi nguy cơ cháy tài khoản (Ruin Probability $\to 0$).

---

## 4. THIẾT KẾ AGENT CHUYÊN BIỆT (`finrl/agents/portfolio_optimization`)

Repository FinRL cung cấp một module agent hoàn toàn độc lập với Stable-Baselines3 tại thư mục `finrl/agents/portfolio_optimization/`:

### 4.1. Kiến Trúc Mạng Nơ-ron EIIE (Ensemble of Identical Independent Evaluators)
Được đề xuất bởi Jiang et al. (2017), mạng EIIE giải quyết vấn đề bùng nổ tham số khi số lượng cổ phiếu lớn:
- **Khối trích xuất đặc trưng thời gian (Temporal Conv2D)**:
  - Tầng 1: `Conv2d(in_channels=F, out_channels=2, kernel_size=(1, k_size))` + ReLU.
  - Tầng 2: `Conv2d(in_channels=2, out_channels=20, kernel_size=(1, time_window - k_size + 1))` + ReLU.
  - Phép tích chập chỉ trượt trên trục thời gian $M$, áp dụng chung trọng số (Weight-sharing) cho tất cả $N$ cổ phiếu. Điều này đảm bảo tính **bất biến theo thứ tự tài sản** (Permutation-equivariant).
- **Ghép kênh Trọng số quá khứ (Last Action Concatenation)**:
  - Vector trọng số bước trước của các cổ phiếu $w_{t-1, 1:N}$ được ghép vào tensor đặc trưng (kênh thứ 21). Giúp mạng nơ-ron nhận thức được vị thế hiện tại và hạn chế giao dịch quá mức (Chống bào mòn phí).
- **Tầng Quyết định (Decision Layer)**:
  - `Conv2d(in_channels=21, out_channels=1, kernel_size=(1, 1))` thu nhỏ về 1 điểm số tiềm năng cho mỗi cổ phiếu.
  - Nối thêm tham số học được của Tiền mặt (`cash_bias`).
  - Áp dụng `Softmax` để đảm bảo đầu ra là vector trọng số trên Simplex.

### 4.2. Kiến Trúc EI3 và GPM
- **EI3 (Inception Architecture)**: Tích hợp nhiều nhánh tích chập với kích thước kernel khác nhau: ngắn hạn ($k=3$), trung hạn ($k=21$), dài hạn (Max-pooling toàn bộ cửa sổ $M=50$).
- **GPM (Graph-based Portfolio Management)**: Sử dụng mạng nơ-ron đồ thị quan hệ (**RGCN - Relational Graph Convolutional Network**) từ thư viện PyTorch Geometric (`torch_geometric`) để mô hình hóa mối liên hệ giữa các công ty trong cùng ngành, chuỗi cung ứng hoặc quan hệ sở hữu chéo.

### 4.3. Thuật Toán Huấn Luyện Policy Gradient & Bộ Nhớ PVM
- **Hàm mất mát trực tiếp (Direct Policy Gradient Loss)**:
  $$\mathcal{L}(\theta) = - \frac{1}{B} \sum_{k=1}^B \ln\left( \mu_k \sum_{i=0}^N \pi_\theta(S_k)_i \cdot y_{k, i} \right)$$
  Mạng trực tiếp tối đa hóa Log Return thông qua thuật toán tối ưu hóa AdamW. Không cần hàm Critic $Q(s, a)$ vì đạo hàm của phần thưởng theo hành động có thể tính giải tích trực tiếp!
- **Bộ nhớ Portfolio Vector Memory (PVM)**:
  - Do thuật toán cần $w_{t-1}$ để tính $w_t$, nếu lưu trong Replay Buffer ngẫu nhiên sẽ làm đứt gãy tính liên tục của chuỗi trọng số.
  - PVM là mảng cố định kích thước $T \times (N+1)$ lưu giữ trọng số liên tục của danh mục qua các bước thời gian.
- **Online Learning Trong Kiểm Thử (Test-time Adaptation)**:
  - Trong pha kiểm thử (`test()`), agent tiếp tục thực hiện gradient ascent mỗi chu kỳ $K$ bước (`online_training_period=10`) để thích ứng với sự thay đổi chế độ thị trường (Regime Shift).

---

## 5. THIẾT KẾ MÔ HÌNH TRONG STABLE-BASELINES3 (`finrl/agents/stablebaselines3`)

Module `finrl/agents/stablebaselines3/models.py` cung cấp các wrapper bao bọc các thuật toán Continuous Control phổ biến của thư viện Stable-Baselines3:

### 5.1. Danh mục Mô hình & Action Noise
Mô hình hỗ trợ:
- On-policy: `A2C`, `PPO`.
- Off-policy Actor-Critic: `DDPG`, `TD3`, `SAC`.
- Nâng cao (sb3-contrib): `TQC` (Truncated Quantile Critics), `CrossQ` (Batch Normalization thay thế Target Networks).

Đối với các mô hình Off-policy (DDPG, TD3), wrapper tự động chèn nhiễu thăm dò continuous:
- `NormalActionNoise`: Nhiễu Gaussian $\mathcal{N}(0, \sigma^2)$.
- `OrnsteinUhlenbeckActionNoise`: Nhiễu tự tương quan theo thời gian (thích hợp cho bài toán có quán tính vị thế).

### 5.2. Kiến Trúc DRLEnsembleAgent
`DRLEnsembleAgent` là thiết kế độc đáo của FinRL kết hợp nhiều thuật toán DRL qua cơ chế cửa sổ trượt (Rolling Window):
1. **Giai đoạn Train Window**: Huấn luyện đồng thời các agent (ví dụ: PPO, A2C, DDPG) trên cửa sổ lịch sử $T_{\text{train}}$.
2. **Giai đoạn Validation Window**: Chạy kiểm thử trên tập Validation $T_{\text{val}}$.
3. **Lựa chọn mô hình tốt nhất (Sharpe-based Selection)**:
   Mô hình đạt tỷ số **Sharpe Ratio** cao nhất trên tập Validation sẽ được chọn để giao dịch trên tập Trade tiếp theo $T_{\text{trade}}$.
4. **Cơ chế Kiểm soát Hỗn loạn (Turbulence Threshold)**:
   Nếu chỉ số hỗn loạn thị trường (Mahalanobis distance của lợi nhuận cổ phiếu) vượt ngưỡng `turbulence_threshold`, agent tự động thanh lý toàn bộ danh mục và nắm giữ 100% Cash để phòng vệ rủi ro.

### 5.3. Tinh Chỉnh Siêu Tham Số Với Optuna (`hyperparams_opt.py`)
Module `hyperparams_opt.py` sử dụng thư viện Optuna để tìm kiếm không gian siêu tham số:
- **Tốc độ học (Learning rate)**: Lấy mẫu log-uniform từ $10^{-5}$ đến $1.0$, hỗ trợ `linear_schedule`.
- **Cấu trúc mạng nơ-ron (`net_arch`)**:
  - Small: `pi=[64, 64], vf=[64, 64]`
  - Medium: `pi=[256, 256], vf=[256, 256]`
- **Hệ số chiết khấu $\gamma$**: Lấy mẫu categorical từ $[0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]$.
- **GAE Lambda $\lambda$**: $[0.8, 0.9, 0.95, 0.98, 1.0]$.
- **Entropy Coefficient**: Khống chế mức độ thăm dò (Exploration).
- **Batch size & n_steps**: Đồng bộ kích thước batch và rollout buffer.

---

## 6. MÔ HÌNH TRONG RLLIB (`finrl/agents/rllib/models.py`)

Module RLlib tích hợp Ray Framework phục vụ huấn luyện phân tán trên cụm máy tính:
- Hỗ trợ các thuật toán `a2c`, `ddpg`, `td3`, `sac`, `ppo`.
- Truyền dữ liệu thị trường qua `env_config` dưới dạng các mảng NumPy (`price_array`, `tech_array`, `turbulence_array`).
- Tận dụng cơ chế Ray Actor để mở rộng số lượng worker thu thập dữ liệu (rollout workers) song song, giải quyết bài toán dữ liệu lớn khi mở rộng sang hàng trăm cổ phiếu.

---

## 7. ĐỐI CHIẾU THỰC HÀNH VỚI WORKSPACE VÀ KHUYẾN NGHỊ NGHIÊN CỨU

### 7.1. Đối chiếu với `portfolios-zero.ipynb`
Trong notebook `portfolios-zero.ipynb` của bạn:
- Bạn đã thiết lập lớp `PortfolioZeroEnv` dựa trên đúng tinh thần toán học của Jiang et al. (2017) với vector $N+1$, trôi dạt giá (Price Drift), phí tái cân bằng và Log Return reward.
- Tuy nhiên, phần mô hình trong `portfolios-zero.ipynb` đang hướng tới **Representation Learning** (Transformer Encoder-Decoder tiền huấn luyện dự báo giá $D=1, D=5$).

### 7.2. Đối chiếu với notebook vừa triển khai `FinRL_Baseline_PM.ipynb`
Tại `FinRL_Baseline_PM.ipynb`, chúng ta đã chuẩn hóa thành công:
1. Môi trường `FinRLPortfolioOptimizationEnv` chuẩn Gymnasium 1.4.0, hỗ trợ cả 2 mô hình phí TRF & Linear Turnover.
2. Bộ sinh dữ liệu GBM Correlated đa tài sản đảm bảo tính tự lập (Stand-alone runnable).
3. Đầy đủ các benchmark so sánh (Equal-Weight $1/N$, Cash Only, Random).
4. Lớp bọc `SB3PortfolioEnvWrapper` giúp các thuật toán của Stable-Baselines3 (PPO, SAC, A2C) có thể học và tương tác trực tiếp trên môi trường Simplex!

### 7.3. Định hướng cho Đề tài "Theory of Games" (Lý thuyết Trò chơi)
Dựa trên nền tảng môi trường này, đề tài nghiên cứu Lý thuyết Trò chơi có thể phát triển theo 2 hướng đột phá:
1. **Adversarial Market Game (Trò chơi đối kháng)**: Thiết lập một tác tử thị trường (Adversary) cố tình tạo ra các cú sốc thanh khoản hoặc trượt giá trong không gian chi phí $\mu_t$, huấn luyện Agent tối ưu danh mục đạt trạng thái cân bằng Nash bền vững (Robust Portfolio Optimization).
2. **Multi-Agent Cooperative/Competitive Allocation**: Mô phỏng nhiều nhà quản lý quỹ cạnh tranh nguồn thanh khoản thị trường với hàm mục tiêu tương đối (Relative Wealth / Market Share game).

