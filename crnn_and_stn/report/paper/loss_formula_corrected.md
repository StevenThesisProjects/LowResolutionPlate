# Công thức loss đã sửa — bản sẵn dán vào paper

> **Trả lời câu hỏi "có cần train lại model không": ❌ KHÔNG.**
> Cả 4 lỗi đều là **sai lệch giữa mô tả trong paper và code đã chạy** — code vẫn
> đúng và không đổi. Sửa ở đây là **sửa chữ trong bản thảo**, mọi số liệu
> J1/S1/S4 giữ nguyên hiệu lực. **0 giờ GPU.**
>
> Repo hiện **không chứa bản thảo paper** (chỉ có `paper_revision_plan.md`), nên
> tài liệu này đóng vai trò **bản nguồn** để copy sang Word/Overleaf.

---

## 1. Khối công thức chính — DÁN THAY CHO KHỐI CŨ

### 1.1. Bản LaTeX (Overleaf)

```latex
\begin{equation}
\mathcal{L}_{\text{Total}}
  = \mathcal{L}_{\text{CTC}}
  + \lambda_{\text{SR}} \cdot \mathcal{L}_{\text{SR}}
\label{eq:total}
\end{equation}

\begin{equation}
\mathcal{L}_{\text{SR}}
  = \underbrace{\frac{1}{N}\sum_{i=1}^{N}
      \left\| I_{\text{SR}}^{(i)} - \mathrm{Warp}_{\mathrm{sg}[\theta_i]}\!\left(I_{\text{HR}}^{(i)}\right) \right\|_1
    }_{\mathcal{L}_{1}}
  + \alpha \cdot \mathcal{L}_{\text{VGG}}
\label{eq:sr}
\end{equation}
```

Với $\mathrm{sg}[\cdot]$ là toán tử **stop-gradient** và $\theta_i$ là tham số affine
6 chiều do STN sinh ra cho frame $i$.

**Quan hệ với ký hiệu $\lambda_{\text{Perceptual}}$ quen thuộc:**

```latex
\lambda_{\text{Perceptual}} = \lambda_{\text{SR}} \cdot \alpha
```

### 1.2. Bản Unicode (Word)

```
L_Total = L_CTC + λ_SR · L_SR

L_SR    = (1/N) Σᵢ ‖ I_SR⁽ⁱ⁾ − Warp_sg[θᵢ](I_HR⁽ⁱ⁾) ‖₁  +  α · L_VGG

λ_Perceptual = λ_SR · α
```

### 1.3. Giá trị siêu tham số

| Cấu hình | $\lambda_{\text{SR}}$ | $\alpha$ | $\lambda_{\text{Perceptual}}$ hiệu dụng |
|---|---:|---:|---:|
| **S1** (đề xuất, có error bar) | 0.1 | **0** | **0** |
| **S4** (ablation, có error bar) | 0.1 | **0** | **0** |
| **J1** (ablation, có error bar) | — | — | — (không có nhánh SR) |
| _(ablation 1-seed, ở `backup/report/`)_ | 0.1 | 0.1 | 0.01 |

> 🚨 **Cả 3 cấu hình có error bar đều chạy $\alpha = 0$.** Nếu paper in công thức có
> số hạng $\mathcal{L}_{\text{VGG}}$ mà không chú thích, reviewer sẽ hiểu nhầm là con
> số headline được tạo ra **có** perceptual loss. Xem câu chú thích bắt buộc ở §3.

---

## 2. Bốn chỗ đã sửa và lý do

### Lỗi 1 — Bỏ `Warp` khỏi $I_{\text{SR}}$

| | |
|---|---|
| **Paper cũ (sai)** | $\mathcal{L}_{\text{SR}} = \frac{1}{N}\sum \lVert \mathrm{Warp}_\theta(I_{\text{SR}}) - \mathrm{Warp}_\theta(I_{\text{HR}}) \rVert_1$ |
| **Đã sửa** | $\mathcal{L}_{\text{SR}} = \frac{1}{N}\sum \lVert I_{\text{SR}} - \mathrm{Warp}_{\mathrm{sg}[\theta]}(I_{\text{HR}}) \rVert_1$ |
| **Vì sao** | Nhánh SR chạy **sau** STN, nên $I_{\text{SR}}$ **đã nằm sẵn trong khung đã nắn**. Warp nó lần nữa là biến đổi hình học **hai lần**. Chỉ target $I_{\text{HR}}$ (vốn ở khung gốc) cần được warp để hai vế cùng một hệ toạ độ. |
| **Nguồn** | [`src/training/trainer.py:263-272`](../../src/training/trainer.py) — chỉ `hr_flat` đi qua `grid_sample` |

### Lỗi 2 — "Smooth L1" → **L1**

| | |
|---|---|
| **Paper cũ (sai)** | mô tả $\mathcal{L}_{\text{SR}}$ là "Smooth L1 loss" |
| **Đã sửa** | **L1 loss** ($\ell_1$ thuần) |
| **Vì sao** | Code gọi `F.l1_loss`, **không** phải `F.smooth_l1_loss`. Đây thuần tuý là sai tên. |
| **Quyết định** | **Giữ L1, KHÔNG đổi code.** Đổi sang Smooth L1 sẽ đổi hàm mục tiêu → toàn bộ J1/S1/S4 (39 giờ GPU) phải chạy lại. Không đáng cho một thay đổi chưa có bằng chứng là tốt hơn. |
| **Nguồn** | [`src/training/losses.py:84`](../../src/training/losses.py) |

### Lỗi 3 — Thêm stop-gradient $\mathrm{sg}[\theta]$ 🆕

| | |
|---|---|
| **Paper cũ (thiếu)** | $\mathrm{Warp}_\theta(\cdot)$ — ngụ ý gradient chảy qua $\theta$ |
| **Đã sửa** | $\mathrm{Warp}_{\mathrm{sg}[\theta]}(\cdot)$ |
| **Vì sao** | Code gọi `theta_sel.detach()`. Nghĩa là **gradient của $\mathcal{L}_{\text{SR}}$ KHÔNG chảy ngược vào STN**. Đây là **lựa chọn thiết kế có chủ đích**, không phải bug. |
| **Nguồn** | [`src/training/trainer.py:270`](../../src/training/trainer.py) |

**Câu giải thích bắt buộc đưa vào paper** (reviewer chắc chắn sẽ hỏi):

> *Chúng tôi chặn gradient qua $\theta$ khi warp ảnh target. Nếu không, nhánh SR có
> thể tối thiểu hoá $\mathcal{L}_{\text{SR}}$ bằng cách kéo STN về phép biến đổi hình
> học **dễ tái tạo nhất** thay vì phép **nắn chỉnh đúng** — tức là lấy mục tiêu phụ
> (khôi phục pixel) chi phối mục tiêu chính (nhận dạng ký tự). Với stop-gradient,
> nhánh SR chỉ được phép làm ảnh nét hơn trong khung mà STN đã chọn.*

### Lỗi 4 — Vị trí $\lambda_{\text{Perceptual}}$

| | |
|---|---|
| **Paper cũ (sai)** | $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}}\mathcal{L}_{\text{SR}} + \lambda_{\text{Perc}}\mathcal{L}_{\text{VGG}}$ — 3 số hạng **song song** |
| **Đã sửa** | $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{SR}}(\mathcal{L}_1 + \alpha\mathcal{L}_{\text{VGG}})$ — perceptual **lồng trong** $\mathcal{L}_{\text{SR}}$ |
| **Vì sao** | Code nhân $\lambda_{\text{SR}}$ cho **cả cụm**. Hai cách viết chỉ tương đương khi $\lambda_{\text{Perc}} = \lambda_{\text{SR}} \times \alpha$. Ở ablation từng bật perceptual: $0.1 \times 0.1 = 0.01$ ✅ đúng bằng con số review nêu — **nhưng chỉ đúng tình cờ ở giá trị đó**. Nếu sau này đổi $\lambda_{\text{SR}}$ mà quên đổi $\lambda_{\text{Perc}}$, hai công thức lệch nhau. |
| **Nguồn** | [`src/training/trainer.py:336`](../../src/training/trainer.py) + [`losses.py:87-88`](../../src/training/losses.py) |

---

## 3. Ba câu chú thích BẮT BUỘC kèm công thức

Không có 3 câu này thì công thức đúng vẫn gây hiểu nhầm:

1. **Về perceptual loss:**
   > *Số hạng $\alpha\mathcal{L}_{\text{VGG}}$ được đặt $\alpha = 0$ ở toàn bộ các cấu
   > hình có error bar (J1, S1, S4). Đóng góp của nó chỉ được khảo sát ở một ablation
   > 1 seed và **chưa được xác nhận** bằng multi-seed.*

2. **Về stop-gradient:** dùng nguyên đoạn ở Lỗi 3.

3. **Về $\mathcal{L}_{\text{Edge}}$** (số hạng có trong code nhưng chưa từng nêu trong paper):
   > *Cài đặt còn hỗ trợ một số hạng edge loss (Sobel) với trọng số $\beta$, đặt
   > $\beta = 0$ ở mọi cấu hình báo cáo trong bài.*

   Nếu **không** muốn nhắc $\mathcal{L}_{\text{Edge}}$ trong paper thì bỏ luôn — vì
   $\beta = 0$ ở mọi cấu hình báo cáo nên nó không ảnh hưởng con số nào. Nhưng phải
   **nhất quán**: đã bỏ thì đừng để lại dấu vết ở chỗ khác.

---

## ✅ Checklist Nhóm 1 (cập nhật)

- [x] Xác nhận quy đổi $\lambda_{\text{Perceptual}} = \lambda_{\text{SR}} \times \alpha$
- [x] Verify $\alpha = 0$ trên banner của cả 3 lần chạy có error bar
- [x] Chốt **giữ L1**, không đổi sang Smooth L1 (tránh chạy lại 39h GPU)
- [x] **Soạn xong bản công thức đã sửa** (tài liệu này) — LaTeX + Unicode
- [x] **Xác nhận không cần train lại** — cả 4 lỗi là sai lệch mô tả, không phải sai code
- [ ] ✍️ Dán khối §1 vào bản thảo, thay khối công thức cũ
- [ ] ✍️ Thêm 3 câu chú thích ở §3
- [ ] ✍️ Rà lại phần Method xem còn chỗ nào mô tả "Smooth L1" hoặc warp kép không
