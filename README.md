## J1. Đối chứng GroupNorm — KHÔNG SR. CHẠY TRƯỚC J2.
## Bắt buộc: GroupNorm TỰ NÓ cải thiện hội tụ. Bỏ qua bước này thì nếu J2
## vượt 76.68% sẽ KHÔNG THỂ biết công lao thuộc về SR hay GroupNorm.
## => 76.88% (best epoch 60/80, early stop epoch 78). +0.20 so với baseline 76.68%.
## GPU: RTX 4090, 1:31/epoch (nhanh ~5.9x so với V100 8:55/epoch).
arser hoạt động, chart vẽ đúng dữ liệu J1 thật. Panel loss cho thấy overfit rất rõ: đường train (nét liền) chạm 0, đường val (nét đứt) chạm đáy quanh epoch 19 rồi đi ngang/nhích lên.

Tóm tắt J1:

Val Acc
Baseline ResBlock	76.68%
J1 — + GroupNorm	76.88% (+0.20)
Nhưng phải nói thẳng: +0.20 nằm trong biên độ nhiễu (validation 999 sample → CI 95% ≈ ±2.7 điểm). Giá trị thật của J1 không nằm ở con số này mà ở chỗ nó là đối chứng — từ giờ J2 phải so với 76.88%, không phải 76.68%.

Phát hiện quan trọng hơn con số accuracy — model overfit nặng:

Epoch	Train Loss	Val Loss	Val Acc
19	0.0788	0.2591 ← đáy	71.37%
60	0.0074	0.3308	76.88%
78	0.0042	0.3535	75.88%
Train loss 0.0042 = gần như thuộc lòng tập train. Val loss tăng 36% từ đáy. Điều này khớp với bài học cũ (run E: 60 epoch tệ hơn 30 epoch) → dư địa cải thiện nằm ở chống overfit, không phải thêm tham số hay train lâu hơn. Đáng cân nhắc sau khi xong J2/J3.

python train.py \
 --preset stable \
 --experiment-name crnn_resblock_groupnorm_nosr \
 --backbone-norm group \
 --num-workers 8 --aug-level full


 <img width="1158" height="694" alt="image" src="https://github.com/user-attachments/assets/54a2759e-2e69-4290-ad14-b0ee3e7b1566" />
<img width="1141" height="740" alt="image" src="https://github.com/user-attachments/assets/ed483bb3-e794-480c-857c-89e6a6b2e966" />
<img width="1188" height="788" alt="image" src="https://github.com/user-attachments/assets/4c058dd4-c7bc-4ed1-9ebb-37b796f4dbfe" />
<img width="1133" height="805" alt="image" src="https://github.com/user-attachments/assets/9464a457-046a-4dc0-8ec9-da93252c64f7" />
<img width="1133" height="791" alt="image" src="https://github.com/user-attachments/assets/60446406-285f-435c-9c17-e89bf4aaf846" />
<img width="1140" height="165" alt="image" src="https://github.com/user-attachments/assets/8b2b3855-5fc2-45f1-9f61-c69dee36f217" />


