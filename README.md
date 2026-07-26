## J1. Đối chứng GroupNorm — KHÔNG SR. CHẠY TRƯỚC J2.
## Bắt buộc: I-b cho thấy GroupNorm TỰ NÓ cải thiện hội tụ. Bỏ qua bước này thì nếu J2
## vượt 76.68% sẽ KHÔNG THỂ biết công lao thuộc về SR hay GroupNorm.
## => 76.88% (best epoch 60/80, early stop epoch 78). +0.20 so với baseline 76.68%.
## GPU: RTX 4090, 1:31/epoch (nhanh ~5.9x so với V100 8:55/epoch).
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


