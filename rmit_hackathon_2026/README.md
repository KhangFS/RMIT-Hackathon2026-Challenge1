# RMIT GenAI & Cybersecurity Hackathon 2026

## 1. Cấu trúc thư mục
```
rmit_hackathon_2026/
├── src/
│   ├── dataset.py            # Dataset xử lý 3 classes: BENIGN, JAILBREAK, PROMPT_INJECTION
│   ├── preprocessing.py      # Chuẩn hóa Unicode, de-obfuscation, khử leetspeak đa ngôn ngữ
│   ├── utils.py              # Tính Macro-F1 3 lớp theo chuẩn chấm điểm của Kaggle
│   ├── prompt_security.py    # Hardened prompt, XML isolation, Canary injection
│   ├── egress_guard.py       # PII scrubbing & canary leak detector
│   ├── inference.py          # LLM inference engine & simulator
│   └── pipeline.py           # Kiến trúc phòng thủ đa tầng (Defense-in-Depth)
├── train.py                  # Script huấn luyện Stratified K-Fold + Weighted Cross-Entropy
├── predict.py                # Script suy luận tập test và sinh file submission.csv
└── submission.csv            # File nộp bài Kaggle (đúng 6.000 dòng, id,label)
```

## 2. Thông tin cuộc thi & Yêu cầu
- **Nhiệm vụ**: Phân loại prompt đa ngôn ngữ thành **3 classes**:
  1. `BENIGN` (~62%)
  2. `JAILBREAK` (~31%)
  3. `PROMPT_INJECTION` (~7%)
- **Ngôn ngữ**: Tiếng Bồ Đào Nha (`pt`), Pháp (`fr`), Ý (`it`), Anh (`en`).
- **Thang đo (Metric)**: **Macro-averaged F1 score** (cả 3 lớp tính trọng số bằng nhau).
- **Quy tắc đặt tên Kernel/Notebook**:
  `<university>_<campus>_<team>_C1`
  *(Ví dụ: `RMIT_SGS_teamname_C1`)*

## 3. Hướng dẫn sử dụng

### Huấn luyện mô hình (Fine-tuning)
Mặc định dùng `FacebookAI/xlm-roberta-base` (hoặc các approved model khác trong danh sách: `rembert`, `distilbert-base-multilingual-cased`):
```powershell
python train.py --model xlm-roberta-base --batch 16 --epochs 3 --folds 5
```

### Sinh file nộp bài (Inference test.csv)
```powershell
python predict.py --test_file "D:\Code\rmit\data\Data-rmit2026\test.csv" --output submission.csv
```
File `submission.csv` được sinh ra sẽ tự động kiểm tra đủ 6,000 dòng khớp 100% với `sample_submission.csv`.
