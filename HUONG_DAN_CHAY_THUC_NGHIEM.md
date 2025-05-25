# HƯỚNG DẪN CHẠY THỰC NGHIỆM AE-CPGNN

## 1. CHUẨN BỊ MÔI TRƯỜNG

### Cài đặt dependencies:
```bash
pip install -r requirements_AE.txt
```

### Kiểm tra môi trường:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
```

## 2. KIỂM TRA TÍNH ĐÚNG ĐẮN CỦA IMPLEMENTATION

### Chạy test để xác minh implementation:
```bash
python test_AE.py
```

Test này sẽ kiểm tra:
- Tải dữ liệu DBLP
- AttributeEncoder (DistilBERT)
- AE-ContextGNN model
- Forward pass

## 3. CHẠY THỰC NGHIỆM

### 3.1 Chạy với tham số mặc định (α=0.5):
```bash
python main_AE.py
```

### 3.2 Chạy với tham số α tùy chỉnh:
```bash
python main_AE.py -alpha 0.3
python main_AE.py -alpha 0.7
```

### 3.3 Chạy grid search để tìm α tối ưu:
```bash
python main_AE.py --grid_search
```

Grid search sẽ thử các giá trị α = [0.3, 0.5, 0.7] và tìm giá trị tốt nhất.

### 3.4 Chạy trên GPU cụ thể:
```bash
python main_AE.py -n 0  # GPU 0
python main_AE.py -n 1  # GPU 1
```

## 4. ĐÁNH GIÁ KẾT QUẢ

### 4.1 Đánh giá model đã train:
```bash
python evaluate_AE.py -path checkpoint/AE_DBLP_alpha_0.5 -alpha 0.5
```

### 4.2 So sánh với baseline CP-GNN:
```bash
python evaluate_AE.py --compare
```

Lệnh này sẽ:
- Chạy baseline CP-GNN
- Chạy AE-CP-GNN với α = [0.3, 0.5, 0.7]
- So sánh kết quả classification và clustering

## 5. HIỂU KẾT QUẢ THỰC NGHIỆM

### 5.1 Metrics đánh giá:
- **Classification Accuracy**: Độ chính xác phân loại node
- **ARI (Adjusted Rand Index)**: Đánh giá clustering
- **NMI (Normalized Mutual Information)**: Đánh giá clustering

### 5.2 Ý nghĩa tham số α:
- α = 1.0: Chỉ dùng structural embeddings (CP-GNN gốc)
- α = 0.0: Chỉ dùng attribute embeddings (DistilBERT)
- α = 0.5: Kết hợp cân bằng cả hai
- α tối ưu thường trong khoảng [0.3, 0.7]

### 5.3 Kết quả mong đợi:
AE-CP-GNN nên cho kết quả tốt hơn baseline CP-GNN vì:
- Kết hợp thông tin cấu trúc và ngữ nghĩa
- DistilBERT hiểu được ngữ nghĩa text
- Hybrid embeddings phong phú hơn

## 6. PHÂN TÍCH CHI TIẾT

### 6.1 Xem log training:
```bash
tail -f checkpoint/AE_DBLP_alpha_0.5/training.log
```

### 6.2 Kiểm tra embeddings:
```python
import torch
import numpy as np
from models import AE_ContextGNN
from utils import load_data
import config_AE

# Load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataloader = load_data(config_AE.data_config)
model = AE_ContextGNN(dataloader.heter_graph, config_AE.model_config, config_AE.data_config)

# Load checkpoint
checkpoint = torch.load('checkpoint/AE_DBLP_alpha_0.5/model.pth')
model.load_state_dict(checkpoint)
model.eval()

# Get embeddings
with torch.no_grad():
    embeddings = model(3).cpu().numpy()  # 3-hop embeddings
    
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
```

## 7. TROUBLESHOOTING

### 7.1 Lỗi CUDA out of memory:
- Giảm batch_size trong config_AE.py
- Giảm max_text_length
- Sử dụng CPU: không truyền -n

### 7.2 Lỗi transformers:
```bash
pip install transformers==4.36.0 tokenizers==0.15.0
```

### 7.3 Lỗi DGL:
```bash
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/repo.html
```

### 7.4 Kiểm tra dữ liệu:
```bash
ls -la data/DBLP/
python -c "from utils import load_data; import config_AE; load_data(config_AE.data_config)"
```

## 8. CẤU TRÚC KẾT QUẢ

```
checkpoint/
├── AE_DBLP_alpha_0.3/
│   ├── model.pth
│   ├── config.json
│   └── training.log
├── AE_DBLP_alpha_0.5/
└── AE_DBLP_alpha_0.7/

results/
├── comparison_results.json
├── grid_search_results.json
└── evaluation_metrics.csv
```

## 9. SCRIPT TỰ ĐỘNG

### Chạy toàn bộ thực nghiệm:
```bash
#!/bin/bash
echo "=== Chạy thực nghiệm AE-CPGNN ==="

# Test implementation
echo "1. Kiểm tra implementation..."
python test_AE.py

# Grid search
echo "2. Tìm α tối ưu..."
python main_AE.py --grid_search

# Compare with baseline
echo "3. So sánh với baseline..."
python evaluate_AE.py --compare

echo "=== Hoàn thành thực nghiệm ==="
```

## 10. KẾT LUẬN

Để chứng minh cải tiến của AE-CPGNN:
1. Chạy baseline CP-GNN
2. Chạy AE-CP-GNN với các α khác nhau
3. So sánh metrics (accuracy, ARI, NMI)
4. Phân tích kết quả và chọn α tối ưu
5. Báo cáo cải tiến về hiệu suất

Cải tiến mong đợi: 2-5% accuracy, cải thiện clustering metrics nhờ thông tin ngữ nghĩa từ DistilBERT. 