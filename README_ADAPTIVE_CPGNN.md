# Adaptive CP-GNN: MLP thích ứng độ dài context path

Dự án này giới thiệu một cải tiến cho mô hình CP-GNN gốc bằng cách thêm một mạng MLP thích ứng để dự đoán độ dài context path tối ưu cho từng nút trong đồ thị. Cải tiến này giúp mô hình tập trung vào context path có ý nghĩa nhất, cải thiện hiệu suất và khả năng giải thích.

## Ý tưởng chính

Thay vì sử dụng một độ dài context path cố định cho tất cả các nút trong đồ thị, Adaptive CP-GNN sử dụng một mạng MLP để dự đoán độ dài tối ưu cho từng nút dựa trên đặc trưng cấu trúc và ngữ cảnh của nút đó. Cách tiếp cận này mang lại nhiều lợi ích:

1. **Hiệu quả tính toán**: Tập trung nguồn lực vào việc tính toán embeddings cho các context path có ý nghĩa nhất
2. **Biểu diễn thích ứng**: Mỗi nút có biểu diễn phù hợp nhất với vai trò cấu trúc của nó trong đồ thị
3. **Khả năng giải thích**: Attention weights cung cấp insights về độ dài context path quan trọng cho từng nút

## Kiến trúc mô hình

Adaptive CP-GNN bao gồm các thành phần chính sau:

### 1. Context Path Length Predictor (CPLP)

MLP này lấy đầu vào là đặc trưng của nút và các đặc trưng cấu trúc (độ, hệ số phân cụm, centrality) để dự đoán attention weights cho các độ dài context path khác nhau.

```python
ContextPathLengthPredictor(
    feature_dim,      # Chiều của node features
    structural_dim,   # Chiều của structural properties
    hidden_dim,       # Chiều ẩn của MLP
    max_path_length,  # Độ dài context path tối đa
    use_structural    # Có sử dụng structural properties không
)
```

### 2. Structural Feature Extractor

Module này trích xuất các đặc trưng cấu trúc của nút từ đồ thị, bao gồm:
- Độ của nút (degree): Số lượng kết nối
- Hệ số phân cụm (clustering coefficient): Mức độ kết nối giữa các láng giềng
- Centrality: Mức độ quan trọng của nút trong đồ thị

### 3. Adaptive Context Aggregation

Module này kết hợp các embeddings từ nhiều độ dài context path khác nhau bằng cách sử dụng attention weights từ CPLP.

### 4. Path Length Regularization

Loss function đặc biệt để điều chỉnh quá trình học của CPLP, khuyến khích:
- Entropy cao (nhiều path lengths khác nhau được sử dụng)
- Diversity cao (các nút khác nhau sử dụng path lengths khác nhau)

## Tích hợp với Dynamic CP-GNN

Adaptive CP-GNN được tích hợp với Dynamic CP-GNN để xử lý đồ thị động theo thời gian, kết hợp cả thông tin cấu trúc, thuộc tính và thời gian.

## Cách sử dụng

### Cài đặt

1. Đảm bảo đã cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Huấn luyện mô hình

Sử dụng script `train_adaptive_cpgnn.py` để huấn luyện Adaptive Dynamic CP-GNN:

```bash
python train_adaptive_cpgnn.py --rnn_type gru --use_temporal_attention --max_path_length 4 --use_structural
```

### Tham số chính

#### Tham số mô hình:
- `--input_dim`: Chiều của node features (mặc định: 128)
- `--hidden_dim`: Chiều ẩn (mặc định: 64)
- `--num_channels`: Số kênh trong CP-GNN (mặc định: 4)
- `--rnn_type`: Loại RNN ('gru' hoặc 'lstm')
- `--use_temporal_attention`: Bật cơ chế temporal attention

#### Tham số đặc biệt cho Adaptive Path:
- `--max_path_length`: Độ dài context path tối đa (mặc định: 4)
- `--use_structural`: Sử dụng structural properties
- `--structural_dim`: Chiều của structural properties (mặc định: 16)
- `--entropy_weight`: Trọng số cho entropy regularization (mặc định: 0.1)
- `--diversity_weight`: Trọng số cho diversity regularization (mặc định: 0.1)
- `--path_reg_weight`: Trọng số cho path length regularization (mặc định: 0.01)

#### Tham số huấn luyện:
- `--epochs`: Số epoch huấn luyện (mặc định: 100)
- `--batch_size`: Kích thước batch (mặc định: 32)
- `--lr`: Learning rate (mặc định: 0.001)
- `--task`: Loại tác vụ ('link_prediction' hoặc 'node_classification')

#### Tham số khác:
- `--visualize_attention`: Trực quan hóa path attention weights

### Ví dụ

#### Dự đoán liên kết động với Adaptive CP-GNN:

```bash
python train_adaptive_cpgnn.py --task link_prediction --rnn_type gru --max_path_length 4 --use_structural --visualize_attention
```

#### Phân loại nút theo thời gian:

```bash
python train_adaptive_cpgnn.py --task node_classification --rnn_type lstm --max_path_length 3 --num_classes 3
```

## Trực quan hóa path attention

Khi bật flag `--visualize_attention`, mô hình sẽ tạo 2 hình ảnh trực quan:

1. **path_attention.png**: Biểu đồ đường thể hiện attention weights trung bình cho mỗi độ dài path ở mỗi thời điểm
2. **preferred_path_length.png**: Biểu đồ histogram thể hiện phân phối độ dài path được ưa thích nhất

Những biểu đồ này giúp hiểu:
- Độ dài path nào quan trọng nhất cho dữ liệu
- Mức độ phân tán trong việc sử dụng các độ dài path khác nhau
- Sự khác biệt về độ dài path ưa thích giữa các thời điểm khác nhau

## Cải tiến và ứng dụng tiềm năng

1. **Sampling động**: Điều chỉnh tiến trình lấy mẫu context path dựa trên attention weights
2. **Transfer learning**: Sử dụng CPLP đã học cho các đồ thị mới với cấu trúc tương tự
3. **Interpretable AI**: Phân tích attention weights để hiểu hơn về cấu trúc đồ thị
4. **Active learning**: Tập trung vào việc lấy mẫu các context paths quan trọng cho các nút khó phân loại

## Lưu ý cài đặt

- Đối với đồ thị lớn, có thể cần điều chỉnh `max_path_length` để cân bằng giữa hiệu suất và hiệu quả tính toán
- Các trọng số regularization (`entropy_weight`, `diversity_weight`, `path_reg_weight`) có thể cần tinh chỉnh tùy theo dữ liệu 