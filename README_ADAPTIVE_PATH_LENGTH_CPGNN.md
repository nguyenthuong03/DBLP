# Adaptive Path Length CP-GNN

Mô hình Adaptive CP-GNN với MLP thích ứng độ dài context path (Adaptive Path Length CP-GNN) là một cải tiến của mô hình CP-GNN nhằm nâng cao khả năng xử lý các đường dẫn ngữ cảnh (context path) có độ dài khác nhau trong đồ thị.

## Tổng quan

Trong mô hình CP-GNN gốc, các context path thường được xử lý bằng các phương pháp cố định, không phản ánh đầy đủ tầm quan trọng của độ dài path. Adaptive Path Length CP-GNN giải quyết vấn đề này bằng cách sử dụng các mạng MLP thích ứng có kiến trúc thay đổi theo độ dài của context path.

### Các tính năng chính:

- **Kiến trúc MLP thích ứng**: Tự động điều chỉnh kiến trúc MLP (số lớp, kích thước) dựa trên độ dài context path
- **Cơ chế Attention dựa trên độ dài path**: Trọng số attention cho các path dựa trên độ dài của chúng
- **Path Encoder động**: Encoder đặc biệt cho các path dựa trên độ dài, tự động điều chỉnh số tham số và chiều sâu mạng
- **Pooling theo độ dài path**: Cơ chế pooling riêng biệt cho từng nhóm độ dài path (ngắn, trung bình, dài)
- **Phương pháp tổng hợp biểu diễn**: Kết hợp các biểu diễn từ các nhóm độ dài path khác nhau bằng cơ chế fusion module

## Cấu trúc dự án

```
.
├── models/
│   ├── adaptive_path_length_cpgnn.py    # Mô hình Adaptive Path Length CP-GNN
│   └── adaptive_cp_gnn.py               # Mô hình Adaptive CP-GNN gốc
├── utils/
│   └── visualization.py                 # Tiện ích trực quan hóa kết quả
├── train_adaptive_path_length_cpgnn.py  # Script huấn luyện cho mô hình mới
└── README_ADAPTIVE_PATH_LENGTH_CPGNN.md # Tài liệu hướng dẫn
```

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Cài đặt các phụ thuộc:
```bash
pip install -r requirements.txt
```

## Cách sử dụng

### Huấn luyện mô hình

Để huấn luyện mô hình Adaptive Path Length CP-GNN, sử dụng script `train_adaptive_path_length_cpgnn.py`:

```bash
python train_adaptive_path_length_cpgnn.py --dataset DBLP --task node_classification --hidden_dims 64,64,64 --max_path_length 6 --use_pooling
```

### Tham số chính

- `--dataset`: Tên tập dữ liệu
- `--task`: Loại tác vụ ('node_classification' hoặc 'link_prediction')
- `--dynamic`: Sử dụng mô hình đồ thị động
- `--hidden_dim`: Kích thước hidden dimension
- `--hidden_dims`: List chiều hidden cho adaptive MLP (định dạng: số,số,số)
- `--max_path_length`: Độ dài tối đa của context path
- `--num_channels`: Số kênh propagation
- `--use_pooling`: Bật pooling thích ứng với độ dài path
- `--rnn_type`: Loại RNN cho mô hình động ('gru' hoặc 'lstm')
- `--visualize`: Trực quan hóa phân phối độ dài path và cấu trúc mô hình

Xem tất cả các tùy chọn:
```bash
python train_adaptive_path_length_cpgnn.py --help
```

## Kiến trúc mô hình

### Mô hình AdaptivePathLengthCPGNN

Mô hình Adaptive Path Length CP-GNN bao gồm các thành phần chính sau:

1. **AdaptiveLengthMLP**: MLP thay đổi kiến trúc dựa trên độ dài context path
   - Path ngắn (1-2): MLP nhỏ gọn, ít lớp
   - Path trung bình (3-4): MLP với số lớp cân đối
   - Path dài (5+): MLP sâu hơn, nhiều lớp hơn

2. **PathLengthAttention**: Cơ chế attention đặc biệt cho các path dựa trên độ dài
   - Attention đa đầu (multi-head) cho các embeddings ở các độ dài path khác nhau
   - Dự đoán tầm quan trọng toàn cục của mỗi độ dài path

3. **LengthAdaptivePooling**: Phương pháp pooling khác nhau cho các độ dài path
   - Path ngắn: max pooling đơn giản
   - Path trung bình: attention-based pooling
   - Path dài: hierarchical pooling

4. **Path Length Predictor**: Dự đoán độ dài path tối ưu cho mỗi node

### Mô hình AdaptivePathLengthDynamicCPGNN

Mở rộng của mô hình trên với khả năng xử lý đồ thị động:

1. **Temporal Module**: GRU hoặc LSTM để học biểu diễn theo thời gian
2. **Temporal Attention**: Cơ chế attention trên các snapshot đồ thị khác nhau

## Ví dụ

### Phân loại nút (Node Classification)

```bash
python train_adaptive_path_length_cpgnn.py --dataset DBLP --task node_classification --hidden_dims 64,64,64 --max_path_length 6 --use_pooling --visualize
```

### Dự đoán liên kết (Link Prediction)

```bash
python train_adaptive_path_length_cpgnn.py --dataset DBLP --task link_prediction --hidden_dims 64,64,64 --max_path_length 6 --use_pooling --visualize
```

### Mô hình động (Dynamic Model)

```bash
python train_adaptive_path_length_cpgnn.py --dataset DBLP --task node_classification --dynamic --rnn_type gru --use_temporal_attention --time_steps 10
```

## Trực quan hóa kết quả

Mô hình Adaptive Path Length CP-GNN cung cấp nhiều công cụ trực quan hóa:

1. **Path Length Distribution**: Phân phối trọng số của các độ dài path khác nhau
2. **Adaptive MLP Structure**: Cấu trúc MLP khác nhau cho từng độ dài path
3. **Graph with Path Lengths**: Đồ thị với các node được tô màu theo độ dài path tối ưu
4. **Training Curves**: Đường cong huấn luyện và validation

## Tài liệu tham khảo

1. CP-GNN: Channel-Projection Graph Neural Networks for semi-supervised node classification
2. Adaptive GNN: Understanding different structural patterns in graphs
3. Hierarchical Heterogeneous Graph Representation Learning
4. Dynamic Graph Neural Networks with Temporal Self-Attention

## Đóng góp và phát triển

Một số hướng phát triển tiềm năng:

1. Tích hợp với các kỹ thuật embedding đồ thị hiệu quả hơn
2. Mở rộng cho các đồ thị không đồng nhất (heterogeneous graphs)
3. Tối ưu hiệu suất cho đồ thị quy mô lớn
4. Tích hợp các kỹ thuật self-supervised learning cho đồ thị
5. Kết hợp với transformer cho xử lý graph sequence

## Giấy phép

MIT 