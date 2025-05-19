# Dynamic CP-GNN: GRU/LSTM cho đồ thị động

Dự án này giới thiệu một cải tiến quan trọng cho mô hình CP-GNN (Channel Propagation Graph Neural Network) bằng cách tích hợp các mô hình GRU/LSTM để xử lý đồ thị động theo thời gian.

## Tổng quan

Dynamic CP-GNN mở rộng mô hình CP-GNN gốc bằng cách thêm khả năng mô hình hóa biến đổi theo thời gian trong đồ thị. Điều này đặc biệt hữu ích cho các ứng dụng như mạng xã hội, hệ thống giao thông, hoặc bất kỳ dữ liệu đồ thị nào thay đổi theo thời gian.

### Các tính năng chính:

- **Temporal Modeling**: Sử dụng GRU hoặc LSTM để nắm bắt sự phụ thuộc thời gian giữa các snapshot đồ thị
- **Temporal Attention**: Cơ chế attention để tập trung vào các thời điểm quan trọng trong quá khứ
- **Adaptive Channel Weighting**: Điều chỉnh động trọng số của các kênh thông tin theo ngữ cảnh thời gian
- **Multi-task Learning**: Hỗ trợ các tác vụ như dự đoán liên kết động và phân loại nút theo thời gian

## Cấu trúc dự án

```
.
├── models/
│   ├── attribute_transformer.py     # Transformer modules cho thuộc tính nút
│   └── dynamic_cp_gnn.py           # Mô hình Dynamic CP-GNN với GRU/LSTM
├── train_dynamic_cpgnn.py          # Script huấn luyện cho Dynamic CP-GNN
└── README.md                       # Tài liệu hướng dẫn
```

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd dynamic-cp-gnn
```

2. Cài đặt các phụ thuộc:
```bash
pip install -r requirements.txt
```

## Cách sử dụng

### Huấn luyện mô hình

Để huấn luyện mô hình Dynamic CP-GNN, sử dụng script `train_dynamic_cpgnn.py`:

```bash
python train_dynamic_cpgnn.py --rnn_type gru --use_temporal_attention --adaptive_channel
```

### Tham số chính

- `--rnn_type`: Loại RNN để sử dụng ('gru' hoặc 'lstm')
- `--use_temporal_attention`: Bật cơ chế temporal attention
- `--adaptive_channel`: Bật cơ chế adaptive channel weighting
- `--task`: Loại tác vụ ('link_prediction' hoặc 'node_classification')
- `--time_steps`: Số bước thời gian sử dụng làm đầu vào
- `--forecast_horizon`: Số bước thời gian dự đoán trong tương lai

Xem tất cả các tùy chọn:
```bash
python train_dynamic_cpgnn.py --help
```

## Ví dụ

### Dự đoán liên kết động

```bash
python train_dynamic_cpgnn.py --task link_prediction --rnn_type gru --time_steps 10 --forecast_horizon 1
```

### Phân loại nút theo thời gian

```bash
python train_dynamic_cpgnn.py --task node_classification --rnn_type lstm --time_steps 5 --num_classes 3
```

## Kiến trúc mô hình

Dynamic CP-GNN bao gồm các thành phần chính sau:

1. **Attribute Transformer**: Xử lý thuộc tính nút bằng mô hình Transformer
2. **CP-GNN**: Xử lý thông tin cấu trúc đồ thị với cơ chế propagation đa kênh
3. **Temporal Module**: GRU hoặc LSTM để học biểu diễn theo thời gian
4. **Temporal Attention**: Cơ chế attention trên các snapshot khác nhau
5. **Adaptive Channel Weight**: Điều chỉnh trọng số động cho các kênh
6. **Task-specific Layers**: Các lớp chuyên biệt cho dự đoán liên kết hoặc phân loại nút

## Các bước tiếp theo và cải tiến

Một số hướng phát triển tiềm năng:

1. Tích hợp với các kỹ thuật đồ thị động tiên tiến khác
2. Thêm khả năng xử lý đồ thị không đều (irregular time intervals)
3. Phát triển phương pháp tăng cường dữ liệu cho đồ thị động
4. Mở rộng sang các bài toán dự đoán đa bước (multi-step forecasting)
5. Tối ưu hiệu suất cho đồ thị quy mô lớn

## Trích dẫn

Nếu bạn sử dụng mã nguồn này trong nghiên cứu của mình, vui lòng trích dẫn:

```
@article{dynamic-cp-gnn,
  title={Dynamic CP-GNN: Using GRU/LSTM for Temporal Graph Neural Networks},
  author={Your Name},
  year={2023}
}
```

## Giấy phép

MIT 