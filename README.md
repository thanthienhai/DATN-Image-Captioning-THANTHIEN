# Đề tài: Xây dựng ứng dụng tạo chú thích ảnh tự động

## 1. Mục tiêu thực hiện
- Nghiên cứu và phát triển hệ thống tự động tạo chú thích cho ảnh (image captioning) dựa trên các mô hình học sâu.
- Xây dựng chức năng tạo caption cho bài đăng mạng xã hội dựa trên nội dung ảnh, với đoạn caption phù hợp để đăng lên các nền tảng như Facebook, Instagram.

**Ví dụ**:
- **Ảnh mô tả**: "A serene view of a lake at sunset with mountains in the background."
- **Caption đề xuất cho Instagram**: "Basking in the beauty of this tranquil lake as the sun sets behind the mountains 🌅 #NatureLover #SunsetVibes #PeacefulMoments"

## 2. Phương pháp tiếp cận
- Sử dụng mô hình học sâu kết hợp giữa xử lý ảnh và ngôn ngữ tự nhiên để tạo chú thích cho ảnh.
- Ứng dụng Transformer models như GPT-3 hoặc các mô hình ngôn ngữ tương tự để tạo ra caption phù hợp với giọng văn mạng xã hội.
- Sử dụng thuật toán tìm kiếm vector (FAISS) hoặc các kỹ thuật nhúng (embedding) để tìm kiếm và so khớp ảnh dựa trên đoạn văn bản.

## 3. Những công việc cần thực hiện

### 3.1. Nghiên cứu về Image Captioning
- **Mục tiêu**: Tự động tạo chú thích cho ảnh.
- **Kiến thức cần nghiên cứu**:
  - **CNN** (Convolutional Neural Networks): Trích xuất đặc trưng từ ảnh.
  - **Transformer Models**: Như Vision Transformer (ViT) hoặc CNN + LSTM để kết hợp xử lý ảnh và ngôn ngữ tự nhiên.
  - **Mô hình Attention**: Tập trung vào các phần quan trọng của ảnh khi tạo chú thích.
  - **Datasets**: Bộ dữ liệu ảnh và chú thích (MS COCO, Flickr8k, Flickr30k).
  - **Transfer Learning**: Sử dụng mô hình huấn luyện trước (ResNet, EfficientNet) để cải thiện hiệu suất.

### 3.2. Tạo caption bài đăng mạng xã hội
- **Mục tiêu**: Tạo caption phù hợp với giọng văn mạng xã hội (Facebook, Instagram, Twitter).
- **Kiến thức cần nghiên cứu**:
  - **Transformer Models**: Nghiên cứu các mô hình ngôn ngữ như GPT-3, BERT để tạo caption tự nhiên và phù hợp ngữ cảnh.
  - **Ngữ cảnh mạng xã hội**: Cách sử dụng emoji, hashtags, và Call-to-Action (CTA) phổ biến trên Facebook, Instagram.
  - **Tối ưu hóa caption**: Dựa trên yêu cầu của từng nền tảng (Instagram cần nhiều hashtag hơn, Twitter giới hạn ký tự).

### 3.3. Phát triển ứng dụng sử dụng Python với FastAPI và Streamlit
- **Mục tiêu**: Xây dựng giao diện trực quan cho người dùng.
- **Kiến thức cần nghiên cứu**:
  - **FastAPI**: Xây dựng API bằng Python.
  - **Streamlit**: Tạo giao diện web đơn giản, tương tác với mô hình AI (tải ảnh, nhập mô tả văn bản, hiển thị kết quả tìm kiếm).
  - **API Integration**: Tích hợp mô hình sinh caption và tìm kiếm ảnh vào giao diện Streamlit.
