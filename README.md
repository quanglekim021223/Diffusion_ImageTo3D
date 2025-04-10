# Diffusion Image to 3D

Dự án này sử dụng mô hình Zero123++ với Latent Diffusion để chuyển đổi hình ảnh 2D thành mô hình 3D. Dự án được tối ưu hóa để chạy trên môi trường Kaggle với tài nguyên hạn chế.

## Tính năng

- **Chuyển đổi hình ảnh 2D sang 3D**: Sử dụng mô hình Zero123++ với Latent Diffusion
- **Tạo nhiều góc nhìn**: Tạo ra nhiều góc nhìn mới từ một hình ảnh đầu vào
- **Tái tạo bề mặt 3D**: Sử dụng thuật toán Poisson Surface Reconstruction để tạo mô hình 3D chất lượng cao
- **Áp dụng kết cấu**: Tự động áp dụng kết cấu cho mô hình 3D
- **Xuất nhiều định dạng**: Hỗ trợ xuất mô hình 3D ở nhiều định dạng (GLB, OBJ, PLY, STL)
- **Tối ưu hóa bộ nhớ**: Sử dụng mixed precision và quản lý bộ nhớ để chạy trên môi trường hạn chế tài nguyên
- **Tùy chỉnh góc nhìn**: Cho phép tùy chỉnh góc nâng và góc phương vị để tạo ra các góc nhìn mới
- **Điều chỉnh chất lượng mesh**: Tùy chọn chất lượng mesh (thấp, trung bình, cao) để cân bằng giữa chất lượng và hiệu suất

## Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/diffusion-imageto3d.git
cd diffusion-imageto3d

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Sử dụng

### Sử dụng từ dòng lệnh

```bash
python kaggle_notebook.py --input path/to/input/image.jpg --output path/to/output/model.glb
```

### Các tùy chọn nâng cao

```bash
python kaggle_notebook.py \
  --input path/to/input/image.jpg \
  --output path/to/output/model.glb \
  --num_views 8 \
  --elevation_min -20 \
  --elevation_max 40 \
  --azimuth_min 0 \
  --azimuth_max 360 \
  --num_inference_steps 50 \
  --apply_texture \
  --mesh_quality high
```

### Sử dụng trong Python

```python
from kaggle_notebook import process_image

# Xử lý hình ảnh
output_path = process_image(
    input_path='path/to/input/image.jpg',
    output_path='path/to/output/model.glb',
    num_views=8,
    elevation_range=(-20, 40),
    azimuth_range=(0, 360),
    num_inference_steps=50,
    apply_texture=True,
    mesh_quality='high'
)

print(f"3D model created successfully: {output_path}")
```

## Cách hoạt động

1. **Tiền xử lý hình ảnh**: Hình ảnh đầu vào được tiền xử lý để phù hợp với mô hình
2. **Mã hóa hình ảnh**: Hình ảnh được mã hóa vào không gian latent bằng VAE
3. **Tạo góc nhìn mới**: Sử dụng Latent Diffusion để tạo ra các góc nhìn mới từ không gian latent
4. **Giải mã góc nhìn**: Các góc nhìn mới được giải mã từ không gian latent
5. **Tạo đám mây điểm**: Từ các góc nhìn mới, tạo ra đám mây điểm 3D
6. **Tái tạo bề mặt**: Sử dụng thuật toán Poisson Surface Reconstruction để tạo mô hình 3D
7. **Áp dụng kết cấu**: Tự động áp dụng kết cấu cho mô hình 3D
8. **Xuất mô hình**: Xuất mô hình 3D ở định dạng được chỉ định

## Cấu trúc dự án

```
diffusion-imageto3d/
├── src/
│   ├── inference.py       # Mô hình Zero123++ với Latent Diffusion
│   ├── preprocess.py      # Tiền xử lý hình ảnh
│   └── postprocess.py     # Hậu xử lý và tạo mô hình 3D
├── kaggle_notebook.py     # Notebook chính để chạy trên Kaggle
├── requirements.txt       # Các thư viện cần thiết
└── README.md              # Tài liệu hướng dẫn
```

## Tối ưu hóa hiệu suất

Dự án này đã được tối ưu hóa để chạy trên môi trường Kaggle với tài nguyên hạn chế:

- **Mixed Precision**: Sử dụng mixed precision để giảm sử dụng bộ nhớ và tăng tốc độ xử lý
- **Quản lý bộ nhớ**: Xóa bộ nhớ cache định kỳ và chuyển dữ liệu từ GPU sang CPU khi cần thiết
- **Tùy chỉnh chất lượng mesh**: Cho phép người dùng điều chỉnh chất lượng mesh để cân bằng giữa chất lượng và hiệu suất

## Chất lượng tái tạo 3D

Dự án này sử dụng Latent Diffusion Model để cải thiện chất lượng tái tạo 3D:

- **Không gian latent**: Sử dụng không gian latent để nén thông tin và giảm nhiễu
- **Tái tạo bề mặt nâng cao**: Sử dụng thuật toán Poisson Surface Reconstruction để tạo mô hình 3D chất lượng cao
- **Áp dụng kết cấu**: Tự động áp dụng kết cấu cho mô hình 3D để tăng tính thực tế

## Các mức chất lượng mesh

Dự án hỗ trợ 3 mức chất lượng mesh:

- **Thấp**: Sử dụng convex hull (nhanh nhất, chất lượng thấp nhất)
- **Trung bình**: Sử dụng Poisson reconstruction với depth=7 (cân bằng chất lượng và hiệu suất)
- **Cao**: Sử dụng Poisson reconstruction với depth=9 (chất lượng cao nhất, có thể chậm hơn)

## Giới hạn

- Chất lượng mô hình 3D phụ thuộc vào chất lượng hình ảnh đầu vào
- Mô hình có thể không tái tạo chính xác các chi tiết phức tạp
- Thời gian xử lý phụ thuộc vào kích thước hình ảnh và số lượng góc nhìn

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## Giấy phép

MIT 