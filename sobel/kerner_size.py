import numpy as np
import math

def max_filter_zero_padding(image, kernel_size=3):
    h, w = image.shape
    
    # 1. Tính kích thước padding
    pad = kernel_size // 2
    
    # 2. Zero Padding (Thêm viền 0)
    # constant_values=0 là mặc định, nhưng viết rõ để dễ hiểu
    image_padded = np.pad(image, pad_width=pad, mode='constant', constant_values=0)
    
    # Tạo ma trận kết quả
    output = np.zeros_like(image, dtype=int)
    
    # 3. Quét qua từng pixel
    for i in range(h):
        for j in range(w):
            # Lấy vùng lân cận (ROI) kích thước 3x3 từ ảnh đã pad
            roi = image_padded[i : i + kernel_size, j : j + kernel_size]
            
            # Tìm giá trị cực đại (MAX) trong vùng
            max_val = np.max(roi)
            
            # Hàm floor (theo yêu cầu của bạn)
            # Dù max của số nguyên vẫn là số nguyên, nhưng mình vẫn thêm vào cho đúng để bài
            output[i, j] = math.floor(max_val)
            
    return output, image_padded

# --- DỮ LIỆU ĐẦU VÀO ---
image = np.array([
    [1, 4, 8],
    [2, 2, 10],
    [5, 5, 12]
])

# --- THỰC HIỆN ---
result, padded_view = max_filter_zero_padding(image, kernel_size=3)

# --- IN KẾT QUẢ ---
print("Ảnh sau khi Zero Padding:")
print(padded_view)
print("\nKết quả sau bộ lọc Max Filter:")
print(result)