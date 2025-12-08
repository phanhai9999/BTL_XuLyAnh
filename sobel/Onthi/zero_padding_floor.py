import numpy as np
import math

def apply_custom_filter(image, kernel):
    # 1. Lấy kích thước ảnh và kernel
    h, w = image.shape
    k_h, k_w = kernel.shape
    
    # Tính kích thước viền cần thêm (pad)
    pad = k_h // 2  # Với kernel 3x3 thì pad = 1
    
    # 2. Thêm viền số 0 (Zero Padding)
    image_padded = np.pad(image, pad_width=pad, mode='constant', constant_values=0)
    
    # Tạo ma trận kết quả
    output = np.zeros((h, w), dtype=int)
    
    # Tính tổng trọng số của kernel (để chia trung bình)
    kernel_weight = np.sum(kernel) # = 10
    
    # 3. Quét qua từng pixel của ảnh gốc
    for i in range(h):
        for j in range(w):
            # Trích xuất vùng lân cận (ROI) từ ảnh đã padA
            # Vùng này có kích thước bằng kernel (3x3)
            # Lưu ý: Do đã pad, toạ độ trên ảnh pad sẽ lệch đi 'pad' đơn vị
            roi = image_padded[i:i+k_h, j:j+k_w]
            
            # Nhân chập: Nhân từng phần tử ROI với Kernel rồi cộng lại
            k_sum = np.sum(roi * kernel)
            
            # Chia cho tổng trọng số (10) và làm tròn xuống (floor)
            # 190 / 10 = 19
            # 260 / 10 = 26 ...
            val = math.floor(k_sum / kernel_weight)
            
            output[i, j] = val
            
    return output

# --- DỮ LIỆU ĐẦU VÀO ---
image = np.array([
    [10, 20, 30],
    [80, 70, 30],
    [40, 60, 50]
])

kernel = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 1]
])

# --- THỰC HIỆN ---
result = apply_custom_filter(image, kernel)

# --- IN KẾT QUẢ ---
print("Ma trận ảnh gốc:\n", image)
print("\nMa trận sau khi lọc (Zero Pad + Chia 10 + Floor):")
print(result)