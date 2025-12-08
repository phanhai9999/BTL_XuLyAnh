import numpy as np

def convolution_sobel(image, kx, ky):
    h, w = image.shape
    k_h, k_w = kx.shape
    pad = k_h // 2
    
    # 1. Zero Padding
    padded_img = np.pad(image, pad_width=pad, mode='constant', constant_values=0)
    
    Gx = np.zeros_like(image, dtype=float)
    Gy = np.zeros_like(image, dtype=float)
    
    # --- QUAN TRỌNG: LẬT KERNEL 180 ĐỘ ---
    # Đây là bước tạo nên sự khác biệt giữa Convolution và Correlation
    kx_flipped = np.flip(kx) 
    ky_flipped = np.flip(ky)
    
    # 2. Quét qua từng pixel
    for i in range(h):
        for j in range(w):
            # Lấy vùng lân cận (ROI)
            roi = padded_img[i:i+k_h, j:j+k_w]
            
            # Nhân vùng ảnh với Kernel ĐÃ LẬT
            Gx[i, j] = np.sum(roi * kx_flipped)
            Gy[i, j] = np.sum(roi * ky_flipped)
            
    return Gx, Gy

# --- Dữ liệu đầu vào ---
image = np.array([
    [1, 1, 5],
    [10, 10, 5],
    [5, 5, 10]
])

# Kernel gốc (chưa lật)
kx = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

ky = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# --- Thực thi ---
val_gx, val_gy = convolution_sobel(image, kx, ky)

print("Ma trận Gx (Chuẩn Convolution):\n", val_gx)
print("\nMa trận Gy (Chuẩn Convolution):\n", val_gy)