import numpy as np

def apply_laplacian_edge_padding(image, kernel):
    h, w = image.shape
    k_h, k_w = kernel.shape
    
    # Tính kích thước pad
    pad = k_h // 2
    
    # --- QUAN TRỌNG: Padding mode 'edge' ---
    # mode='edge' sẽ sao chép giá trị viền ra ngoài
    image_padded = np.pad(image, pad_width=pad, mode='edge')
    
    # Tạo ma trận kết quả
    output = np.zeros_like(image, dtype=int)
    
    # Quét qua từng pixel
    for i in range(h):
        for j in range(w):
            # Lấy vùng lân cận (ROI) từ ảnh đã pad
            roi = image_padded[i:i+k_h, j:j+k_w]
            
            # Nhân chập (Kernel đối xứng nên không cần lật)
            k_sum = np.sum(roi * kernel)
            
            output[i, j] = k_sum
            
    return output, image_padded

# --- DỮ LIỆU ĐẦU VÀO ---
image = np.array([
    [1, 1, 5],
    [10, 10, 2],
    [5, 6, 1]
])

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

# --- THỰC HIỆN ---
result, padded_view = apply_laplacian_edge_padding(image, kernel)

# --- IN KẾT QUẢ ---
print("Ảnh sau khi Padding (mode='edge'):")
print(padded_view)
print("\nKết quả đạo hàm bậc 2 (Laplacian):")
print(result)