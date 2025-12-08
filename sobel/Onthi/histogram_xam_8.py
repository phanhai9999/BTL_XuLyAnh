import numpy as np

def histogram_equalization(image, L):
    # Lấy kích thước ảnh
    M, N = image.shape
    total_pixels = M * N
    
    # Bước 1: Tính Histogram (đếm số lần xuất hiện của mỗi mức xám)
    # unique: các giá trị mức xám có trong ảnh
    # counts: số lần xuất hiện tương ứng
    unique, counts = np.unique(image, return_counts=True)
    
    # Tạo từ điển để tra cứu nhanh số lượng của mỗi mức xám
    hist_dict = dict(zip(unique, counts))
    
    # Tạo mảng nk chứa số lượng pixel cho tất cả các mức từ 0 đến L-1
    nk = np.array([hist_dict.get(i, 0) for i in range(L)])
    
    # Bước 2: Tính xác suất xuất hiện (PMF)
    pr = nk / total_pixels
    
    # Bước 3: Tính hàm phân phối tích lũy (CDF)
    cdf = np.cumsum(pr)
    
    # Bước 4: Tính giá trị mức xám mới
    # Công thức: round((L-1) * cdf)
    sk = np.round((L - 1) * cdf).astype(int)
    
    # Bước 5: Ánh xạ giá trị cũ sang giá trị mới
    equalized_image = np.zeros_like(image)
    for i in range(M):
        for j in range(N):
            old_val = image[i, j]
            equalized_image[i, j] = sk[old_val]
            
    return equalized_image, sk

# Dữ liệu đầu vào
# image = np.array([
#     [1, 2, 7],
#     [0, 2, 5],
#     [1, 1, 6]
# ])

image = np.array([
    [1, 2, 7],
    [0, 0, 5],
    [1, 1, 6]
])

L = 8 # Số mức xám

# Thực hiện cân bằng
new_image, mapping = histogram_equalization(image, L)

print("Ma trận ảnh gốc:\n", image)
print("-" * 20)
print("Ma trận ảnh sau khi cân bằng:\n", new_image)