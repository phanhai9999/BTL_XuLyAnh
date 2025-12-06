import numpy as np
from scipy.ndimage import binary_dilation
# pip install scipy
# 0 den 1 trang
A = np.array([
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0]
])

B = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

# Dùng thư viện chuẩn để đảm bảo tính chính xác
# binary_dilation mặc định dùng tâm ở giữa (center)
result = binary_dilation(A, structure=B).astype(int)

print("Kết quả chuẩn:")
print(result)