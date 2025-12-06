import numpy as np
from scipy.ndimage import binary_erosion

# --- DỮ LIỆU ĐẦU VÀO ---
# 0 đen 1 trắng
A = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1]
])

# 0 đen 1 trắng
B = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])
# --- THỰC HIỆN PHÉP CO (EROSION) ---
# A erosion B
result = binary_erosion(A, structure=B).astype(int)

print("\nKết quả Phép Co (A - B):\n", result)