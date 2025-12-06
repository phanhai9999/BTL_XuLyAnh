import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def opening_operation(image, kernel):
    eroded_image = binary_dilation(image, structure=kernel).astype(int)
    opened_image = binary_erosion(eroded_image, structure=kernel).astype(int)
    
    return eroded_image, opened_image

# 0 đen 1 trắng
A = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1]
])

# 0 đen 1 trắng
B = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

# --- THỰC HIỆN ---
step1_erosion, final_opening = opening_operation(A, B)
print("Bước 2: Sau khi Giãn lại (Opening Result):\n", final_opening)