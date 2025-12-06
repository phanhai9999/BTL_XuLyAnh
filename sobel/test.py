"""
Câu 1: Phân vùng ảnh đa ngưỡng. Yêu cầu Cho một ảnh xám chứa
nhiều lớp vùng sáng tối khac nhau. Hãy viết chương trình python thực hiện phân vùng
đa mức theo phương pháp ngưỡng cứng
1. Đọc ảnh xám không dùng opencv
2. tính histogram của ảnh
3. cho 2 ngưỡng giá trị T1 và T2 sao cho chia ảnh thành 3 vùng
vùng tối f(X,Y)<T1
vùng trung bình T1<f(X,Y)<T2
Vùng sáng f(X,Y)>T2
4, gán nhãn mỗi vùng theo các giá trị xám mới: 
vùng tối gán bằng 0
vùng trung bình gán là 128
vùng sáng là 255
5. hiển thị ảnh gốc và sau khi phân vùng
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===========================
# 1. Đọc ảnh xám KHÔNG dùng OpenCV
# ===========================
path = "img/loi1.jpg"   # ----> Đổi ảnh tại đây
img = Image.open(path).convert("L")   # chuyển sang ảnh xám
img_np = np.array(img)

# ===========================
# 2. Tính histogram
# ===========================
hist, bins = np.histogram(img_np.flatten(), bins=256, range=[0, 256])

# ===========================
# 3. Chọn ngưỡng T1, T2
# ===========================
T1 = 80    # ngưỡng vùng tối
T2 = 160   # ngưỡng vùng sáng

# ===========================
# 4. Ánh xạ phân vùng đa mức
# ===========================
segmented = np.zeros_like(img_np)

segmented[(img_np < T1)] = 0          # vùng tối
segmented[(img_np >= T1) & (img_np < T2)] = 128   # vùng trung bình
segmented[(img_np >= T2)] = 255       # vùng sáng

# ===========================
# 5. Hiển thị ảnh gốc & ảnh phân vùng
# ===========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.plot(hist)
plt.title("Histogram ảnh xám")
plt.xlabel("Mức xám")
plt.ylabel("Số điểm ảnh")

plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap="gray")
plt.title("Ảnh sau khi phân vùng đa ngưỡng")
plt.axis("off")

plt.tight_layout()
plt.show()
