""" 4
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


1. (toán tử Laplace) điểm chú ý nhất: các điểm giao ko của đạo hàm bậc 2 
2. toán tử prewitt và sobel: cùng hướng gradient nhưng sobel có trọng số lớn hơn ở tâm
7. 1 ảnh 8 bit có histogram ở 0-70, -> độ tương pphản ở vùng tối tăng và vùng sáng giảm
8. biến đổ power-law : gamma <1 : tăng độ sáng,, gamma >1 : giảm độ sáng -> ảnh sáng hơn ảnh gốc
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ===========================
# 1. Đọc ảnh xám KHÔNG dùng OpenCV
# ===========================
# sử dụng thư viện PIL để đọc ảnh và chuyển sang ảnh xám
path = "img/loi1.jpg"   # ----> Đổi ảnh tại đây 
img = Image.open(path).convert("L")   # chuyển sang ảnh xám
img_np = np.array(img) # chuyển ảnh PIL thành mảng numpy

# ===========================
# 2. Tính histogram
# ===========================
# tính histogram sử dụng numpy bằng công thức histogram
# flatten là chuyển mảng 2D thành 1D để tính histogram và đơn giản hoá
# công thức histogram: đếm số điểm ảnh tại mỗi mức xám
# bin là số lượng thùng (mức xám)
# range là khoảng giá trị mức xám
# hist là mảng chứa số điểm ảnh tại mỗi mức xám
hist, bins = np.histogram(img_np.flatten(), bins=256, range=[0, 256]) # tính histogram bằng numpy
#in ra histogram
print(hist)

# ===========================
# 3. Chọn ngưỡng T1, T2
# ===========================
T1 = 80    # ngưỡng vùng tối
T2 = 160   # ngưỡng vùng sáng

# ===========================
# 4. Ánh xạ phân vùng đa mức
# ===========================
# segmented là ảnh phân vùng đa mức
# zeros_like: tạo mảng numpy rỗng có cùng kích thước với img_np
segmented = np.zeros_like(img_np)  # sử dụng mảng numpy rỗng có cùng kích thước với ảnh gốc
# sau đó gán nhãn cho từng vùng
segmented[(img_np < T1)] = 0          # vùng tối
segmented[(img_np >= T1) & (img_np < T2)] = 128   # vùng trung bình
segmented[(img_np >= T2)] = 255       # vùng sáng

# in
# ===========================
# 5. Hiển thị ảnh gốc & ảnh phân vùng
# ===========================
# sử dụng matplotlib để hiển thị ảnh
# tạo figure với kích thước 12x5 inches
plt.figure(figsize=(12, 5))

# hiển thị ảnh gốc chuyển thành xám
plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Ảnh gốc")
plt.axis("off")

# hiển thị histogram  
plt.subplot(1, 3, 2)
plt.plot(hist) # hiển thị đồ thị histogram
plt.title("Histogram ảnh xám")
plt.xlabel("Mức xám") # trục x
plt.ylabel("Số điểm ảnh") # trục y

# hiển thị ảnh sau khi phân vùng đa ngưỡng
plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap="gray")
plt.title("Ảnh sau khi phân vùng đa ngưỡng")
plt.axis("off") # ẩn trục toạ độ

plt.tight_layout() # căn chỉnh bố cục
plt.show()
