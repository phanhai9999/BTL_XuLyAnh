""" 2
Câu 1: Lọc trung vị 
Hãy viết chương trình python thực hiện phân vùng đa mức theo phương pháp ngưỡng cứng
1. Đọc ảnh xám không dùng opencv ( dùng PIL)
2. Tự viết hàm pad ảnh (zero padding hoặc replicate-padding)
3. Tự cài đặt bộ lọc trung vị (median filter) kích thước 3x3
Với mỗi vị trí (i,j) trong ảnh đầu vào, ta lấy vùng lân cận 3x3 quanh nó,
sắp xếp các giá trị điểm ảnh trong vùng này theo thứ tự tăng dần,
thay pixel trung tâm bằng median
4, Thử nghiệm với ảnh có nhiễu muối tiêu : tự viết hàm thêm nhiễu salt & pepper ( tỉ lệ 5% - 10%)
5. hiển thị ảnh gốc, ảnh thêm nhiễu và ảnh đã khử nhiễu

1. blackman có độ rò rỉ thấp nhất
2. vùng tâm phổ sẽ bị loại bỏ
3. hướng chéo 45 và 135 độ  để tính gradient theo hướng (roberts)
4. Điểm yếu lớn nhất của toán tử Roberts là dễ khuếch đại biến thiên nhỏ vì chỉ dùng vùng 2x2

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# =========================================
# 1. Đọc ảnh xám bằng PIL
# =========================================
path = "img/loi1.jpg"   # ----> Đổi ảnh tại đây
img = Image.open(path).convert("L")
img_np = np.array(img)


# =========================================
# 2. Hàm pad ảnh (zero padding)
# =========================================
def pad_image(img, pad=1, mode="zero"):
    h, w = img.shape
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=img.dtype)

    if mode == "zero":
        padded[pad:pad+h, pad:pad+w] = img
        return padded
    
    elif mode == "replicate":
        # copy vùng chính giữa
        padded[pad:pad+h, pad:pad+w] = img

        # pad viền trên và dưới
        padded[:pad, pad:pad+w] = img[0:1, :]
        padded[pad+h:, pad:pad+w] = img[-1:, :]

        # pad trái/phải
        padded[:, :pad] = padded[:, pad:pad+1]
        padded[:, pad+w:] = padded[:, pad+w-1:pad+w]

        return padded


# =========================================
# 3. Median Filter 3x3 tự cài đặt
# =========================================
def median_filter(img):
    h, w = img.shape
    padded = pad_image(img, pad=1, mode="replicate")
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            # Lấy vùng 3x3
            region = padded[i:i+3, j:j+3].flatten()
            median_val = np.median(region)
            output[i, j] = median_val

    return output


# =========================================
# 4. Hàm thêm nhiễu Salt & Pepper
# =========================================
def add_salt_pepper(img, amount=0.05):
    noisy = img.copy()
    h, w = img.shape
    num_noise = int(amount * h * w)

    # Salt (255)
    for _ in range(num_noise // 2):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        noisy[x, y] = 255

    # Pepper (0)
    for _ in range(num_noise // 2):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        noisy[x, y] = 0

    return noisy


# =========================================
# 5. Chạy thử nghiệm
# =========================================
noisy_img = add_salt_pepper(img_np, amount=0.08)   # thêm 8% noise
denoised = median_filter(noisy_img)

# =========================================
# 6. Hiển thị kết quả
# =========================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap='gray')
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title("Ảnh thêm nhiễu (Salt & Pepper)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised, cmap='gray')
plt.title("Ảnh sau khi lọc median")
plt.axis("off")

plt.tight_layout()
plt.show()
