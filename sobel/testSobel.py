# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # =======================================================
# # 1. HÀM TÍNH GRADIENT SOBEL
# # =======================================================
# def sobel_gradients(gray: np.ndarray):
#     # Kernel Sobel theo X và Y
#     Kx = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]], dtype=np.float32)

#     Ky = np.array([[ 1,  2,  1],
#                    [ 0,  0,  0],
#                    [-1, -2, -1]], dtype=np.float32)

#     # Tính gradient theo X và Y bằng filter 2D
#     Gx = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=Kx, borderType=cv2.BORDER_DEFAULT)
#     Gy = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=Ky, borderType=cv2.BORDER_DEFAULT)

#     # Magnitude = sqrt(Gx^2 + Gy^2)
#     mag = np.hypot(Gx, Gy)

#     # Tính hướng biên (0..180 độ)
#     angle = (np.rad2deg(np.arctan2(Gy, Gx)) + 180.0) % 180.0

#     return Gx, Gy, mag, angle


# # =======================================================
# # 2. HÀM LẤY MAGNITUDE + MASK (nếu threshold != None)
# # =======================================================
# def sobel_edges(gray: np.ndarray, threshold):
#     _, _, mag, _ = sobel_gradients(gray)

#     # Chuẩn hoá magnitude về 0..255
#     mmax = float(mag.max())
#     if mmax > 0:
#         mag_uint8 = (mag / mmax * 255.0).clip(0, 255).astype(np.uint8)
#     else:
#         mag_uint8 = np.zeros_like(mag, dtype=np.uint8)

#     # Nếu không threshold → trả về ảnh magnitude
#     if threshold is None:
#         return mag_uint8
    
#     # Ngược lại → tạo mask nhị phân
#     mask = (mag >= threshold * mmax).astype(np.uint8) * 255
#     return mag_uint8, mask


# # =======================================================
# # 3. ĐỌC ẢNH & CHẠY SOBEL
# # =======================================================
# img = cv2.imread("img/group.jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = img_gray.astype(np.float32)

# # Ảnh magnitude (không threshold)
# mag = sobel_edges(gray, None)

# # Magnitude + Mask biên mạnh
# mag_t, mask = sobel_edges(gray, threshold=0.25)

# # =======================================================
# # 4. HIỂN THỊ KẾT QUẢ
# # =======================================================
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Origin")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(mag_t, cmap='gray')
# plt.title("Sobel Magnitude (threshold)")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(mask, cmap='gray')
# plt.title("Sobel Binary Mask")
# plt.axis("off")

# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =======================================================
# 1. HÀM TÍNH GRADIENT SOBEL (GIỮ NGUYÊN)
# =======================================================
def sobel_gradients(gray: np.ndarray):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    Gx = cv2.filter2D(gray, cv2.CV_32F, Kx)
    Gy = cv2.filter2D(gray, cv2.CV_32F, Ky)

    mag = np.hypot(Gx, Gy)
    angle = (np.rad2deg(np.arctan2(Gy, Gx)) + 180) % 180
    return Gx, Gy, mag, angle

# =======================================================
# 2. LẤY MAGNITUDE + MASK (GIỮ NGUYÊN)
# =======================================================
def sobel_edges(gray, threshold=None):
    _, _, mag, _ = sobel_gradients(gray)
    mmax = np.max(mag)
    if mmax > 0:
        mag_norm = (mag / mmax * 255).astype(np.uint8)
    else:
        mag_norm = np.zeros_like(gray, dtype=np.uint8)

    if threshold is None:
        return mag_norm
    
    mask = (mag >= threshold * mmax).astype(np.uint8) * 255
    return mag_norm, mask

# =======================================================
# 3. PHÁT HIỆN VÙNG MỐI HÀN (ROI)
# =======================================================
img = cv2.imread("img/loi1.jpg") 
# Lưu ý: Đảm bảo đường dẫn ảnh đúng. Nếu ảnh quá lớn nên resize lại.
if img is None:
    print("Lỗi: Không đọc được ảnh!")
    exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur nhẹ hơn chút để giữ chi tiết lỗi
smooth = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sobel để lấy biên (giữ nguyên logic của bạn)
mag, edge_mask = sobel_edges(smooth.astype(np.float32), threshold=0.15) 
# Hạ threshold xuống 0.15 để bắt biên tốt hơn nếu ảnh mờ

# Morphology: Đóng lỗ hổng -> Nối liền -> Lấp đầy
kernel = np.ones((5, 5), np.uint8)
weld_region = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
weld_region = cv2.dilate(weld_region, kernel, iterations=2) # Dilate mạnh hơn để bao trùm

# Tìm contour lớn nhất làm Mask Mối Hàn
contours, _ = cv2.findContours(weld_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
weld_mask_final = np.zeros_like(img_gray)

if contours:
    c_max = max(contours, key=cv2.contourArea)
    cv2.drawContours(weld_mask_final, [c_max], -1, 255, thickness=-1)

# ==================================================================
# 4. PHÁT HIỆN LỖI (CẢI TIẾN QUAN TRỌNG)
# ==================================================================

# BƯỚC 1: Co vùng tìm kiếm (Erode)
# Vì biên mối hàn thường có bóng đen (shadow), dễ nhầm là lỗi. 
# Ta co mask vào trong một chút để tránh biên.
search_mask = cv2.erode(weld_mask_final, kernel, iterations=4)

# BƯỚC 2: Cân bằng sáng cục bộ (Optional - giúp nổi bật lỗi trên nền)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced_gray = clahe.apply(smooth)

# BƯỚC 3: Dùng ADAPTIVE THRESHOLD thay vì OTSU
# BlockSize: 15, C: 4 (Bạn có thể chỉnh C tăng lên nếu muốn bắt lỗi gắt hơn)
# Detect các điểm đen (lỗ khí/nứt) trên nền sáng
defect_candidates = cv2.adaptiveThreshold(
    enhanced_gray, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, # Đảo ngược: Lỗi (đen) thành trắng để dễ đếm
    21, # Block Size (phải là số lẻ)
    5   # Constant C (C càng lớn thì càng ít nhiễu nhưng có thể mất lỗi mờ)
)

# BƯỚC 4: Chỉ giữ lỗi nằm TRONG vùng mối hàn đã co (search_mask)
defect_on_weld = cv2.bitwise_and(defect_candidates, defect_candidates, mask=search_mask)

# Lọc nhiễu hạt tiêu (Morphology Open)
defect_on_weld = cv2.morphologyEx(defect_on_weld, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

# =======================================================
# 5. LỌC LỖI THEO DIỆN TÍCH & HIỂN THỊ
# =======================================================
contours_err, _ = cv2.findContours(defect_on_weld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_draw = img.copy()
final_defect_mask = np.zeros_like(img_gray)
total_defect_area = 0

min_area_defect = 10   # Bỏ qua chấm li ti
max_area_defect = 500  # Bỏ qua mảng quá lớn (thường là sai sót mask)

for c in contours_err:
    area = cv2.contourArea(c)
    if min_area_defect < area < max_area_defect:
        total_defect_area += area
        # Vẽ contour lỗi lên ảnh gốc
        cv2.drawContours(img_draw, [c], -1, (0, 0, 255), 2) # Màu đỏ
        # Vẽ vòng tròn bao quanh để dễ nhìn
        (x,y), radius = cv2.minEnclosingCircle(c)
        cv2.circle(img_draw, (int(x),int(y)), int(radius)+3, (0, 255, 255), 1) # Màu vàng
        # Cập nhật vào mask kết quả cuối cùng
        cv2.drawContours(final_defect_mask, [c], -1, 255, thickness=-1)

weld_area = cv2.countNonZero(weld_mask_final)

print(f"Diện tích mối hàn: {weld_area}")
print(f"Diện tích vùng lỗi thực tế: {total_defect_area}")
if weld_area > 0:
    print(f"Tỷ lệ lỗi: {(total_defect_area/weld_area)*100:.2f}%")

# =======================================================
# 6. HIỂN THỊ KẾT QUẢ
# =======================================================
plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Ảnh Gốc")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(weld_mask_final, cmap='gray')
plt.title("Vùng Mối Hàn (Mask)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(defect_on_weld, cmap='gray')
plt.title("Adaptive Thresh (Raw)")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
plt.title("Kết Quả Phát Hiện")
plt.axis("off")

results_text = (f"Area Bead: {weld_area}\nArea Defect: {total_defect_area}")
plt.figtext(0.5, 0.05, results_text, ha="center", fontsize=12, 
            bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.tight_layout()
plt.show()
