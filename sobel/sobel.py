import cv2
import numpy as np
import matplotlib.pyplot as plt

# =======================================================
# 1. HÀM TÍNH GRADIENT SOBEL
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
# 2. LẤY MAGNITUDE + MASK
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
# 3. PHÁT HIỆN VÙNG MỐI HÀN (khoanh đúng bead chính)
# =======================================================
img = cv2.imread("img/loi1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Làm mượt ảnh để giảm nhiễu
smooth = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Sobel để lấy biên
# test them chi so threshold 0.22
mag, edge_mask = sobel_edges(smooth.astype(np.float32), threshold=0.22)
# threshold nhỏ hơn 0.25 để biên mối hàn liền mạch hơn

# Morphology để lấp kín vùng mối hàn
kernel = np.ones((5, 5), np.uint8)
weld_region = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
weld_region = cv2.dilate(weld_region, kernel, iterations=1)

# Giữ lại thành phần (contour) lớn nhất – chính là mối hàn
contours, _ = cv2.findContours(weld_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
weld_region = np.zeros_like(weld_region)
if contours:
    c_max = max(contours, key=cv2.contourArea)
    cv2.drawContours(weld_region, [c_max], -1, 255, thickness=-1)


# ==================================================================
# 4. PHÁT HIỆN LỖI (vùng có độ sáng bất thường)
# ==================================================================
# Dùng threshold tự động Otsu để phát hiện lỗi
_, defect_mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Chỉ giữ lỗi nằm trên vùng mối hàn (AND)
defect_on_weld = cv2.bitwise_and(defect_mask, defect_mask, mask=weld_region)

# Loại bỏ nhiễu nhỏ
defect_on_weld = cv2.morphologyEx(defect_on_weld, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# =======================================================
# 5. TÍNH DIỆN TÍCH MỐI HÀN & LỖI
# =======================================================
weld_area = cv2.countNonZero(weld_region)
defect_area = cv2.countNonZero(defect_on_weld)

print("Diện tích mối hàn:", weld_area)
print("Diện tích vùng lỗi:", defect_area)

# Vẽ contour lỗi lên ảnh gốc
contours, _ = cv2.findContours(defect_on_weld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_draw = img.copy()
cv2.drawContours(img_draw, contours, -1, (0, 0, 255), 2)

# =======================================================
# 6. HIỂN THỊ KẾT QUẢ
# =======================================================
plt.figure(figsize=(16, 6))
plt.subplot(1, 4, 1),plt.imshow(img[..., ::-1]),plt.title("Ảnh Gốc"),plt.axis("off")
plt.subplot(1, 4, 2),plt.imshow(weld_region, cmap="gray"),plt.title("Vùng Mối Hàn"),plt.axis("off")
plt.subplot(1, 4, 3),plt.imshow(defect_on_weld, cmap="gray"),plt.title("Lỗi Trên Mối Hàn"),plt.axis("off")
plt.subplot(1, 4, 4),plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)),plt.title("Kết Quả Cuối"),plt.axis("off")
results_text = (f"Diện tích mối hàn: {weld_area}\n Diện tích lỗi: {defect_area}")
plt.figtext(0.01, 0.01, results_text,ha="left", fontsize=12, color="black",bbox={"facecolor": "white", "alpha": 0.7, "pad": 5})
plt.tight_layout()
plt.show()
