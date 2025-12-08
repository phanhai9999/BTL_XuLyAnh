import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = "img/loi1.jpg"   
img = Image.open(path).convert("L")  
img_np = np.array(img)

hist, bins = np.histogram(img_np.flatten(), bins=128, range=[0, 256])
print(hist)

T1 = 200  
T2 = 250 

segmented = np.zeros_like(img_np)
segmented[(img_np < T1)] = 0         
segmented[(img_np >= T1) & (img_np < T2)] = 128   
segmented[(img_np >= T2)] = 255      

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1),plt.imshow(img_np, cmap="gray"),plt.title("Ảnh gốc"),plt.axis("off")
plt.subplot(1, 3, 2),plt.plot(hist),plt.title("Histogram ảnh xám"),plt.xlabel("Mức xám"),plt.ylabel("Số điểm ảnh")
plt.subplot(1, 3, 3),plt.imshow(segmented, cmap="gray"),plt.title("Ảnh sau khi phân vùng đa ngưỡng"),plt.axis("off")
plt.tight_layout() 
plt.show()
