from ultralytics import YOLO
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

# --- YOLO ---
model_yolo = YOLO("yolov8n.pt")

# Medir tiempo YOLO
t0 = time.time()
results = model_yolo("codelabs6/cars.webp")
t1 = time.time()
yolo_time = (t1 - t0) * 1000  # ms

r = results[0]
yolo_boxes = [box.xyxy.squeeze().tolist() for box in r.boxes]
yolo_scores = [float(box.conf) for box in r.boxes]
yolo_classes = [int(box.cls) for box in r.boxes]

# NO FILTRAMOS NADA → tomamos todas las detecciones
yolo_objs = [(b, s, c) for b, s, c in zip(yolo_boxes, yolo_scores, yolo_classes)]

# --- SSD ---
weights = SSD300_VGG16_Weights.DEFAULT
model_ssd = ssd300_vgg16(weights=weights).eval()
preprocess = weights.transforms()

img = Image.open("codelabs6/cars.webp").convert("RGB")
x = preprocess(img).unsqueeze(0)

# Medir tiempo SSD
t0 = time.time()
with torch.no_grad():
    out = model_ssd(x)[0]
t1 = time.time()
ssd_time = (t1 - t0) * 1000  # ms

ssd_boxes = out["boxes"].tolist()
ssd_scores = out["scores"].tolist()
ssd_labels = out["labels"].tolist()

# NO FILTRAMOS, tomamos todas las detecciones con confianza > 0.5
ssd_objs = [(b, s, l) for b, s, l in zip(ssd_boxes, ssd_scores, ssd_labels) if s > 0.5]

# --- FUNCIÓN IoU ---
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if (boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

# --- EMPAREJAR TODAS LAS CAJAS ---
pairs = []
best_match = None
best_iou = 0.0

for i, (yolo_box, yolo_conf, yolo_cls) in enumerate(yolo_objs):
    for j, (ssd_box, ssd_conf, ssd_cls) in enumerate(ssd_objs):
        current_iou = iou(yolo_box, ssd_box)
        if current_iou > best_iou:
            best_iou = current_iou
            best_match = (i, j, yolo_conf, ssd_conf)
        if current_iou > 0.3:  # solo pares con buen solapamiento
            pairs.append((i, j, yolo_conf, ssd_conf, current_iou))

# --- GRAFICAR ---
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(img)

for (i, j, yc, sc, current_iou) in pairs:
    yolo_box = yolo_objs[i][0]
    ssd_box = ssd_objs[j][0]

    # Caja YOLO (roja)
    x1, y1, x2, y2 = yolo_box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor="red", facecolor="none"))
    ax.text(x1, y1 - 5, f"YOLO[{i}] conf:{yc:.2f}", color="red", fontsize=8)

    # Caja SSD (azul)
    x1, y1, x2, y2 = ssd_box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor="blue", facecolor="none"))
    ax.text(x1, y2 + 10, f"SSD[{j}] conf:{sc:.2f} IoU:{current_iou:.2f}",
            color="blue", fontsize=8)

ax.set_title("Coincidencias YOLO vs SSD (IoU > 0.3)")
plt.axis("off")
plt.show()

# --- MÉTRICAS PARA TABLA ---
yolo_count = len(yolo_objs)
ssd_count = len(ssd_objs)
ious = [p[4] for p in pairs]
iou_promedio = np.mean(ious) if ious else 0.0

print("\n=== RESULTADOS PARA TABLA ===")
print(f"Tiempo YOLO (ms): {yolo_time:.2f}")
print(f"Tiempo SSD (ms): {ssd_time:.2f}")
print(f"Número de objetos detectados YOLO: {yolo_count}")
print(f"Número de objetos detectados SSD: {ssd_count}")
print(f"IoU promedio: {iou_promedio:.3f}")