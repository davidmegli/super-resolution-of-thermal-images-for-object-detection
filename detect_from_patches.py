import os
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

output_path = "results\\"  # output images directory

def iou(box1, box2):
    # Calcola l'area di intersezione
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calcola l'area delle box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calcola l'IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def apply_nms(results, iou_threshold=0.5):
    # Ordina le box per confidenza decrescente
    results = sorted(results, key=lambda x: x.conf[0], reverse=True)
    filtered_boxes = []

    while results:
        chosen_box = results.pop(0)
        filtered_boxes.append(chosen_box)
        results = [box for box in results if iou(chosen_box.xyxy[0], box.xyxy[0]) < iou_threshold]

    return filtered_boxes

model = YOLO("yolo11n.pt")
img_sr = Image.open("inputs\\test\\super_res_image.jpg")  # Carica l'immagine SR
img_sr = np.array(img_sr)  # Converti l'immagine in un array numpy se necessario

patch_size = (640, 480)
#patch_size = (1280, 960)
overlap = 400  # Definisci l'overlap desiderato in pixel

# Ritaglia l'immagine SR in patches
patches = []
height, width = img_sr.shape[:2]

for y in range(0, height, patch_size[1] - overlap):
    for x in range(0, width, patch_size[0] - overlap):
        # Assicurati che il patch non superi i bordi dell'immagine
        x_end = min(x + patch_size[0], width)
        y_end = min(y + patch_size[1], height)

        patch = img_sr[y:y_end, x:x_end]
        patches.append((patch, x, y))  # Salva la posizione del patch

        # Evita di superare i bordi dell'immagine
        if x_end == width:
            break
    if y_end == height:
        break

# Passa ogni patch a YOLO e raccogli i risultati
results = []
for patch, x_offset, y_offset in patches:
    result = model(patch)
    # Aggiusta le coordinate dei bounding box in base alla posizione del patch
    updated_boxes = []
    for box in result[0].boxes:
        new_box_xyxy = box.xyxy.clone()
        new_box_xyxy[:, [0, 2]] += x_offset  # Adjust x
        new_box_xyxy[:, [1, 3]] += y_offset  # Adjust y
        #box.xyxy = new_box_xyxy  # Reassign the adjusted tensor
        #new_box = torch.tensor(new_box_xyxy, dtype=box.xyxy.dtype, device=box.xyxy.device) #coverto il box in tensore
        new_box = {
            "xyxy": new_box_xyxy,
            "conf": box.conf[0].item(),
            "cls": int(box.cls[0].item())
        }
        updated_boxes.append(new_box)
    #results.extend(result[0].boxes)
    results.extend(updated_boxes)

# Applica Non-Maximum Suppression per rimuovere duplicati
final_results = apply_nms(results)  # Usa NMS per combinare i risultati
#final_results = results

## Disegna i bounding box sull'immagine originale
img = Image.open("inputs\\test\\super_res_image.jpg")
for box in final_results:
    # Estrai le coordinate dal tensor del box
    x1, y1, x2, y2 = box[0].tolist()  # Usa `tolist()` per convertire in lista
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

output_path = "results\\"
os.makedirs(output_path, exist_ok=True)
img.save(output_path + "super_res_image_yolo11n.jpg")