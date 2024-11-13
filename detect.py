from ultralytics import YOLO
import os
from pathlib import Path

#library to get image size
from PIL import Image

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

input_path = "inputs\\test\\"  # input images directory
images = [os.path.join(input_path, img) for img in os.listdir(input_path) if img.endswith((".jpg", ".jpeg", ".png"))] # list of image paths
output_path = "results\\"  # output images directory
'''
image_size = 640
# Get image size
with Image.open(images[0]) as img:
    width, height = img.size
    image_size = max(width, height)
    print(image_size)
print(image_size)
'''
# Run batched inference on a list of images
results = model(images, imgsz=1280)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # Save to disk with the original filename plus '_result.jpg'

    path = Path(result.path)
    result.save(filename=output_path+path.stem+"_yolo11n.pt.jpg")  # save to disk

# This code snippet is from the Ultralytics YOLOv5 repository. The YOLOv5 model is loaded and used to perform object detection on a batch of images. The model is run on the list of images, and the results are processed to extract bounding boxes, segmentation masks, keypoints, classification probabilities, and oriented bounding boxes. The results are then displayed on the screen and saved to disk.

#TODO: capire in quale riga viene chiamato il predict, vedere se è possibile passare imgsz al predict e richiamarlo manualmente. O trovare alternative per passare image size.
#altrimenti dividere l'immagine grande in patches della dimensione in cui le riduce YOLO, con overlapping parziale.