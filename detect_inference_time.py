from ultralytics import YOLO
import os
from pathlib import Path
import time

#library to get image size
from PIL import Image

start_time = time.time()
# Load a model
pretrained_model_path = "yolo11m.pt"
finetuned_model_path = "experiments/yolo_finetune_epoch99.pt"
model = YOLO(finetuned_model_path)  # pretrained YOLO11n model


original_images_path = "datasets\\FLIR\\val (original)\\"  # input images directory
SR_images_path = "datasets\\FLIR\\val (SR finetune)\\"  # input images directory
input_path = original_images_path
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
#split images list in lists of 10 images
images_list = [images[i:i + 10] for i in range(0, len(images), 10)]
# Run batched inference on a list of images for each list in images
for images in images_list:
    results = model(images, imgsz=640, stream=True)  # return a list of Results objects
    # Nota: stream=True è necessario per evitare di caricare tutte le immagini in memoria
    # ritorna un generatore di oggetti Results

    # Process results list
    for result in results:
        '''boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs'''
        # result.show()  # display to screen
        # Save to disk with the original filename plus '_result.jpg'

        #path = Path(result.path)
        #result.save(filename=output_path+path.stem+"_yolo11n.pt.jpg")  # save to disk
    break

end_time = time.time()
print(f"Time elapsed: {end_time - start_time} seconds")
print("Average time per image: ", (end_time - start_time)/len(images))
print("Processed images: ", len(images))

# This code snippet is from the Ultralytics YOLOv5 repository. The YOLOv5 model is loaded and used to perform object detection on a batch of images. The model is run on the list of images, and the results are processed to extract bounding boxes, segmentation masks, keypoints, classification probabilities, and oriented bounding boxes. The results are then displayed on the screen and saved to disk.

#TODO: capire in quale riga viene chiamato il predict, vedere se è possibile passare imgsz al predict e richiamarlo manualmente. O trovare alternative per passare image size.
#altrimenti dividere l'immagine grande in patches della dimensione in cui le riduce YOLO, con overlapping parziale.