import cv2
from ultralytics import YOLO

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results_usual_yolo_inference,
    visualize_results,
)

# Load the image
img_path = 'inputs\\test\\super_res_image.jpg'
img = cv2.imread(img_path)

# Load the YOLO model
element_crops = MakeCropsDetectThem(
    image=img,
    model_path="yolo11m.pt",
    segment=False,
    show_crops=False,
    shape_x=600,
    shape_y=500,
    overlap_x=50,
    overlap_y=50,
    conf=0.5,
    iou=0.7,
    classes_list=[0, 1, 2, 3, 5, 7],
)
result = CombineDetections(element_crops, nms_threshold=0.05) # Combine the detections
#save the results

print('Basic yolo inference:')
visualize_results_usual_yolo_inference(
    img,
    model=YOLO("yolo11m.pt") ,
    imgsz=640,
    conf=0.5,
    iou=0.7,
    segment=False,
    thickness=2,
    show_boxes=True,
    delta_colors=3,
    show_class=True,
    show_confidences=True,
)

print('YOLO-Patch-Based-Inference:')
visualize_results(
    img=result.image,
    confidences=result.filtered_confidences,
    boxes=result.filtered_boxes,
    classes_ids=result.filtered_classes_id,
    classes_names=result.filtered_classes_names,
    segment=False,
    thickness=2,
    show_boxes=True,
    delta_colors=3,
    show_class=True,
    show_confidences=True,
)


#output_path = "results\\"  # output images directory
# Save the results
#cv2.imwrite(output_path + 'super_res_image_yolo11m_pt.jpg', result.image)