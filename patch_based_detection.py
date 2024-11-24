### Compattare codice per calcolare tempo di inferenza: PATCHED YOLO INFERENCE ON SUPER RESOLVED FLIR IMAGES
import cv2
import os
from ultralytics import YOLO
import time
import argparse

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results_usual_yolo_inference,
    visualize_results,
)

def patch_YOLO_detection(
        yolo_model: str = "experiments\\yolo_finetune_epoch99.pt",
        superres_path: str = "inputs\\FLIR_SR_val\\",
        combine_LR: bool = True,
        lowres_path: str = "inputs\\FLIR_GT_val\\",
        show_result_images: bool = False,
        output_path: str = "results\\",
        save_predictions: bool = False,
    ):
    '''
    Function to perform patched YOLO inference on super resolved images.
    Args:
        yolo_model (str): path to the yolo model to use for inference
        superres_path (str): path to the directory containing the super resolved images
        combine_LR (bool): whether to combine the detections of the super resolved image with the detections of low resolution image
        lowres_path (str): path to the directory containing the low resolution images
        show_results (bool): whether to show the results of the inference and save the images with the detections
    '''
    if save_predictions:
        import dill
    start_time = time.time()
    #yolo_model = "experiments\\yolo_finetune_epoch99.pt" #"yolo11m.pt"
    #superres_path = "inputs\\FLIR_SR_val\\"  # input images directory
    #ground_truth_path = "inputs\\FLIR_GT_val\\"
    superres_images_paths = [os.path.join(superres_path, img) for img in os.listdir(superres_path) if img.endswith((".jpg", ".jpeg", ".png"))] # list of image paths
    if combine_LR:
        ground_truth_images = [os.path.join(lowres_path, img) for img in os.listdir(lowres_path) if img.endswith((".jpg", ".jpeg", ".png"))] # list of image paths
    #output_path = "results\\"  # output images directory
    #del superres_images_paths[2:] # TODO: remove this line: it's just for testing
    patchyolo_SR_predictions_list = {}
    #yolo_SR_predictions_list = {}
    #yolo_SR_downsized_predictions_list = {}
    yolo_GT_predictions_list = {}
    print(f"Inferencing on {len(superres_images_paths)} images")
    iterations = min (len(superres_images_paths), len(ground_truth_images)) if combine_LR else len(superres_images_paths)
    elapsed_time = time.time() - start_time
    for i in range(iterations):
        start_iteration_time = time.time()
        superres_image_path = superres_images_paths[i]
        superres_image = cv2.imread(superres_image_path)
        SR_y, SR_x,SR_c = superres_image.shape
        print(f"Processing image {i+1}/{len(superres_images_paths)}")
        image_name = os.path.basename(superres_image_path).split('.')[0] # get the image name without the extension
        VALIDATION_IOU = 0.7
        VALIDATION_CONF = 0.01
        PREDICT_CONF = 0.25

        # yolo on SR image, PATCHES
        element_crops_patched = MakeCropsDetectThem(
            image=superres_image,
            model_path=yolo_model,
            segment=False,
            show_crops=False,
            imgsz=640,
            shape_x=640,
            shape_y=512,
            overlap_x=20,
            overlap_y=20,
            conf=0.4,#.5,
            iou=0.7,
            classes_list=[0, 1, 2, 3, 5, 7],
            )
        element_crops = []
        element_crops.append(element_crops_patched)

        #crops1 = element_crops_patched.crops
        if combine_LR:
            ground_truth_image_path = ground_truth_images[i]
            original_image = cv2.imread(ground_truth_image_path)
            GT_y, GT_x,GT_c = original_image.shape
            # original yolo on GT image
            original_image_element_crops = MakeCropsDetectThem(
                image=original_image,
                model_path=yolo_model,
                segment=False,
                show_crops=False,
                imgsz=640,
                shape_x=640,
                shape_y=512,
                overlap_x=0,
                overlap_y=0,
                conf=0.3,#.5,
                iou=0.7,
                classes_list=[0, 1, 2, 3, 5, 7],
                )
            # resize crops to match the super resolution
            original_image_element_crops.image = superres_image
            original_image_element_crops.shape_x = SR_x
            original_image_element_crops.shape_y = SR_y
            original_image_element_crops.imgsz = SR_x
            for i in range(len(original_image_element_crops.crops)):
                original_image_element_crops.crops[i].detected_xyxy = original_image_element_crops.crops[i].detected_xyxy * 2
                original_image_element_crops.crops[i].detected_xyxy_real = original_image_element_crops.crops[i].detected_xyxy_real * 2
                original_image_element_crops.crops[i].source_image = superres_image
                original_image_element_crops.crops[i].shape_x = SR_x
                original_image_element_crops.crops[i].shape_y = SR_y
                original_image_element_crops.crops[i].resize_results()

            element_crops.append(original_image_element_crops) #FIXME: DECOMMENTA. Il commento serve solo per testare il funzionamento di Patch YOLO SENZA unire le detections con quelle di YOLO su immagine originale

        #resize_initial_size = True # devo passare questo parametro a MakeCropsDetectThem per fare il resize dell'immagine iniziale

        patch_yolo_results = CombineDetections(
            element_crops,
            nms_threshold=0.65,
            match_metric="IOS",
            class_agnostic_nms=False,
            )

        #Saving the predictions in the dictionaries
        patchyolo_SR_predictions_list[image_name] = []
        for i in range(len(patch_yolo_results.filtered_boxes)):
            class_id = patch_yolo_results.filtered_classes_id[i]
            box = patch_yolo_results.filtered_boxes[i]
            confidence = patch_yolo_results.filtered_confidences[i]
            patchyolo_SR_predictions_list[image_name].append((class_id, box, confidence))

        elapsed_time += time.time() - start_iteration_time

        if show_result_images:
            print('Finetuned YOLO-Patch-Based-Inference (after super resolution):')
            img_SR_patch_yolo11m = visualize_results(
                img=patch_yolo_results.image,
                confidences=patch_yolo_results.filtered_confidences,
                boxes=patch_yolo_results.filtered_boxes,
                classes_ids=patch_yolo_results.filtered_classes_id,
                classes_names=patch_yolo_results.filtered_classes_names,
                return_image_array=True,
                segment=False,
                thickness=2,
                show_boxes=True,
                delta_colors=3,
                show_class=True,
                show_confidences=True,
                font_scale=0.5,
                show_classes_list=[0, 1, 2, 3, 5, 7],
            )
            cv2.imwrite(output_path + os.path.basename(ground_truth_image_path)[:-4] + '_SR_yolo11m_finetuned_pt_PATCH.jpg', img_SR_patch_yolo11m)
            if combine_LR:
                print('Finetuned yolo inference (on original image):')
                img_yolo11m = visualize_results_usual_yolo_inference(
                    original_image,
                    return_image_array=True,
                    model=YOLO(yolo_model),
                    imgsz=640,
                    conf=0.2,#.5, # confidence threshold
                    iou=0.7, # IoU threshold
                    segment=False,
                    thickness=1,
                    show_boxes=True,
                    delta_colors=3,
                    show_class=True,
                    show_confidences=True,
                    font_scale=0.25,
                    show_classes_list=[0, 1, 2, 3, 5, 7],
                )
                cv2.imwrite(output_path + os.path.basename(ground_truth_image_path)[:-4] + '_GT_yolo11m_finetuned_pt.jpg', img_yolo11m)

    if save_predictions:
        patchyolo_SR_predictions_list_file = open(output_path + "patchyolo_SR_predictions_list" + time.time() + ".pkl", 'wb')
        dill.dump(patchyolo_SR_predictions_list, patchyolo_SR_predictions_list_file)
        patchyolo_SR_predictions_list_file.close()

    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"Average time per image: {elapsed_time/len(superres_images_paths)} seconds")
    return patchyolo_SR_predictions_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model", type=str, default="yolo11m.pt", help="path to the yolo model to use for inference")
    parser.add_argument("--superres_path", type=str, default="inputs\\FLIR_SR_val\\", help="path to the directory containing the super resolved images")
    parser.add_argument("--combine_LR", type=bool, default=True, help="whether to combine the detections of the super resolved image with the detections of low resolution image")
    parser.add_argument("--lowres_path", type=str, default="inputs\\FLIR_GT_val\\", help="path to the directory containing the low resolution images")
    parser.add_argument("--show_result_images", type=bool, default=False, help="whether to show the results of the inference and save the images with the detections")
    parser.add_argument("--output_path", type=str, default="results\\", help="output images directory")
    parser.add_argument("--save_predictions", type=bool, default=False, help="whether to save the predictions in a file")
    args = parser.parse_args()
    patch_YOLO_detection(args.yolo_model, args.superres_path, args.combine_LR, args.lowres_path, args.show_result_images, args.output_path)