
# import multiprocessing
# import roboflow
# from roboflow import Roboflow
# from pathlib import Path  # For path manipulation (optional)
# import torch  # Assuming you have PyTorch and torchvision installed
# import torch.utils.data as data
# from ultralytics import YOLO
# rf = Roboflow(api_key="xWjdxcfbbn9Dg2E4hdiI")
# project = rf.workspace("zaibi-rnd").project("olives-detection")
# version = project.version(1)
# dataset  = version.download("yolov8")
# # Adjust hyperparameters based on your dataset and hardware (epochs, batch size)
import cv2
import re
from io import StringIO
import sys
from ultralytics import YOLO

# Redirect standard output to a StringIO object


# Perform YOLO inference
model = YOLO("best.pt")
results = model("zitoun3.jpg", show=True, save=True)


    # # Consider using a custom DataLoader or Roboflow's YOLOv8 export options if the provided example doesn't match your data structure
# train_dataset = "./Olives-Detection-1/data.yaml"  # Implement your custom DataLoader or use Roboflow's export

# # Train the YOLOv8 model (initialize with a pretrained model for faster convergence)
# model = YOLO("yolov8x.pt")  # Use a pretrained YOLOv8 model (e.g., yolov8x.pt)
# data_path = Path(r"C:\Users\keasar\Desktop\yolo\Olives-Detection-1\train\images")
# res = model.train(data=data_path, epochs=100, imgsz=640)
# print(res)
# # Save the trained model
# model.save("trained_yolov8_olives.pt")
# if __name__=='__main__':
#     multiprocessing.freeze_support()
#     model = YOLO("trained_yolov8_olives.pt") # load a pretrained model 

#     results = model('./zitoun.jpg', show= True,save= True )

# import ultralytics
# import multiprocessing
# import cv2
# import numpy as np
# def estimate_tree_volume(image):
#     # This is a simple placeholder function, you need to implement a more
#     # accurate method for estimating the volume of the olive tree
#     return 1000
# if __name__ == "__main__":
#     multiprocessing.freeze_support()

#     # Load a pretrained YOLOv8 model
#     model = ultralytics.YOLO("yolov8x.pt")

#     # Set the input image
#     image_path = "zitounv2.jpg"

#     # Load the image
#     image = cv2.imread(image_path)

#     # Get the image dimensions
#     height, width, _ = image.shape

#     # Run YOLOv8 on the image
#     results = model(image, show=True, save=True)
#     print(results)
#     # Get the number of detected olives
#     num_olives = len(results.boxes[0]) 

#     # Estimate the volume of the olive tree
#     # This is a placeholder, you need to implement this function
#     volume = estimate_tree_volume(image)

#     # Estimate the olive fruit production
#     # This is a simple formula, you can improve it
#     production = num_olives * volume / 1000

#     print("Estimated olive fruit production:", production)

# This function is a placeholder, you need to implement it


# from ultralytics import YOLO
# import os

# def count_olives(image_path):
#     model = YOLO("yolov5s.pt")  # Load a pretrained model
#     results = model(image_path, show=True, save=True)

#     # Assuming that the saved results are in the current directory
#     result_image_path = os.path.join("runs", "detect", "predict", os.path.basename(image_path))
#     result_file_path = os.path.join("runs", "detect", "predict", "labels", os.path.splitext(os.path.basename(image_path))[0] + ".txt")

#     # Count the number of olives detected
#     if os.path.exists(result_file_path):
#         with open(result_file_path, 'r') as f:
#             lines = f.readlines()
#             num_olives = len(lines)
#     else:
#         num_olives = 0

#     return num_olives

# if __name__ == '__main__':
#     image_path = "zitoun.jpg"
#     num_olives = count_olives(image_path)
#     print("Number of olives detected:", num_olives)
