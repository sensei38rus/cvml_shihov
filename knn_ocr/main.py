import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread


data_directory = Path("./task")
training_data_path = data_directory / "train"


class_mapping = []

def extract_features(img):
 
    if img.ndim == 2:
        grayscale = img
        binary_mask = grayscale > 0
    else:
        grayscale = np.mean(img, axis=2).astype(np.uint8)
        binary_mask = grayscale > 0
    
 
    labeled = label(binary_mask)
    regions = regionprops(labeled)
    
  
    filtered_regions = [r for r in regions if r.extent <= 0.81]
    
 
    if filtered_regions:
        region = filtered_regions[0]
        features = [
            region.eccentricity,
            region.solidity, 
            region.extent,
            region.perimeter / region.area,
            region.area_convex / region.area
        ]
    else:
      
        features = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    return np.array(features, dtype=np.float32)

def prepare_training_data(data_folder):

    features_list = []
    labels_list = []
    current_class = -1
    
  
    for class_folder in sorted(data_folder.glob("*")):
        current_class += 1
       
        class_mapping.append(str(class_folder)[-1])
        
    
        for image_file in sorted(class_folder.glob("*.png")):
            img_data = imread(image_file)
            feature_vector = extract_features(img_data)
            features_list.append(feature_vector)
            labels_list.append(current_class)
    
  
    features_array = np.array(features_list, dtype=np.float32).reshape(-1, 5)
    labels_array = np.array(labels_list, dtype=np.float32).reshape(-1, 1)
    
    return features_array, labels_array


for image_index in range(7):

    test_image = imread(data_directory / f"{image_index}.png")
    
  
    training_features, training_labels = prepare_training_data(training_data_path)
    
   
    knn_classifier = cv2.ml.KNearest.create()
    knn_classifier.train(training_features, cv2.ml.ROW_SAMPLE, training_labels)
    
   
    grayscale_img = np.mean(test_image, axis=2).astype(np.uint8)
    binary_img = grayscale_img > 0
    labeled_regions = label(binary_img.T)
    region_properties = regionprops(labeled_regions)
    
  
    test_features = []
    for idx, region in enumerate(region_properties):
        if region.extent < 0.7:
            region_features = extract_features(region.image)
            test_features.append(region_features)
    
    
    test_features = np.array(test_features, dtype=np.float32).reshape(-1, 5)
    
    
    if len(test_features) > 0:
        retval, predictions, neighbors, distances = knn_classifier.findNearest(test_features, k=3)
        
       
        for prediction in predictions:
            predicted_class = int(prediction.item())
            if predicted_class < len(class_mapping):
                print(class_mapping[predicted_class], end="")
    
    print("\n")
    
    class_mapping.clear()