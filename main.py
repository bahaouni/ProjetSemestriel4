import numpy as np
import cv2 
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, skeletonize
from math import pi

similarité = 0.001  
tree_height = 3 

def comparer_images(image1, image2):
    # Calculate the Euclidean distance between the images
    distance = np.linalg.norm(image1 - image2)
    
    # Determine if the images are redundant based on a threshold
    if distance < similarité:
        return True
    else:
        return False 

def gerer_redondance_images(images):
    # Resize images to the same dimensions
    images_numeriques = [cv2.resize(image, (2240, 1680)) for image in images]
    
    # Initialize a list to store redundant images
    images_redondantes = []
    
    # Iterate through each image
    for i, image in enumerate(images_numeriques):
        # Compare the current image with all previous images
        for j in range(i):
            image_precedente = images_numeriques[j]
            if comparer_images(image, image_precedente):
                images_redondantes.append(i)
                break
    
    # Remove redundant images from the original list
    images_sans_redondance = [image for i, image in enumerate(images) if i not in images_redondantes]
    
    return images_sans_redondance

def estimate_volume(images):
    total_volume = 0
    for image in images:
        # Calculate the area covered by branches in the image
        area = image.shape[0] * image.shape[1]  # Assuming each pixel represents a unit area
        
        # Estimate the radius of the tree trunk (assuming it's circular)
        # This is a rough estimation and may need to be adjusted based on the actual tree shape
        trunk_radius = min(image.shape[0], image.shape[1]) / 20  # Adjust the divisor as needed
        
        # Calculate the volume of a cylinder with the area of the image as the base
        volume = pi * (trunk_radius ** 2) * tree_height
        
        # Add the volume of this cylinder to the total volume
        total_volume += volume
    
    return total_volume

def estimate_small_branches(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Initialize morphological closing kernel
    kernel = np.ones((5, 5), np.uint8)

    # Initialize thresholding parameters
    min_branch_area = 200
    max_branch_area = 600
    threshold_increment = 5

    # Apply adaptive thresholding to binarize the image
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological closing
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Thinning (skeletonization) using skimage
    skeleton = skeletonize(closed_image.astype(bool))

    # Count connected components in the skeleton (represents branches)
    num_branches, _ = cv2.connectedComponents(skeleton.astype(np.uint8))

    # Calculate total area of branches
    total_branch_area = np.sum(skeleton)

    # Iteratively adjust thresholding until the desired range is achieved or max iterations reached
    iterations = 0
    while (total_branch_area < min_branch_area or total_branch_area > max_branch_area) and iterations < 10:
        # Update thresholding parameters
        if total_branch_area < min_branch_area:
            min_branch_area -= threshold_increment
        elif total_branch_area > max_branch_area:
            max_branch_area += threshold_increment

        # Apply adaptive thresholding with updated parameters
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Apply morphological closing
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        
        # Thinning (skeletonization) using skimage
        skeleton = skeletonize(closed_image.astype(bool))

        # Count connected components in the skeleton (represents branches)
        num_branches, _ = cv2.connectedComponents(skeleton.astype(np.uint8))

        # Calculate total area of branches
        total_branch_area = np.sum(skeleton)

        iterations += 1

    return num_branches

def main():
    image_paths = ["s1.jpeg","s5.jpeg","s2.jpeg","s3.jpeg","s4.jpeg","s2.jpeg"]
    images = [cv2.imread(path) for path in image_paths]

    images_without_redundancy = gerer_redondance_images(images.copy())

    # Estimate volume
    estimated_volume = estimate_volume(images_without_redundancy)
    print("Estimated volume of the tree:", estimated_volume/10000, "Cubic centimeter")

    # Estimate number of small branches (consider each image)
    total_small_branches = 0

    for image in images_without_redundancy:
        num_branches = estimate_small_branches(image)
        total_small_branches += num_branches

    print("Estimated number of small branches:", total_small_branches/10)
    #Dans un arbre à branches, on estime en moyenne à 18 le nombre de drupes par branche
    print("Estimated number of drupes in one branche :18 ")
    print("Estimated number of all drupes  ",(total_small_branches/10)*18 )
    print("Estimated number of kgs   ",((total_small_branches/10)*30)/180 )

if __name__ == "__main__":
    main()
