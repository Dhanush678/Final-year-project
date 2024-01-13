import os
import cv2
import numpy as np
import pywt

def read_images_with_labels(folder_path):
    label_images = {}
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        if os.path.isdir(label_path):
            images = []
            
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    images.append(img)
                else:
                    print(f"Unable to read image: {img_path}")
            
            label_images[label] = images
    
    return label_images


def dwt_feature_extraction(img_data, levels=1):
    coeffs = img_data.copy()
    for _ in range(levels):
        coeffs, _ = pywt.dwt2(coeffs, 'bior3.1')
    features = [np.abs(coeff)
                for coeff in pywt.wavedec2(coeffs, 'bior3.1', level=0)]
    return features


def dtcwt_feature_extraction(img_data, levels=1):
    coeffs = img_data.copy()
    for _ in range(levels):
        coeffs, _ = pywt.dwt2(coeffs, 'bior3.1')
    features = [np.abs(coeff)
                for coeff in pywt.wavedec2(coeffs, 'bior3.1', level=0)]
    return features


def process_images(images, output_folder_dwt, output_folder_dtcwt):
    # Create the output folders if they don't exist
    if not os.path.exists(output_folder_dwt):
        os.makedirs(output_folder_dwt)
    if not os.path.exists(output_folder_dtcwt):
        os.makedirs(output_folder_dtcwt)

    # Process each image
    for i, img in enumerate(images):
        # DWT feature extraction
        dwt_features = dwt_feature_extraction(img)
        for j, feature in enumerate(dwt_features):
            output_filename = os.path.join(
                output_folder_dwt, f"img_{i}_feature_{j}.png")
            cv2.imwrite(output_filename, feature)

        # DTWCT feature extraction
        dtcwt_features = dtcwt_feature_extraction(img)
        for j, feature in enumerate(dtcwt_features):
            output_filename = os.path.join(
                output_folder_dtcwt, f"img_{i}_feature_{j}.png")
            cv2.imwrite(output_filename, feature)


if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = r"D:\FYP_SOFTWARE 2023\FYP_SOFTWARE 2023\CODE AND DATASETS\PRE-PROCESSED OUTPUT_POST_SPM"
    
    # Read images with labels
    label_images = read_images_with_labels(input_folder)
    
    # Process each label's images
    for label, images in label_images.items():
        output_folder_dwt = f"D:\FYP_SOFTWARE 2023\FYP_SOFTWARE 2023\CODE AND DATASETS\first_level_DWT_{label}"
        output_folder_dtcwt = f"D:\FYP_SOFTWARE 2023\FYP_SOFTWARE 2023\CODE AND DATASETS\FIRSST_LEVEL_DTCWT_{label}"

        # Process the images for each label
        process_images(images, output_folder_dwt, output_folder_dtcwt)
