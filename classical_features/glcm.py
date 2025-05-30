import SimpleITK as sitk
from radiomics import featureextractor
from PIL import Image
import numpy as np
import cv2
import logging

# Suppress radiomics warnings
logging.getLogger("radiomics").setLevel(logging.ERROR)

def extract_glcm_features(image_path):
    image = Image.open(image_path).resize((256, 256), Image.ANTIALIAS)
    image_np = np.array(image)
    threshold_value = 5

    # Initialize extractor for GLCM only
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glcm')

    all_glcm_features = []

    for channel in range(3):  # R, G, B channels
        channel_data = image_np[:, :, channel]

        _, binary_mask = cv2.threshold(channel_data, threshold_value, 255, cv2.THRESH_BINARY)
        binary_mask = (binary_mask // 255).astype(np.uint8)

        image_sitk = sitk.GetImageFromArray(channel_data)
        mask_sitk = sitk.GetImageFromArray(binary_mask)

        features = extractor.execute(image_sitk, mask_sitk)
        glcm_features = [v for k, v in features.items() if "glcm" in k]
        all_glcm_features.extend(glcm_features)

    return np.array(all_glcm_features)

