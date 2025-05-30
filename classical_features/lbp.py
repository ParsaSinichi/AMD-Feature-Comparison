import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os

def extract_lbp_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    feat_list = []
    radius = 1
    n_points = 8 * radius
    for i in range(3):
        lbp_image = local_binary_pattern(image[:, :, i], n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype('float')
        lbp_hist /= lbp_hist.sum()
        feat_list.extend(lbp_hist)

    return feat_list
