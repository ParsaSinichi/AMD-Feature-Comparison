import cv2
import numpy as np
import pyfeats
def extract_dwt_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    feat_list = []
    for i in range(3):
        feat_list.extend(pyfeats.dwt_features(image[:,:,i],None))
    return feat_list
