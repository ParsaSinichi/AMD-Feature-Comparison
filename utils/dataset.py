import os
import yaml
from collections import defaultdict
import numpy as np
def calculate_class_weights(y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    class_weights = {}
    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (2.0 * class_count)
        class_weights[class_label] = class_weight
    return class_weights

def get_label_from_path(path, label_map):
    for label_name, label_value in label_map.items():
        if label_name in path:
            return label_value
    raise ValueError(f"Unknown label for path: {path}")

def dataset_part():
    with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    dataset_dict = defaultdict(list)
    for subset in os.listdir(config["dataset_path"]):
        subset_path = os.path.join(config["dataset_path"], subset)
        if os.path.isdir(subset_path):
            # For each class in the subset
            for class_name in os.listdir(subset_path):
                class_path = os.path.join(subset_path, class_name)
                if os.path.isdir(class_path):
                    for file in os.listdir(class_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, file)
                            dataset_dict[subset].append(img_path)
    return  dataset_dict
