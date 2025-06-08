from classical_features.dwt import extract_dwt_features
# from classical_features.glcm import extract_glcm_features
from classical_features.lbp import extract_lbp_features
from deep_featuers.retfound_feature_extraction import rf_extract_features
from deep_featuers.retfound_green_feature_extraction import rfg_extract_features
import argparse
import yaml
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from util.dataset import dataset_part, get_label_from_path, calculate_class_weights
from util.metrics import evaluate_model, plot_roc, plot_confusion_matrix
FEATURE_EXTRACTORS = {
    'dwt': extract_dwt_features,
    # 'glcm': extract_glcm_features,
    'lbp': extract_lbp_features,
    'rf': rf_extract_features,
    'rfg': rfg_extract_features,
}


def prepare_data(feat_dict, scaler_type='minmax'):
    X_train = np.array([sample['feat'] for sample in feat_dict["Train"]])
    y_train = np.array([sample['label'] for sample in feat_dict["Train"]])
    X_test = np.array([sample['feat'] for sample in feat_dict["Test"]])
    y_test = np.array([sample['label'] for sample in feat_dict["Test"]])

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def extract_features(image_paths, feat_list, label_map):
    features_dict = {}
    for split, paths in image_paths.items():
        split_features = []
        for image_path in tqdm(paths, desc=f"Extracting features for {split}"):
            combined_features = []
            for feat in feat_list:
                combined_features.extend(FEATURE_EXTRACTORS[feat](image_path))
            label = get_label_from_path(image_path, label_map)
            split_features.append({'feat': combined_features, 'label': label})
        features_dict[split] = split_features
    return features_dict


with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
image_paths = dataset_part()
parser = argparse.ArgumentParser(description="LBP feature extractor")
parser.add_argument('--features_dict', nargs='+', default=['dwt', 'lbp'])
args = parser.parse_args()
feat_dict = extract_features(
    image_paths, args.features_dict, config['label_map'])
X_train, y_train, X_test, y_test = prepare_data(
    feat_dict, scaler_type='standard')

class_weights = calculate_class_weights(y_train)
log_reg = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs',
                             max_iter=10000, class_weight=class_weights, random_state=82)

# Fit the model on the training data
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = log_reg.score(X_test, y_test)
print("Accuracy:", accuracy)

y_prob = log_reg.predict_proba(X_test)[:, 1]

results = evaluate_model(y_test, y_pred, y_prob)
for metric, value in results.items():
    print(f"{metric.capitalize()}: {value:.4f}")
plot_roc(y_test, y_prob, save_path='roc_curve.png')
plot_confusion_matrix(y_test, y_pred, [
                      'Non-AMD', 'AMD'], normalize=True, save_path='confusion_matrix.png')
