# Comparison of Classical and Deep Learning-Based Feature Representations for Age-Related Macular Degeneration

### Parsa Sinichi, Miguel O. Bernabeu, Malihe Javidi

This repository contains the code for the paper:

**Comparison of Classical and Deep Learning-Based Feature Representations for Age-Related Macular Degeneration [MICAD2024]**  
[DOI: 10.1007/978-981-96-3863-5_46](https://doi.org/10.1007/978-981-96-3863-5_46)

## Abstract

Age-related Macular Degeneration (AMD) is a leading cause of visual impairment among the elderly worldwide. This study compares deep learning-based and classical feature extraction methods for AMD classification using colour fundus images, a cost-effective imaging modality. This comparative approach aims to evaluate the performance of advanced deep learning methods against traditional computer vision approaches in generating effective feature representation. To achieve this, we compared various classical feature extraction methods alongside deep learning-based approaches, specifically VGG16 pre-trained on ImageNet, RETFound and RETFound-Green, which are foundation models pre-trained on a significant number of retinal images. The results demonstrate that combinations of classical features still obtained notable performance and even outperformed the VGG16 model pre-trained on ImageNet. These results also highlight the need for deep learning models to be trained on domain-specific data to surpass classical approaches. In this regard, the pre-trained RETFound and RETFound-Green models yielded promising outcomes for AMD classification, and the highest performance was achieved using a fine-tuned RETFound model. Furthermore, the best proposed method was compared with the state-of-the-art methods. This approach achieved competitive results on the ADAM dataset, demonstrating its robustness and effectiveness in AMD classification while utilising a significantly simpler architecture.

---

<img src="figures/Fig1 (1).png" alt="Duration Predictor" width="50%" style="width:50%">

---

## Feature Extraction and Classification

### Folder Structure

To run the code, ensure your folder structure resembles the following:

```
Project/
├── Dataset/
│   ├── Train/
│   │   ├── AMD
│   │   └── Non-AMD
│   ├── Test/
│   │   ├── AMD
│   │   └── Non-AMD
│   └── config/
│       └── config.yaml
```

### Downloading Weights

To use RETFound and RETFound-Green, you need to download their weights and specify their paths in the `config.yaml` file.

[Download RETFound](https://github.com/rmaphoh/RETFound_MAE)

[Downlaod RETFound-green](https://github.com/justinengelmann/RETFound_Green)

### Running the Code

Execute the main script `main.py` with the feature extractor(s) of your choice. Use the `--features` argument followed by one or more feature extractor names.

Examples:

```console
main.py --featuers lbp dwt
```

```console
main.py --featuers rf rfg
```

---
