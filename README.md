# Tinyactions

Tiny Action Recognition
Low-Res Activity Detection and Recognition

[![GitHub license](https://img.shields.io/github/license/mrterrific007/Tinyactions)](https://github.com/mrterrific007/Tinyactions/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/mrterrific007/Tinyactions)](https://github.com/mrterrific007/Tinyactions/issues)
[![GitHub stars](https://img.shields.io/github/stars/mrterrific007/Tinyactions)](https://github.com/mrterrific007/Tinyactions/stargazers)

## Overview

Tiny Action Recognition is a deep learning project focused on recognizing and detecting subtle actions in low-resolution video footage. Leveraging benchmarks like Tiny-VIRAT-v2 and MAMA datasets, this repository explores innovative approaches to activity recognition and detection, including:

- Advanced model architectures (ResNet, Mamba)
- Robust hyperparameter tuning
- Custom evaluation metrics
- Effective data preprocessing and augmentation

The goal is to push the boundaries of what's possible with activity recognition, even when dealing with the challenges of poor video quality.

## Features

- Extensive experimentation with different model backbones (ResNet, Mamba) for optimal performance
- Systematic hyperparameter optimization (learning rate, batch size, epochs)
- Exploration of advanced activation functions (Leaky ReLU, ELU, SELU) for enhanced feature representation
- Incorporation of batch normalization for improved training stability and generalization
- Implementation of sophisticated data augmentation techniques (random cropping, rotation, color jittering)
- Leveraging transfer learning and fine-tuning of pre-trained models
- Custom loss function design to better capture task-specific nuances

## Getting Started

1. Clone the repository:

```
git clone https://github.com/mrterrific007/tiny-action-recognition.git
```

2. Install the required dependencies:

```
cd tiny-action-recognition
pip install -r requirements.txt
```

3. Prepare the dataset:

   - Download the Tiny-VIRAT-v2 or MAMA dataset
   - Update the dataset paths in the configuration files

4. Run the experiments:

```
python train.py --config config/resnet.yaml
```

5. Evaluate the model performance:

```
python evaluate.py --config config/resnet.yaml
```

## Contributing

Contributions to this project are welcome! If you have any ideas, bug fixes, or improvements, please feel free to submit a pull request. Before starting, please review the [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).
