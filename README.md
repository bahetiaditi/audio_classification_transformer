# Transformer Networks for Audio Classification

This project demonstrates the implementation of transformer networks to classify environmental audio recordings into predefined categories. We explore two main architectures: one based solely on convolutional neural networks (CNNs) and another that combines CNNs with transformer encoders for feature extraction and classification.

## Project Structure

audio_classification_transformer/
│
├── src/
│ ├── init.py
│ ├── model.py # Contains model definitions
│ ├── dataset.py # Custom dataset and dataloader definitions
│ ├── train.py # Script for running training experiments
│ ├── experiments.py # Definitions of experiment functions
│ └── utils.py # Utility functions for training and evaluation
│
├── requirements.txt # Project dependencies
│
└── README.md # Project overview and setup instructions


## Experiments

The project includes various experiments to analyze the performance of different network architectures and configurations:

- **Architecture 1**: Utilizes CNNs for feature extraction and classification.
- **Architecture 2**: Combines CNNs with transformer encoders for enhanced feature extraction and classification.

Each architecture is evaluated under different configurations, including the application of dropout, early stopping, regularization, and variations in the number of attention heads in the transformer model.

Results of these experiments, including accuracy, loss per epoch, confusion matrices, F1 scores, and ROC-AUC curves, are logged and visualized using the Weights & Biases platform.

