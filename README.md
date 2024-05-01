# Handwritten Digit and Character Classification with Neural Networks

## Project Description
This project explores the development and optimization of Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models for image classification tasks using the EMNIST (Extended MNIST) dataset. The goal is to build robust models capable of accurately classifying handwritten digits and characters.

## What It Does
- **Data Preprocessing:** The project begins with data preprocessing steps, including loading the EMNIST dataset, visualizing sample images, and preparing the data for model training.
- **Model Building:** Two types of neural network models are constructed: an MLP model and a CNN model. The MLP model comprises multiple fully connected layers, while the CNN model includes convolutional layers for feature extraction from images.
- **Model Training and Evaluation:** Both models are trained using the training data and evaluated using the validation set. The training process involves optimizing model parameters and hyperparameters to achieve high accuracy and generalization performance.
- **Hyperparameter Tuning:** The project explores various hyperparameters and techniques to optimize model performance, including activation functions, optimizers, regularization techniques, dropout rates, and batch normalization.
- **Result Analysis:** The performance of the optimized models is analyzed using evaluation metrics such as precision, recall, F1-score, and accuracy. Confusion matrices are also examined to identify misclassifications and areas for improvement.

## Key Features
- Implementation of MLP and CNN models for image classification.
- Data preprocessing and visualization techniques.
- Hyperparameter tuning and optimization for model performance.
- Evaluation of model performance using standard metrics and visualizations.

## How to Use
1. Clone the repository to your local machine.
2. Open the Jupyter notebook MLP_CNN_Project.ipynb in your preferred environment.
3. Follow the step-by-step instructions in the notebook to explore the project, run code cells, and analyze results.
4. Experiment with different hyperparameters or model architectures to further optimize performance.

## Requirements
- Python 3.x
- Jupyter Notebook or JupyterLab
- TensorFlow
- Keras
- Pandas
- Matplotlib
- NumPy
- Scikit-learn

## Results
The optimized CNN and MLP models achieved remarkable results in classifying handwritten digits and characters from the EMNIST dataset. Here are the key findings:

**Accuracy:** Both the CNN and MLP models achieved an accuracy of approximately 85% on the test dataset, showcasing their effectiveness in accurately classifying handwritten characters.

**Comparison:** While CNNs are typically favored for image classification tasks due to their ability to exploit spatial data, both CNN and MLP models exhibited excellent performance. The slightly better performance of the MLP model may be attributed to the exhaustive hyperparameter tuning process and the capacity of MLPs to model non-linear relationships.
