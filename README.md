# MNIST-Digit-Classification

## 📌 Overview
This project uses a neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
It demonstrates the end-to-end machine learning workflow: preprocessing, model building, training, evaluation, and visualization.

## ⚙️ Tech Stack
- **Python 3**
- **TensorFlow/Keras**
- **NumPy**
- **Matplotlib**

## 📂 Project Structure
├── mnist_classifier.py   # main script
├── requirements.txt      # dependencies
├── results/              # sample prediction images
└── README.md             # project documentation


## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-classification.git
   cd mnist-digit-classification

Key Features
1) Normalization of pixel values for faster convergence.
2) Dense neural network with ReLU and Softmax layers.
3) Training with Adam optimizer and cross-entropy loss.
4) Visualization of predictions with true vs. predicted labels.

Future Improvements
1) Implement Convolutional Neural Networks (CNNs) for higher accuracy.
2) Add confusion matrix and classification report.
3) Deploy model with Flask/Streamlit for interactive demo.

License
This project is licensed under the MIT License.
