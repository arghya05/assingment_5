# Lightweight MNIST Classification Model

A highly efficient CNN implementation for MNIST digit classification that achieves:
- Over 95% accuracy in just one epoch
- Less than 25,000 parameters
- Consistent performance between local and CI environments

## Model Architecture

The model uses a carefully designed architecture to maximize performance while minimizing parameters: 


Model Architecture
The model uses a carefully designed architecture to maximize performance while minimizing parameters:	


LightMNIST(
  (features): Sequential(
    # First Convolutional Block
    (0): Conv2d(1, 16, kernel_size=3, padding=1)    # 28x28x16
    (1): BatchNorm2d(16)
    (2): ReLU()
    (3): MaxPool2d(2)                               # 14x14x16
    
    # Second Convolutional Block
    (4): Conv2d(16, 32, kernel_size=3, padding=1)   # 14x14x32
    (5): BatchNorm2d(32)
    (6): ReLU()
    (7): MaxPool2d(2)                               # 7x7x32
    (8): Dropout(0.2)
  )
  
  # Classifier
  (classifier): Sequential(
    (0): Linear(7 * 7 * 32, 10)                     # Direct mapping to output
  )
)

Parameter Count Breakdown
First Conv Layer: 16 (3 3 1 + 1) = 144 + 16 = 160 parameters
First BatchNorm: 16 2 = 32 parameters
Second Conv Layer: 32 (3 3 16 + 1) = 4,608 + 32 = 4,640 parameters
Second BatchNorm: 32 2 = 64 parameters
Classifier: (7 7 32) 10 + 10 = 15,680 + 10 = 15,690 parameters
Total Parameters: 20,586 (well under 25,000 limit)	


Key Features
Efficient Architecture
Two convolutional layers for feature extraction
MaxPooling for dimension reduction
Direct mapping to output without hidden layers
BatchNormalization for training stability
Strategic dropout for regularization
Training Optimizations
Learning rate: 0.002
Weight decay: 1e-5
Batch size: 64
Adam optimizer	

Deterministic behavior for reproducibility
3. Performance
Achieves >95% accuracy in one epoch
Consistent results between local and CI environments
Fast training time
Installation
Clone the repository:


Installation
1. Clone the repository:
git clone https://github.com/yourusername/mnist-lightweight.git
cd mnist-lightweight

Results
Sample training output:


2. Install dependencies:
pip install -r requirements.txt	

Requirements
Usage
Training the Model
from mnist_model import LightMNIST, train_model

# Initialize model
model = LightMNIST()

# Train model
train_model(model, train_loader, criterion, optimizer, device)

# Running Tests
python -m pytest test_model.py -v

# Results
Sample training output:
Epoch 1/1
------------------------------------------------------------
Batch [100/938], Loss: 0.4379, Accuracy: 86.83%
Batch [200/938], Loss: 0.1536, Accuracy: 91.12%
Batch [300/938], Loss: 0.1330, Accuracy: 92.69%
Batch [400/938], Loss: 0.1099, Accuracy: 93.65%
Batch [500/938], Loss: 0.0978,#  Accuracy: 94.37%
Batch [600/938], Loss: 0.0867, Accuracy: 94.86%
Batch [700/938], Loss: 0.0871, Accuracy: 95.22%
Batch [800/938], Loss: 0.0713,racy: 95.85%

# Project Structure
mnist-lightweight/
├── mnist_model.py          # Model implementation
├── test_model.py          # Test cases
├── requirements.txt       # Dependencies
└── .github/
    └── workflows/
        └── model_tests.yml # CI configuration
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments
PyTorch team for the deep learning framework
MNIST dataset creators
GitHub Actions for CI/CD capabilities
You can now copy this entire content and paste it into your README.md file. The markdown formatting will render properly on GitHub.











