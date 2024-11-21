import torch
import pytest
from mnist_model import LightMNIST
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

@pytest.fixture
def model():
    return LightMNIST()

@pytest.fixture
def train_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        transform=transform,
        download=True
    )
    
    return DataLoader(train_dataset, batch_size=64, shuffle=True)

def test_parameter_count(model):
    """Test that model has less than 25000 parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds limit of 25000"

def test_model_output(model):
    """Test model output shape"""
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"

def test_training_accuracy(model, train_loader):
    """Test model achieves 95% accuracy in one epoch"""
    device = torch.device('cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= 938:
            break
            
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 100 == 0:
            accuracy = 100. * correct / total
            print(f'Batch [{i + 1}/938], Loss: {running_loss/100:.4f}, Accuracy: {accuracy:.2f}%')
            running_loss = 0.0
    
    accuracy = 100. * correct / total
    print(f"\nFinal test accuracy: {accuracy:.2f}%")
    assert accuracy >= 95.0, f"Model achieved {accuracy:.2f}% accuracy, below required 95%"