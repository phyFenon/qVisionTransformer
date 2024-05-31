import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit import VisionTransformer
import matplotlib.pyplot as plt
import time

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1]
])
# Load CIFAR-10 train and test datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)#train= True, 加载训练模型数据
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)#train= False, 加载测试或验证模型数据)

# Instantiate ViT model
d_model = 128
num_heads = 8
mlp_hidden = 256 # 2*d_model
num_layers = 8
patch_size = 4
image_size = 32
num_classes = 10
dropout_rate = 0.2
batch_size = 64
####################

# Create DataLoader for CIFAR-10 train and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Instantiate ViT model
vit_model = VisionTransformer(d_model, num_heads, mlp_hidden, num_layers, patch_size, image_size, num_classes, dropout_rate)
# Load the trained model parameters
vit_model.load_state_dict(torch.load('vit_model.pth'))
# Define loss function
criterion = nn.CrossEntropyLoss()



# Create figure and axis outside the loop
plt.ion()
fig, ax1 = plt.subplots(figsize=(8, 6))
color1 = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Accuracy (%)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
plt.grid(True)
plt.title('Testing Loss and Accuracy vs. Epoch')

# Training loop
num_epochs = 100  # Increase the number of epochs for demonstration
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):  # Assuming the same number of epochs for testing
    t_start = time.time()
    vit_model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Ensure no gradient calculation during testing
        for i, data in enumerate(test_loader):
            inputs, labels = data

            outputs = vit_model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    t_end = time.time()
    print('[Epoch %d]/[Epoch %d] Average testing loss: %.5f, Testing Accuracy: %.5f %% , Time used: %.2f s' % (epoch + 1,num_epochs, test_loss, test_accuracy, t_end - t_start))

    # Update the dynamic plot for testing results (similar to training plot)
    ax1.plot(range(1, epoch + 2), test_losses, 'o-', label='Test_Loss', color=color1)
    ax2.plot(range(1, epoch + 2), test_accuracies, '*-', label='Test_Accuracy', color=color2)
    plt.pause(0.1)
    plt.ioff()
    
plt.savefig('test.png')
print('Finished Testing')
