import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = SimpleNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    for images, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(train_data, batch_size=64, shuffle=False)

from os import access
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
      predictions = model(images)
      _, predicted_labels = torch.max(predictions, 1)
      total += labels.size(0)
      correct += (predicted_labels == labels).sum().item()


accuracy = 100* correct/ total
print(f'Accuracy: {accuracy}%')

from PIL import Image
import torchvision.transforms as transforms

image_path = 'test_data.png'
image = Image.open(image_path).convert('L')

transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(),
                                ])


image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)

model.eval()

with torch.no_grad():
    prediction = model(image_tensor)
    _, predicted_label = torch.max(prediction, 1)


print(f' The predicted digit is : {predicted_label.item()}')