# fully connected two layer NN (sigmoid,Softmax)activation
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim

#get the data
train_data = torchvision.datasets.FashionMNIST('../data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

valid_data = torchvision.datasets.FashionMNIST('../data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)

train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]
#In [19]:
mask = np.ones(60000)
mask[train_idx] = 0
#In [20]:
valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]
#In [21]:
batch_size = 1000
test_batch_size = 1000
learning_rate = 0.0001

num_epochs = 5

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),batch_size=test_batch_size, shuffle=True)

#In [22]:
print(train_loader.dataset.train_data.shape)
print(test_loader.dataset.test_data.shape)
print(train_data.train_labels[0])
#plt.imshow(train_loader.dataset.train_data[1].numpy(),cmap='gray')
plt.imshow(train_loader.dataset.train_data[1].reshape(28,28),cmap='gray')
#plt.imshow(train_loader.dataset.train_data[10].numpy(),cmap='gray')
#labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
#print(labels_map[train_data[img_xy][1]])
fig = plt.figure(figsize=(8,8))
columns = 4
rows = 5
'''for i in range(1, columns*rows +1):
    img_xy = np.random.randint(len(train_data));
    img = train_data[img_xy][0][0,:,:]
    fig.add_subplot(rows, columns, i)
#    plt.title(labels_map[train_loader.dataset.train_data[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()'''
class FcNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
#instance of the Conv Net
cnn = FcNetwork()
#loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
losses = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        '''if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))'''

print("average loss=",np.sum(losses)/num_epochs/(i + 1))
cnn.eval()
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))
