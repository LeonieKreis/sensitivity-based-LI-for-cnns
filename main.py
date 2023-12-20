import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import torchvision

from nets import VGG,VGG_BN, VGG_BN_fullext, VGG_fullext, extend_VGG

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#model = VGG_fullext()
model = extend_VGG(BN=True, position=0)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

# initialize parameters # TODO

lr_init=0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_init,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

model.train()
for batch, (X,y) in enumerate(trainloader):
    X,y = X.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred,y)
    loss.backward()
    optimizer.step()

print(loss.item())

for p in model.parameters():
    print(p.shape)