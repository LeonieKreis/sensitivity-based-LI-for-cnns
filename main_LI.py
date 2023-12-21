import torch
import torchvision
import torchvision.transforms as transforms
import torchvision

from layer_insertion import training_with_one_LI

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


#################################################################################
epochs = [2,2]
BN=False
lr_init = 0.05

lr_args = {'step_size': 100,
           'gamma': 0.5}






################################################################################

losses, accs, times, grad_norms = training_with_one_LI(
    epochs=epochs, traindataloader=trainloader, testdataloader=testloader,BN=BN,
    optimizer_type='SGD', lr_init=lr_init, mode = 'abs max', stopping_criterion=None,
    lrschedule_type='StepLR', lrscheduler_args=lr_args, 
    decrease_lr_after_li=1.,save_grad_norms=True)


