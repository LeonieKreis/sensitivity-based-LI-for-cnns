import torch
import torchvision
import torchvision.transforms as transforms
import torchvision
import random
import os
import numpy as np
import copy

from layer_insertion import training_with_one_LI
from train_and_test_ import train
from nets import build_vgg_baseline, extend_VGG
from save_to_json import write_losses

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

# for checking the progress of the training in the terminal, use the bash command: jp length filename.json
# to see how many runs are already saved

k = 1

# declare path where json files are saved

path1 = f'results_data/Exp{k}_1.json'
if os.path.isfile(path1):
    print(f' file with path {path1} already exists!')
    quit()

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

torch.set_num_threads(8)


#################################################################################
epochs = [1,1]
epochs_class = sum(epochs)
BN=True
lr_init = 0.05
lr_class_small = lr_init
lr_class_big = lr_init


lr_args = {'step_size': 100,
           'gamma': 0.5}




model_class_small = build_vgg_baseline(BN)
init_vec1 = torch.nn.utils.parameters_to_vector(model_class_small.parameters())
init_vec2 = copy.deepcopy(init_vec1.data)
init_vec3 = copy.deepcopy(init_vec1.data)


max_length = epochs_class



################################################################################
# determine which trainings are run
T1 = True
T2 = True
T3 = True
T4 = True

# define no of training run instances

no_of_initializations = 30  # 50


for init in range(no_of_initializations):

    # absmax
    if T1:
        losses1, accs1, times1, grad_norms1 = training_with_one_LI(
            epochs=epochs, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'abs max', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec2)
        
        write_losses(path1,losses1,max_length,errors=accs1,interval_testerror=1,times=times1, grad_norms=grad_norms1)

    # absmin
    if T2:
        losses2, accs2, times2, grad_norms2 = training_with_one_LI(
            epochs=epochs, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'abs max', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec3)
        
        path2 = f'results_data/Exp{k}_2.json'
        write_losses(path2,losses2,max_length,interval_testerror=1,errors=accs2,times=times2, grad_norms=grad_norms2)

    # train classical small
    if T3:
        
        optimizer_small = torch.optim.SGD(model_class_small.parameters(),lr_class_small,
                                          momentum = 0.9, weight_decay=5e-4)
        scheduler_small = torch.optim.lr_scheduler.StepLR(
            optimizer_small, step_size=lr_args['step_size'], gamma=lr_args['gamma'])
        losses3, accs3, times3, grad_norms3 = train(model_class_small,trainloader,testloader,optimizer_small,epochs_class,scheduler_small,save_grad_norms=True)
        path3 = f'results_data/Exp{k}_3.json'
        write_losses(path3,losses3,max_length,errors=accs3,interval_testerror=1,times=times3, grad_norms=grad_norms3)




    # train classical big
    if T4:
        model_class_big = extend_VGG(position=0,BN=BN)
        optimizer_big = torch.optim.SGD(model_class_big.parameters(),lr_class_big,
                                          momentum = 0.9, weight_decay=5e-4)
        scheduler_big = torch.optim.lr_scheduler.StepLR(
            optimizer_big, step_size=lr_args['step_size'], gamma=lr_args['gamma'])
        losses4, accs4, times4, grad_norms4 = train(model_class_big,trainloader,testloader,optimizer_big,epochs_class,scheduler_big,save_grad_norms=True)
        path4 = f'results_data/Exp{k}_4.json'
        write_losses(path4,losses4,max_length,errors=accs4,interval_testerror=1,times=times4, grad_norms=grad_norms4)

