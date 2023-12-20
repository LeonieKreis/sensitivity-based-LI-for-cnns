VGG nets (feedforward convolutional)

each architecture exists with batch normalization after each convolutional layer and without in pytorch. 
The following nets are implemented:
- vgg11
- vgg13
- vgg16
- vgg19
  
In their publication from 2015, the following training is used on the dataset in paper (not cifar):

- minibatch gd with momentum 0.9
- batch size 256
- l2 reg with fsctor 0.0005
- dropout 0.5 for first 2 fc layers
- lr schedule, start at 0.01 decrease by 0.1 when val acc is stalling
- 74 epochs
- bengio init
  

  VGG16 on cifar: acc 89% ca
  - 40000 training 20000 test
  - lr 1e-3
  - lrscheduler lr(epoch)=lr_init * 0.5*epoch/20
  - sgd with momnetum 0.9
  - ce loss
  - 100 epochs
  - bs 128
  - data augmentation

VGG11 with bn on cifar: accuracy around 91percent
- 0.8 train 0.2 test
- batchsize 64
- lr_init 0.1
- reduce lr on plateua by factor 0.5
- sgd with nesterov momnetum 0.9, weight decay 0.0005
- 150 epochs

all vggs:
- data augmentation
- batchsize 400
- 0.75 train, 0.25 test
- 20 epochs
- max_lr 0.001
- lr scheduler onecyclelr from pytorch
- weight decay 1e-4
- grad clip 0.1
- adam


For the resnet architectures, the follwoing are implemneted:
- resnet18 93%
- resnet34
- resnet50 93,6%
- resnet101 93,75%
- resnet152

accuracies from https://github.com/kuangliu/pytorch-cifar (vgg16 92,64%)
  
They were traimed in their paper from 2015 with the follwoing hyperparameters for cifar-10: (there they had accuracies between 85 and 90%)

- minibatch sgd with momentum 0.9 and weight decay 1e-4
- bn, no dropout
- mb size 128
- lr_init=0.1
- decrease by factor 0.1 after 32K and 48K iterations
- terminate at 64K iterations (ca 165 epochs)
- data augmentation
- 50000 train 10000 test 

https://github.com/kuangliu/pytorch-cifar uses the hyperparameters:

- data augmentation
- bs 128
- sgd with momentum 0.9 and weight decay 5e-4
- cosine annealing lr, lr_init 0.1
- 200 epochs


