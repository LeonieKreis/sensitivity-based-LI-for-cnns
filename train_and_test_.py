import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, traindataloader,testdataloader, optimizer, no_epochs, scheduler, stopping_criterion=None, save_grad_norms=False):
    # TODO stopping criterion
    times = []
    losses = []
    test_accs = []
    grad_norms = []
    grad_norms_layerwise = []
    for p in model.parameters():
        grad_norms_layerwise.append([])

    for e in range(no_epochs):
        toc = time.time()
        print('\nEpoch: %d' % e)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(traindataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss(outputs, targets)
            loss.backward()
            if save_grad_norms:
                norm = 0
                layer = 0
                lr = optimizer.param_groups[0]['lr']
                for p in model.parameters():
                    grad_norms_layerwise[layer].append(
                        lr*torch.square(p.grad).sum().numpy())
                    layer += 1
                for p in model.parameters():
                    norm += torch.square(p.grad).sum()
                grad_norms.append(norm)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        losses.append(train_loss)
        acc = 100.*correct/total
        print(f'training accuracy: {acc} and loss {train_loss}') 
        test_acc = test(model, testdataloader)
        test_accs.append(test_acc)
        print(f'test accuracy: {test_acc}')
        tic=time.time()
        times.append(tic-toc)
        scheduler.step()  


    return losses, test_accs, times, grad_norms_layerwise


def test(model,testdataloader):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testdataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
    return acc