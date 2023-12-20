import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, traindataloader, optimizer, no_epochs, scheduler, stopping_criterion):
    model.train()
    losses = []
    for e in range(no_epochs):
        print('\nEpoch: %d' % e)
        
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(traindataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        losses.append(train_loss)
        acc = 100.*correct/total
        print(f'training accuracy: {acc} and loss {train_loss}') 
        scheduler.step()  


    return losses


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