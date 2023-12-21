import torch
from nets import build_vgg_fullyext, extend_VGG

def _is_freezed(p, freezed):
    '''
    checks if the parameter p is in the list of frozen parameters 'freezed'
    Args:
        p: a model parameter (p in model.parameters())
        freezed (list): list of the frozen parameters of the model
    Out:
         boolean
    '''
    for freezed_p in freezed:
        # print(p.shape, freezed_p.shape)
        try:
            if p is freezed_p:
                return True

        except RuntimeError as m:
            print(m)
            continue
    return False


def _number_of_params(p):
    '''
    computes the dimension of a parameter (vectorized for matrices)

    Args:
        p: model parameter from which we want to know the dimension

    Out:
        integer, which gives the dimension of the parameter'''
    res = 1
    for s in p.shape:
        res *= s
    return res

def _number_of_frozen_parameters_collected(model, freezed):
    '''
     counts total number/dimension of frozen parameters given in 'freezed' in the model
    Args:
        model: model from pytorch
        freezed (list): list of the frozen parameters of the model
    Out:
         dim (int): total number of frozen parameters
    '''
    no = 0
    with torch.no_grad():
        for p in model.parameters():
            if _is_freezed(p, freezed):
                no += 1
    return no

def init_fullyext_net(model_baseline, model_fullyextended,freezed, BN=False):
    '''
    initializes the fully extended model such that it represents the function of the baseline model.
    '''
    with torch.no_grad():
        i=0
        t64 = torch.zeros([64,64,3,3])
        for i in range(64):
            t64[i,i,1,1]=1.
        t128 = torch.zeros([128,128,3,3])
        for i in range(128):
            t128[i,i,1,1]=1.
        t256 = torch.zeros([256,256,3,3])
        for i in range(256):
            t256[i,i,1,1]=1.
        if BN:
            new_weights_inits = [
            torch.ones([64]),
            torch.zeros([64]),
            t64,
            torch.zeros([64]),
            torch.ones([128]),
            torch.zeros([128]),
            t128,
            torch.zeros([128]),
            torch.ones([256]),
            torch.zeros([256]),
            t256,
            torch.zeros([256])
            ] # 64->1,64->0 64x64x3x3, 64->0, 128->1,128->0, 128x128x3x3, 128->0,256->1, 256->0 256x256x3x3, 256->0
        else:
            new_weights_inits = [
            t64,
            torch.zeros([64]),
            t128,
            torch.zeros([128]),
            t256,
            torch.zeros([256])
            ] # 64x64x3x3, 64->0, 128x128x3x3, 128->0, 256x256x3x3, 256->0



        old_param_iterator = model_baseline.parameters()
        for p_new in model_fullyextended.parameters():
            if not _is_freezed(p_new, freezed):
                p = next(old_param_iterator)
                p_new.copy_(p)
            else:
                p_new.copy_[new_weights_inits[i]]
                i+=1




    return model_fullyextended

def init_ext_net(model_baseline, model_ext, position, BN=False): 
    '''
    initializes the extended model (with one layer more) such that it represents the function of the baseline model.
    '''
    freezed = []
    if position==0:
        channels = 64
        if BN: pos=[2,3,4,5]
        else: pos=[2,3]
        for i,p in enumerate(model_ext.parameters()):
            if i in pos:
                freezed.append(p)
        
    if position==1:
        channels=128
        if BN: pos=[6,7,8,9]
        else: pos=[4,5]
        for i,p in enumerate(model_ext.parameters()):
            if i in pos:
                freezed.append(p)

    if position==2:
        channels=256
        if BN: pos=[10,11,12,13]
        else: pos=[6,7]
        for i,p in enumerate(model_ext.parameters()):
            if i in pos:
                freezed.append(p)

    t = torch.zeros([channels,channels,3,3])
    for i in range(channels):
        t[i,i,1,1]=1.
    if BN:
        new_weights_inits= [torch.ones([channels]), torch.zeros([channels]), t,torch.zeros([channels])]
    if not BN:
        new_weights_inits = [t,torch.zeros([channels])]

    with torch.no_grad():
        old_param_iterator = model_baseline.parameters()
        for p_new in model_ext.parameters():
            if not _is_freezed(p_new, freezed):
                p = next(old_param_iterator)
                p_new.copy_(p)
            else:
                p_new.copy_[new_weights_inits[i]]
                i+=1
    return model_ext


def select_new_model(sensitivities, model_baseline, mode='abs max', BN=False): # TODO
    # Unterscheidung BN/not BN
    # We only look at the filter parameters, 
    # this means that we do not consider the biases and in the BN case, 
    # also the trainable parameters of BN are ignored

    # select filter sensitivities
    # for BN this is the third parameter of the 4 of each layer, so 2,6,10
    # wo BN this is the first parameter of the 2 of each layer, so 0,2,4
    freezed_norms_only_filters=[]

    if BN:
        for k, sens in enumerate(sensitivities):
            if k%4 == 2:
                freezed_norms_only_filters.append(sens)

    if not BN:
        for k, sens in enumerate(sensitivities):
            if k%2 == 0:
                freezed_norms_only_filters.append(sens)
        



    # mode absmax, absmin, pos0
    if mode =='absmax':
        max_index = max(range(len(freezed_norms_only_filters)),
                        key=lambda l: freezed_norms_only_filters[l])
        position = max_index

    if mode =='abs min':
        min_index = min(range(len(freezed_norms_only_filters)),
                        key=lambda l: freezed_norms_only_filters[l])
        position = min_index
        

    if mode=='pos 0':
        position=0

    
    model = extend_VGG(position, BN)
    model = init_ext_net(model_baseline, model, position, BN)
    return model
    




def tmp_net(model_baseline, BN=False):
    model, freezed = build_vgg_fullyext(BN)
    model = init_fullyext_net(model_baseline, model, freezed, BN)
    return model, freezed



def calculate_shadow_prices_mb(traindataloader, model, freezed): 
    #not_frozen = number_of_free_parameters_collected(model, freezed)*[0]
    frozen = _number_of_frozen_parameters_collected(model, freezed)*[0]
    loss_values = []

    for X, y in traindataloader:
        model.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(
            model(X), y)
        loss.backward()
        loss_values.append(loss.item())

        k = 0  # counts the number of frozen parameters
        kk = 0  # counts the number of free parameters
        with torch.no_grad():
            for p in model.parameters():  # iterate over all model parameters
                # if parameter is not frozen, append gradient to list
                # if not _is_freezed(p, freezed):
                #     not_frozen[kk] += torch.sum(torch.square(
                #         torch.abs(p.grad)))/_number_of_params(p)
                #     kk += 1
                # if parameters is frozen, append one averaged gradient value to list
                if _is_freezed(p, freezed):
                    # print(p.grad) # uncomment if you want to see all shadow price values of the model parameters
                    # and not just an average
                    frozen[k] += torch.sum(torch.square(torch.abs(p.grad))
                                           )/_number_of_params(p)
                    k += 1

    # scale the frozen and unfrozen lists with by the numbers of batches
    scale = 1/len(loss_values)
    frozen = [scale*x for x in frozen]
    #not_frozen = [scale*x for x in not_frozen]

    sensitivities = []
    return sensitivities
