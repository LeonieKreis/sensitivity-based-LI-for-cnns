import torch
import time

from nets import build_vgg_baseline, build_vgg_fullyext, extend_VGG
from model_selection import tmp_net, calculate_shadow_prices_mb, select_new_model
from train_and_test_ import train, test

def training_with_one_LI(epochs, traindataloader, testdataloader,BN=False, optimizer_type='SGD', lr_init=0.1, mode='abs max', stopping_criterion=None,lrschedule_type='StepLR', lrscheduler_args=None,decrease_lr_after_li=1.,save_grad_norms=False):
    losses = []
    accs = []
    times = []
    grad_norms= []

    model_baseline = build_vgg_baseline(BN)
    print(f'Starting training on baseline model...')

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model_baseline.parameters(), lr_init, momentum=0.9, weight_decay=5e-4)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model_baseline.parameters(), lr_init)

    # build lr scheduler
    if lrschedule_type == 'StepLR':
        if isinstance(lrscheduler_args['step_size'],list):
            step_size = lrscheduler_args['step_size'][0]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
            
    if lrschedule_type == 'MultiStepLR':
        if isinstance(lrscheduler_args['step_size'][0],list):
            step_size = lrscheduler_args['step_size'][0]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size, gamma=gamma)
        
    ############## TRAIN BASELINE ######################################
        
    losses1, test_accs1, times1, grad_norms1 = train(model_baseline,traindataloader,
                                        testdataloader,optimizer,epochs[0],
                                        lrscheduler, stopping_criterion=stopping_criterion, 
                                        save_grad_norms=save_grad_norms)

    ############# BUILD TMP NET ########################################
    print(f'Starting layer selection...')
    toc = time.time()
    model_fullyext, freezed = tmp_net(model_baseline, BN)

    ######## COMPUTE SENSITIVITIES #####################################
    print('calculate sensitivities...')
    sensitivities = calculate_shadow_prices_mb(traindataloader, 
                                               model_fullyext, freezed)
    
    ####### SELECT NEW MODEL ###########################################
    print('build new model...')
    model_ext = select_new_model(sensitivities, model_baseline,mode,BN)

    ##### DECREASE LR ##################################################
    lr_end = optimizer.param_groups[0]['lr']
    lr_init_ext = decrease_lr_after_li * lr_end

    tic = time.time()

    time_model_sel = tic- toc
    ######## TRAIN ON EXTENDED MODEL ###################################

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model_ext.parameters(), lr_init_ext, momentum=0.9, weight_decay=5e-4)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model_ext.parameters(), lr_init_ext)

    # build lr scheduler
    if lrschedule_type == 'StepLR':
        if isinstance(lrscheduler_args['step_size'],list):
            step_size = lrscheduler_args['step_size'][1]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
            
    if lrschedule_type == 'MultiStepLR':
        if isinstance(lrscheduler_args['step_size'][0],list):
            step_size = lrscheduler_args['step_size'][1]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size, gamma=gamma)


    losses2, test_accs2, times2, grad_norms2 = train(model_ext,traindataloader,
                                        testdataloader,optimizer,epochs[1],
                                        lrscheduler, stopping_criterion=stopping_criterion,
                                        save_grad_norms=save_grad_norms)
    
    times2[0]= times2[0]+ time_model_sel

    losses = losses1+losses2
    accs = test_accs1 + test_accs2
    times = times1 + times2
    grad_norms= grad_norms1+grad_norms2

    return losses, accs, times, grad_norms