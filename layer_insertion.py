from model_selection import tmp_net, calculate_shadow_prices_mb, select_new_model
from train_and_test_ import train, test

def training_with_one_LI(epochs, traindataloader, testdataloader, optimizer_type='SGD', lr_init=0.1, mode='abs max', stopping_criterion=None,lrscheduler='StepLR', lrscheduler_args=None,decrease_lr_after_li=1.,save_grads=False):
    losses = []
    accs = []

    return losses, accs