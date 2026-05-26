import torch
import torch.nn as nn
import torch.nn.functional as F

# AtomAction_NSVQ
'''
    Input:
        __init__:
        forward:

    Output:
        forward:

'''
class AtomAction_NSVQ(nn.Module):
    def __init__(self,model_args):
        super(AtomAction_NSVQ,self).__init__()


    def forward(self):
        pass

       
    '''计算模块参数量'''
    def print_module_params(self):
        pass

    ''' 冻结模块参数'''
    def freeze_modules(self, freeze_module=None):
        pass
