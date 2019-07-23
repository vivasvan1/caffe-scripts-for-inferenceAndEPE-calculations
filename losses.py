'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss

import torch
import torch.nn as nn
import math

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue


class MvLoss(nn.Module):
    def __init__(self, args):
        super(MvLoss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target, fusion_param=0.25):
        data_loss = self.loss(output, target)
        fusion_loss = torch.abs(output[:, 0, :, :] + output[:, 1, :, :]).mean()
        lossvalue = data_loss + fusion_param * fusion_loss
        print(output.shape)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                if i == 0:
                    epevalue = EPE(output_, target_) * self.loss_weights[i]
                    lossvalue = self.loss(output_, target_) * self.loss_weights[i]
                else:
                    epevalue += EPE(output_, target_) * self.loss_weights[i]
                    lossvalue += self.loss(output_, target_) * self.loss_weights[i]
            return [lossvalue, epevalue]
        else:
            epevalue = EPE(output, target)
            lossvalue = self.loss(output, target)
            return  [lossvalue, epevalue]
