import losses
import os
import numpy as np
import torch
from torch.autograd import Variable
import IO as io
import glob
import re

# target example = '/home/tabish/Desktop/flownet2/data/FlyingChairs_examples/0000003-gt.flo'
# moddir = '/home/tabish/Desktop/weigts_graph/3/all_out/'
def epe_withinference(moddir, target):
    target = target
    moddir = moddir
    listd = sorted(glob.glob(moddir+"*.flo"))
    for mod_out in listd:
        print(losses.EPE(Variable(torch.from_numpy(io.read(mod_out))),Variable(torch.from_numpy(io.read(target)))))


def epe(comparition_file):
    with open(comparition_file,"r+") as reader:
        lines = reader.readlines()
    for line in lines:
        [target, mod_out] = line[:-1].split("\t")
        print(losses.EPE(Variable(torch.from_numpy(io.read(mod_out))),Variable(torch.from_numpy(io.read(target)))))