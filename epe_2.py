import losses
import os
import numpy as np
import torch
from torch.autograd import Variable
import IO as io
import glob
import re
import glob
import sys
import code
import time
"""

HOW TO USE:
python3 epe_2.py [IMAGE_LIST] [PATH OF FOLDER CONTAING MULTIPLE MODELS]
IMAGE_LIST = "./data/Chair_test_release.list"

"""

def compareEPEonManyModelswithInference(modelsdir,imagepath1,imagepath2,targetflow):
    target = targetflow
    moddir = os.path.abspath(modelsdir)+"/"
    models = glob.glob(moddir+"*.caffemodel")
    for model in models:
        print(str(models.index(model)) + " out of " + str(len(models)))
        command = "python3 ./realKlenstest.py --gpu 1 "+model+" /home/tabish/Desktop/flownet2/models/FlowNet2-SD_deploy.prototxt "+imagepath1+" "+imagepath2+" "+model+".flo"
        os.system(command)

    listd = sorted(glob.glob(moddir+"*.flo"))
    EPEs = {}
    for mod_out in listd:
        # print(targetflow)
        newNP = np.array(io.read(target)[:,:,0:2]).copy()
        # print(torch.from_numpy(io.read(mod_out)).size())
        EPEs.update({mod_out: losses.EPE(Variable(torch.from_numpy(io.read(mod_out))),Variable(torch.from_numpy(newNP)))})
    return EPEs

if __name__ == "__main__":
    totalEPEs = {}
    with open(sys.argv[1]) as imagelist:
        lines = imagelist.readlines()
        # print(lines)
        for line in lines:
            print(str(lines.index(line)) + " out of " + str(len(lines)))
            print(line)
            [Img1Path,Img2Path,flowPath] = line[:-1].split("\t")
            now = time.time()
            EPEs = compareEPEonManyModelswithInference(os.path.abspath(sys.argv[2]), Img1Path, Img2Path, flowPath)
            print(time.time()-now)
            totalEPEs = { k: totalEPEs.get(k, 0) + EPEs.get(k, 0) for k in set(totalEPEs)|set(EPEs) }
            print(totalEPEs)
    code.interact(local=locals)

def onlyinference(modelsdir,imagelist_txt,inferencePath="./work"):
    with open(imagelist_txt,'r') as imagelist:
        lines = imagelist.readlines()
    
    for line in lines:
        print(str(lines.index(line)) + " out of " + str(len(lines)))
        print(line)
        [Img1Path,Img2Path,flowPath] = line[:-1].split("\t")
        now = time.time()
        moddir = os.path.abspath(modelsdir)+"/"
        models = glob.glob(moddir+"*.caffemodel")
        for model in models:
            print(str(models.index(model)) + " out of " + str(len(models)))
            command = "python3 ./realKlenstest.py --gpu 1 "+model+" /home/tabish/Desktop/flownet2/models/FlowNet2-SD_deploy.prototxt "+Img1Path+" "+Img2Path+" "+os.path.abspath(inferencePath)+"/"+os.path.splitext(model)+"_"+os.path.splitext(Img1Path)[0].split("/")[-1]+".flo"
            os.system(command)
            if os.path.exists("./comparition_"+os.path.dirname(modelsdir)+"_"+os.path.splitext(model).split("/")[-1]+".txt"):
                with open("./comparition_"+os.path.dirname(modelsdir)+"_"+os.path.splitext(model).split("/")[-1]+".txt", "a") as writing:
                    writing.write(flowPath+"\t"+os.path.abspath(inferencePath)+"/"+os.path.splitext(model).split("/")[-1]+"_"+os.path.splitext(Img1Path)[0].split("/")[-1]+".flo"+"\n")
            else:
                with open("./comparition_"+os.path.dirname(modelsdir)+"_"+os.path.splitext(model).split("/")[-1]+".txt", "w") as writing:
                    writing.write(flowPath+"\t"+os.path.abspath(inferencePath)+"/"+os.path.splitext(model).split("/")[-1]+"_"+os.path.splitext(Img1Path)[0].split("/")[-1]+".flo"+"\n")
