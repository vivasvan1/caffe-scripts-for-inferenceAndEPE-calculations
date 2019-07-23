import os
import sys
import glob

for i in os.listdir(sys.argv[1]):
    paths = glob.glob(os.path.join(sys.argv[1],i)+"/*.jpg")
    # print(paths)
    for index,j in enumerate(paths):
        paths[index] = os.path.abspath(j)
    # print(paths)
    # print(os.path.dirname(paths[0])+"/out.flo")
    command = """python3 ./realKlenstest.py --gpu 1 --verbose /home/tabish/Desktop/flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5 /home/tabish/Desktop/flownet2/models/FlowNet2-SD_deploy.prototxt """+paths[0]+""" """+paths[1]+""" """+os.path.dirname(paths[0])+"/out.flo"
    print(command)
    os.system(command)