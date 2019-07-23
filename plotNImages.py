import matplotlib.pyplot as plt
import cv2

def plotnasdf(images):
    f = plt.figure()
    for i,image in enumerate(images):
        print(i)
        f.add_subplot(1,len(images), i+1)
        plt.imshow(image)
    plt.show(block=True)