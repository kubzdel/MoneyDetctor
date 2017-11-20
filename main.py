from skimage import io
from code import interact

from matplotlib.pyplot import figure, subplot
from skimage import data as img
from skimage import filters
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import closing
from skimage.segmentation import clear_border
import skimage.morphology as mp
import numpy as np
from scipy import ndimage as ndi
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from ipykernel.pylab.backend_inline import flush_figures
from skimage.measure import find_contours
from skimage import feature
import skimage
import os
import matplotlib.patches as mpatches
import random
from glob import glob

import PIL
from IPython.display import display, Image

def getMean(desc, image):
    mean = np.mean(image)
    print("{} image mean = {}".format(desc,mean))
    return mean

def getMax(desc, image):
    maxV = max(np.max(image, axis=1))
    print("{} image max = {}".format(desc, maxV))
    return maxV

def neighbour(array, x, y):
    for a in range (od,do):
        for b in range(od,do):
            if array[x+a][y+b] !=0 and array[x+a][y+b] !=255:
                array[x][y] = array[x+a][y+b]
                neighbour2(array,x,y)
                return
    array[x][y] = colors.pop()
    neighbour2(array, x, y)



def neighbour2(array, x, y):
    for a in range(od,do):
        for b in range(od, do):
            if array[x + a][y + b] == 255:
                neighbour(array, x+a, y+b)
                return
    return


def colorize(array):
    rows = len(array)
    columns = len(array[0])
    global colors
   # colors = [10,40,50,60,80,120,140,160,180,200,220,240]
    colors = list(range(1,254,2))
    print(colors)
    for x in range(rows):
        for y in range(columns):
            if array[x][y] == 255:
                neighbour(array, x, y)



def displaySaveImage(imgs, filename="planes_bin.png"):
    fig = figure(figsize=(20,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 +1)
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig(filename, dpi=500)



def thresh(t):
    # warnings.simplefilter("ignore")
    binary = (edges > t) * 255
    binary = np.uint8(binary)
    #   binary[500, 500] = 100
    #binary = skimage.morphology.erosion(skimage.morphology.dilation(binary))
    binary = skimage.morphology.dilation(binary)
    # plt.contour(binary)
    #colorize(binary)
    #binary = clear_border(binary)
   # fill_coins = ndi.binary_fill_holes(binary)
   # binary = skimage.color.gray2rgb(binary)
    binary = label(binary)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(binary)

    for region in regionprops(binary):
        # take regions with large enough areas
        if region.area >= 1300:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox

            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
         #   print(region.mean_intensity)
            coin = image[minr:maxr, minc:maxc]
            images.append(coin)
            ax.add_patch(rect)


    ax.set_axis_off()
    plt.tight_layout()
    # print(contours)


    #for list in contours:
     #   xs, ys = [*zip(*list)]
        # plt.plot(xs,ys)


        #  print(contours)
   # flush_figures()




directory = os.getcwd()+"\moje" + '/'
images = []

for file in os.listdir(directory):
    image = img.load(directory+file,True)
    image2 = img.load(directory+file,False)
    edges = skimage.filters.sobel(image)
    edges = skimage.feature.canny(edges,1.2)
    meanV = getMean("sobel_max_", edges)
    thresh(0.08)
displaySaveImage(images)
plt.show()
