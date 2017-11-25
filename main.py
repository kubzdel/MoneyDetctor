from skimage import io
from code import interact

from matplotlib.pyplot import figure, subplot
from skimage import data as img
from skimage import filters
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, disk
from skimage.draw import line
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
import histogram_manipulator
import ORB_detector as detector
import cv2


import PIL
from IPython.display import display, Image


def drawPlot(image):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()

def getMean(desc, image):
    mean = np.mean(image)
    print("{} image mean = {}".format(desc,mean))
    return mean

def getMax(desc, image):
    maxV = max(np.max(image, axis=1))
    print("{} image max = {}".format(desc, maxV))
    return maxV

def region_is_inside_another(region_sizes, region_to_check):
    for region in region_sizes:
        if region_to_check[0] > region[0] and region_to_check[1] > region[1] and region_to_check[2] < region[2] and region_to_check[3] < region[3]:
            return True
    return False

def circle_detector(x, y, num):
    xc = np.mean(x)
    yc = np.mean(y)
    r = (x-xc)**2 + (y-yc)**2 #try to figure out radius of the circle with the middle of the xc and yc
    return ("%.2f" % (100 * np.std(r) / np.mean(r)), ( np.std(r) / np.mean(r)))

def thresh(t):

    binary = skimage.morphology.dilation(edges)
    binary = label(binary)
    #drawPlot(binary)

    regions = regionprops(binary)
    region_sizes = [reg.bbox for reg in regions]
    iter  = 0
    plt.tight_layout()
    for region in regions:
        if region_is_inside_another(region_sizes, region.bbox):
            continue

        # take regions with large enough areas
        if region.area >= 1300 and region.area < (.5*len(binary)*len(binary[0])):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            name  =  "fig_{q}_{w}".format(q = fig_name, w = iter)

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                          fill=False, edgecolor='red', linewidth=2)
            coin = image2[minr:maxr, minc:maxc]
            coin = filters.median()

            #str = "{n};{rgb};{r};{g};{b};{val}".format(n = name, rgb = np.mean(coin), r = np.mean(coin[:,:,0]), g = np.mean(coin[:,:,1]), b = np.mean(coin[:,:,2]), val = np.mean(rgb2gray(coin)))
            #log.write(str + "\n")


            continue
            #coin = filters.median(rgb2gray(coin), np.ones((5,5)))
            ax.imshow(rgb2gray(coin))
            plt.show()
            plt.savefig("out/" + name)
            iter += 1
            minc = 0
            minr = 0
            color = 'k'
            break
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.imshow(coin[:,:,1])
            plt.show()
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.imshow(coin[:,:,2])
            plt.show()
            continue
            colors = [np.mean(coin[:,:,i]) for i in range(0,3)]

            name  =  "fig_{q}_{w}".format(q = fig_name, w = iter)
            text_0 = "r:%.2f" %  np.mean(colors[0])
            text_1 =  "g:%.2f" %  np.mean(colors[1])
            text_2 =  "b:%.2f" %  np.mean(colors[2])
            text_3 = " rozrzut: %.2f" % (np.max(colors) - np.min(colors))
            #text_file.write(name + ";" + text_0 + ";" + text_1 + ";" + text_2+ "\n  ")
            ax.text(minc, minr - 10, name + " " + text_0 + " " + text_1 + " " + text_2 + " " + text_3, color=color, fontsize=10)
            #images.append(coin)
            #ax.add_patch(rect)
            plt.show()
            #fig.savefig("out/" + name)
            iter += 1
            flush_figures()


    #ax.set_axis_off()
    #plt.tight_layout()
    # print(contours)



directory = os.getcwd()+"\\templates" + '/'
images = []
fig_name = 0

img_templates = ((.05, "5gr.png"), (1, "1zl.png"))

img_template = histogram_manipulator.contrast_stretching(img.load(directory + "1zl.png", True))
'''orb = cv2.ORB_create()
kp = orb.detect(imag, None)
kp_template, des = orb.compute(imag, kp)
img3 = np.zeros(imag.shape)
img2 = cv2.drawKeypoints(imag, kp, img3, color=(0,255,0), flags=0)
plt.imshow(img2)
plt.show()
'''
directory = os.getcwd()+"\\moje" + '/'

log = open("out/figures.csv", "w+")
for file in os.listdir(directory):
    # images.append(rgb2gray(img.load(directory + file)))
    #ORB
    '''
    #image = img.load(directory + file, True)
    #detector.exec(img_template, histogram_manipulator.contrast_stretching(image[126:899, 138:871]), file)
    #img = hist.contrast_stretching(img)'''
    image = img.load(directory+file,True)
    image2 = img.load(directory+file,False)
    edges = skimage.filters.sobel(image)
    edges = skimage.feature.canny(edges,1.2)
    meanV = getMean("sobel_max_", edges)
    thresh(0.08)
    fig_name += 1
log.close()
#displaySaveImage(images)
#plt.show()
