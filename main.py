from skimage import io
from code import interact

from matplotlib.pyplot import figure, subplot
from skimage import data as img
from skimage import filters
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, label2rgb
from skimage.feature import ORB, match_descriptors
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, disk, square
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

def method2(oryg,image):

    thresh = threshold_otsu(image)

    bw = closing(image > thresh, square(3))
    drawPlot(bw)
    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=oryg)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def extractRGB(img):
    R = np.zeros(img.shape)
    R[:,:,:] = [1,0,0]
    G = np.zeros(img.shape)
    G[:,:,:] = [0,1,0]
    B = np.zeros(img.shape)
    B[:,:,:] = [0,0,1]
    R = img*R
    G = img*G
    B = img*B
    #gray = rgb2gray(img)*255
    #R = np.maximum(0, gray - rgb2gray(img*G))
    #G = np.maximum(0, gray, rgb2gray(img*G))
    #B = np.maximum(0, gray - rgb2gray(img*B))
    return R,G,B

def color_distance(image):
    from skimage import io, color, img_as_float
    from skimage.feature import corner_peaks, plot_matches

    import matplotlib.pyplot as plt
    import numpy as np
    image = img_as_float(image)

    black_mask = color.rgb2gray(image) < 0.1
    distance_5gr = color.rgb2gray(1 - np.abs(image - (.83, .76, .59)))
    distance_1zl = color.rgb2gray(1 - np.abs(image - (0, 0, 1)))

    distance_5gr[black_mask] = 0
    distance_1zl[black_mask] = 0

    coords_red = corner_peaks(distance_5gr, threshold_rel=0.9, min_distance=50)
    coords_blue = corner_peaks(distance_1zl, threshold_rel=0.9, min_distance=50)

    f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 10))
    ax0.imshow(image)
    ax0.set_title('Input image')
    ax1.imshow(image)
    ax1.set_title('Marker locations')
    ax1.plot(coords_red[:, 1], coords_red[:, 0], 'ro')
    ax1.plot(coords_blue[:, 1], coords_blue[:, 0], 'bo')
    ax1.axis('image')
    ax2.imshow(distance_5gr, interpolation='nearest', cmap='gray')
    ax2.set_title("{dist}_{std}_{mid}".format(dist = np.mean(distance_5gr), std = np.std(distance_5gr), mid = np.median(distance_5gr)))
    ax3.imshow(distance_1zl, interpolation='nearest', cmap='gray')
    ax3.set_title('Distance to pure blue')
    plt.show()

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
            #fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            name  =  "fig_{q}_{w}".format(q = fig_name, w = iter)

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                          fill=False, edgecolor='red', linewidth=2)
            coin = image2[minr:maxr, minc:maxc]
            #r,g,b = extractRGB(coin)
            r = coin[:,:,0]
            g = coin[:,:,1]
            b = coin[:,:,2]
            diffrences = [[np.max(element) - np.min(element) for element in row if np.max(element) - np.min(element) < 20] for row in coin]

            #print(name, np.max([np.median(r),np.median(g),np.median(b)]) - np.min([np.median(r), np.median(g), np.median(b)]))
            ratio = sum(len(row) for row in diffrences) / (len(coin)*len(coin[0]))
            if ratio > .5:
                print(name, "jest srebrna")
            #coin = filters.median(coin, np.ones((5,5)))
            #extractRGB(coin)
            #color_distance(coin)
            #coin = label(rgb2gray(coin))
            #image_label_overlay = label2rgb(coin, image=image2[minr:maxr, minc:maxc])
            #drawPlot(image_label_overlay)
            #str = "{n};{all}".format(n = name, all = np.mean(coin))
            #log.write(str + "\n")
            iter+=1
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

log = open("out/figures_median_ones5.csv", "w+")
for file in os.listdir(directory):
    # images.append(rgb2gray(img.load(directory + file)))
    #ORB
    '''
    #image = img.load(directory + file, True)
    #detector.exec(img_template, histogram_manipulator.contrast_stretching(image[126:899, 138:871]), file)
    #img = hist.contrast_stretching(img)'''

    '''
    image = img.load(directory + file)
    method2(image, rgb2gray(image))
    '''


    #standard
    image = img.load(directory+file,True)
    image2 = img.load(directory+file,False)
    edges = skimage.filters.sobel(image)
    edges = skimage.feature.canny(edges,1.2)
    #meanV = getMean("sobel_max_", edges)
    thresh(0.08)
    fig_name += 1

log.close()
#displaySaveImage(images)
#plt.show()
