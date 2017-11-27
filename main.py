from skimage import io
from code import interact
import operator
from matplotlib.pyplot import figure, subplot
from skimage import data as img
from skimage import filters, segmentation, exposure
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, label2rgb, rgb2hsv
from skimage.feature import ORB, match_descriptors
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, disk, square, erosion, dilation, watershed
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

from skimage.morphology import convex_hull_image



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

def drawPlot(image, patch = False):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(image, cmap=plt.cm.gray)
    if patch:
        rect = mpatches.Circle((len(image[0])/2, len(image)/2), len(image)*.3 , color = 'k', fill=True , edgecolor='k', linewidth=2)
        ax.add_patch(rect)
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



def silver_detector(coin):

    diffrences = [[np.max(element) - np.min(element) for element in row if np.max(element) - np.min(element) < 10] for
                  row in coin]

    ratio = sum(len(row) for row in diffrences) / (len(coin) * len(coin[0]))
    return ratio,ratio > .5

def silver_detector_clear_background(coin):
    ratio = 0
    points = 0
    eligible = 0    #pixels which has small variety of rgb components
    for row in coin:
        for element in row:
            if np.all(element == 0):
                continue
            value = np.max(element) - np.min(element)
            if value < 10:
                eligible += 1
            points += 1

    if points > 0:
        ratio = eligible / points

    return (ratio,ratio > .095)

def circle_detector(x, y, num):
    xc = np.mean(x)
    yc = np.mean(y)
    r = (x-xc)**2 + (y-yc)**2 #try to figure out radius of the circle with the middle of the xc and yc
    return ("%.2f" % (100 * np.std(r) / np.mean(r)), ( np.std(r) / np.mean(r)))

def clear_background(img, mask):
    for row in range(0, len(img)):
        for column in range(0, len(img[0])):
            if mask[row][column]:
                img[row][column] = [0,0,0]

#rgb
def calculate_mean(img, pos):
    img2 =(img)[:,:,pos]
    img2 = np.where(img2 == 0, np.nan, img2)
    mean = np.nanmean(img2)
    return mean

def get_ORB_detector_templates():
    directory = os.getcwd() + "\\templates" + '/'
    img_templates = [[.05, "5gr.png"], [1, "1zl.png"]]
    img_templates = [histogram_manipulator.contrast_stretching(img.load(directory + name[1], True)) for name in img_templates]
    return img_templates

def draw_plot_with_oryginal(image, oryg_image, name = "image.png"):
    fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,10))
    plt.tight_layout()
    ax.imshow(image, cmap=plt.cm.gray)
    ax2.imshow(oryg_image, cmap=plt.cm.gray)
    plt.show()
    #plt.savefig(name)
    flush_figures()

def plot_histogram(img, name):
    fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,10))
    histo, x = np.histogram(img*255, range(0, 256), density=True)
    ax.plot(np.arange(0, 255, 1),histo)
    ax2.imshow(img)
    #plt.show()
    fig.savefig("hist_nowe/" + name + ".png")
    #plt.hist(img.ravel(), 256, [0, 256])
    #plt.show()

#maska będąca otoczką wypukłą
def get_convex_hull_mask(img_binary, label):
    bin = (img_binary == label)
    return convex_hull_image(bin)

#maska bedaca wypelnieniem ksztaltu z labelowania
def get_filled_shape_mask(img_binary):
    return ndi.binary_fill_holes(dilation(img_binary))
def save_coin_with_text(coin, text, dir, name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(coin)
    ax.text(0, -10, text, color='k', fontsize=10)
    plt.savefig(dir + "/" + name + ".png")

def top4_h_values(img_color):
    h = rgb2hsv(img_color)[:, :, 0]
    h = h * 360
    h = h.astype(int)
    unique, counts = np.unique(h, return_counts=True)
    values = dict(zip(unique, counts))
    h_sorted = sorted(values.items(), key=operator.itemgetter(1))
    # pobierz najczęściej występujące top 4 wartości h
    return h_sorted[-4:]
def get_saturaition(img_rgb):
    return rgb2hsv(img_rgb)[:, :, 1]

def find_contours_based_coins_detector(img_binary):
    contours = find_contours(img_binary, .2, fully_connected='high')
    coins = []
    for i, img in enumerate(contours):
        coins.append(img)

    return coins

#uses region based detection
def region_based_coins_detector(img_binary):
    markers = np.zeros_like(img_binary)
    markers[img_binary < np.percentile(img_binary, 45)] = 1
    markers[img_binary > np.percentile(img_binary, 90)] = 2
    unique, counts = np.unique(markers, return_counts=True)
    values = dict(zip(unique, counts))
    print(values)
    elevation_map = filters.sobel(img_binary)
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_coins, _ = ndi.label(segmentation)
    return labeled_coins

def edge_based_coin_detector(img_binary):
    binary = dilation(img_binary)
    fill_coins = fill_coins = ndi.binary_fill_holes(binary)
    label_objects, nb_labels = ndi.label(fill_coins)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 200
    mask_sizes[0] = 0
    coins_cleaned = mask_sizes[label_objects]
    return coins_cleaned

def get_hsv_mean(img_rgb_color):
    img_color = rgb2hsv(img_rgb_color)
    colors = [np.mean(img_color[:, :, i]) for i in range(0, 3)]
    return np.mean(colors)

def save_data_to_log(text):
    log.write(text + "\n")

def coins_detector(fig_name):
    binary = skimage.morphology.dilation(edges)
    binary = label(binary)

    regions = regionprops(binary)
    region_sizes = [reg.bbox for reg in regions]
    coin_number  = 0
    coins_on_image = []
    for region in regions:
        #check region's overlapping
        if region_is_inside_another(region_sizes, region.bbox):
            continue

        # take larger regions that don't exceed 30% of the image surface
        if region.area >=  (.001*len(binary)*len(binary[0])) and region.area  < (.2*len(binary)*len(binary[0])):
            name  =  "{fig_name}_{number}".format(fig_name = fig_name, number = coin_number)

            # get region
            minr, minc, maxr, maxc = region.bbox

            coin = image_color[minr:maxr, minc:maxc]

            #promień na 60% szerokości zdjęcia  int(len(coin)*.3)
            coin2 = cv2.circle(coin, (int(len(coin[0])/2) , int(len(coin)/2)), int(len(coin[0])*.3), color=[0,0,0], thickness=-1)


            #kwadrat w środku
            #coin[int(len(coin)*.3):int(len(coin)*.7), int(len(coin[0])*.3): int(len(coin[0])*.7)] = [0,0,0]


            coin_binary = binary[minr:maxr, minc:maxc]
            convex_hull_mask = get_convex_hull_mask(coin_binary, region.label)


            #clear_background(coin, ~get_filled_shape_mask(coin_binary))
            clear_background(coin2, ~convex_hull_mask)

            drawPlot(coin)
            continue

            ratio, is_silver = silver_detector_clear_background(coin)
            '''
            is_dwojka = 'n'
            if is_silver:
                mean_dwojka = calculate_mean(coin)
                is_dwojka= 't'
            '''
            #oblicz średnie, nie uwzględniaj czarnego, średnie dla wszystkich 3 kanałów rgb
            means = [calculate_mean(coin,i) for i in range(0,3)]
            #mean_cut = calculate_mean(coin)

            #1 średnia z kanałów
            mean_cut = np.mean(means)
            #zapisz histogram
            #plot_histogram(coin, name)

            h_sorted = top4_h_values(coin)[:-1] #take 3 top, dismiss first one cause its black color
            h_values = [val[0] for val in h_sorted]
            s = np.mean(get_saturaition(coin))
            #values = "{};{};{}".format(h_values[0], h_values[1], h_values[2])
            values = ";;;"
            print(values)
            if is_silver:
                coin_class = "s"
            else:
                coin_class = "ns"
            figure_description = "{n};{m};{s};{r};{h};{c_class};{area};{sat};{is_dwoj}".format(n = name, m = mean_cut, s = is_silver, r = ratio, h = values, c_class = coin_class, area =region.area, sat = s, is_dwoj = 'n')
            save_data_to_log(figure_description)
            save_coin_with_text(coin, figure_description, "temp", name)

            coin_number += 1


directory = os.getcwd()+"\\data_3" + '/'
log = open("temp/nowe.csv", "w+")
for file in os.listdir(directory):
    #standard
    image_color = img.load(directory + file, False)
    image_gray = rgb2gray(image_color)
    '''
    image2 = rgb2hsv(image2)
    h = image2[:,:,0]   #take h values
    h_prim = filters.median(h)
    R = np.zeros(image2.shape)
    R[:,:,:] = [1,0,0]
    image2 = image2+h_prim*R
    drawPlot(image2)
    '''
    edges = skimage.filters.sobel(image_gray)
    edges = skimage.feature.canny(edges,1.2)
    coins_detector(file)

log.close()
