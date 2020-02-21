import math
import os
import json
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import skimage
from PIL.ImageOps import _lut
from skimage import morphology
import numpy.matlib
import scipy as sp
import scipy.ndimage
from PIL import Image, ImageOps
import functools
import operator
from tkinter import *
from tkinter.filedialog import askopenfilename
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import  easygui



def openimage():
   # Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    image_path = filename
    return image_path
def thresholding():
    # convert to grayscale
    imageTogray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    cv2.imshow('gray image', imageTogray)
    # cv2.waitKey(0) #here
    # treshold = input("Enter treshold value")
    th, threshed = cv2.threshold(imageTogray, 9, 255,
                                 cv2.THRESH_BINARY)  # tresh the image binary inverse and Otsu method
    cv2.imshow("Images", np.hstack([imageTogray, threshed]))
    # cv2.imwrite("treshed.jpg",threshed)
    # cv2.imwrite("gray.jpg",imageTogray)

    # cv2.waitKey(0) #here
    invTresh = 255 * (threshed < 128).astype(np.uint8)  # To invert the text to white (express from 0 to 255)
    return threshed
def fillholes():
    # Copy the thresholded image.
    im_floodfill = threshed.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = threshed.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Inv    # cv2.imshow("holes filled", im_out)ert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)


    # Combine the two images to get the foreground.
    im_out = threshed | im_floodfill_inv
    cv2.imshow("threshed and holes filled", np.hstack([threshed, im_out]))

    cv2.imwrite("holes filled.jpg",im_out)

    # cv2.waitKey(0) #here

    return im_out


def removetext():
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_out, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1000

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    cleaned = img2

    cv2.imshow("cleaned", cleaned) #this shows teesholded hol;es filled and cleaned image
    # cv2.waitKey(0) #here

    imageTograyinv = cv2.bitwise_not(imageTogray)
    # cv2.imshow("imginb",im_invert)
    # cv2.waitKey(0) #here

    b2 = imageTograyinv.astype(np.float)
    ineww = np.tile(image, (3, 1, 1)).shape

    triii = np.multiply(b2, cleaned);
    # cv2.imshow("ineww",triii)

    what = triii.astype(np.uint8)
    # what= cv2.cvtColor(what, cv2.COLOR_BGR2GRAY)

    cv2.imshow("original cleaned ", what) #the original image cleaned
    cv2.imwrite("cleaned.jpg",what)
    # cv2.waitKey(0) #here
    #################################33




    return  what

def adjust():
    gamma = 0.19
    adjusted = adjust_gamma(cleaned, gamma=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    # cv2.waitKey(0)
    # print(gamma)
    # adjustedhis = cv2.equalizeHist(adjusted)
    # cv2.imshow('equaliz', adjustedhis)
    # cv2.imwrite("equalized.png",)
    # cv2.waitKey(0)

    cv2.imshow('test', Watershed(adjusted))
    # cv2.waitKey(0) #here
    # arr = np.asarray(what)
    # adjusted=imadjust(what,what.min(),what.max(),0.3,0.7)

    hist, bins = np.histogram(adjusted.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    adjusted2 = cdf[adjusted]
    cv2.imshow("Images", np.hstack([adjusted, adjusted2]))
    # cv2.waitKey #here
    cv2.imwrite("contrast.jpg",adjusted)
    cv2.imwrite("histoequal.jpg",adjusted2)

    return adjusted2

def morph(medfilt2,choice):
    if choice=='1': #roi
        # Select ROI
        r = cv2.selectROI(medfilt2)

        # Crop image
        imCrop = medfilt2[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # Display cropped image
        cv2.imshow("Image", imCrop)
        # cv2.waitKey(0) #here

        medfilt2 = imCrop
    elif choice == '2': #auto
        h, w = medfilt2.shape;
        print(h, w)
        medfilt2 = medfilt2[(int)(w / 3):(int)(w / 3 + 40), (int)(h / 2 + 20):(int)(480)]
        cv2.imshow("new crop", medfilt2)
        # cv2.waitKey(0)#here

    th, threshedMed = cv2.threshold(medfilt2, 235, 255, cv2.THRESH_BINARY)
    cv2.imshow('treshed 235', threshedMed)
    # cv2.waitKey(0) #here

    se = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    # se= scipy.ndimage.morphology.generate_binary_structure(5, 1)
    dilated = cv2.dilate(threshedMed, se, 1)

    cv2.imshow('dilated', dilated)
    # cv2.waitKey (0) #here
    return dilated

def Watershed(img):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.14 * dist_transform.max(), 255, 0) # originally 0.14

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    print(image_path)
    img2 = cv2.imread(image_path)
    #img2 = cv2.medianBlur(img2, 5)
    markers = cv2.watershed(img2, markers)
    img2[markers == -1] = [255, 0, 0]

    return img2

def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def showfigure(imageTogray,cleaned,medfilt2,dilated):

    import matplotlib.pyplot as plt



    # create the figure
    fig = plt.figure()

    # display original image
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(imageTogray, cmap=plt.cm.gray,
              vmin=0, vmax=255)

    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')


    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(cleaned, cmap=plt.cm.gray,
              vmin=0, vmax=255)

    ax.set_xlabel('Image Preprocessed')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(medfilt2, cmap=plt.cm.gray,
              vmin=0, vmax=255)

    ax.set_xlabel('Original Morphology')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(dilated, cmap=plt.cm.gray,
              vmin=0, vmax=255)

    ax.set_xlabel('Image Stone')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # display the image patches

    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()





    #############################################################################


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

root = Tk()
root.title("Kidney Disease predictor")

frame1 = Frame(root)
frame1.pack()
button_load_database= Button(frame1,text = "Load image ",command=openimage)
button_load_database.grid(row=0)
image_path=openimage()
# image_path=r'treshed.jpg'
#
frame2=Frame(root)
frame1.pack
photo=PhotoImage('treshed.jpg')
labelpic=Label(frame2,image=photo )

#root.geometry('350x200')
#read normal image# labelpic.pack()
#
image = cv2.imread(image_path)
cv2.imshow('original image',image)
#cv2.waitKey(0)   #heere


imageTogray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshed=thresholding()
im_out=fillholes()
cleaned=removetext()
adjusted=adjust()
medfilt2 = cv2.medianBlur(adjusted, 5)
# choice=input("Press 1 for roi selection and 2 for automatic")
choice = easygui.enterbox("Press 1 for roi selection and 2 for automatic")

h,w,s=image.shape;
dilated = morph(medfilt2, choice) #for roi and morph //
blank_image = np.zeros(shape=[h, w], dtype=np.uint8)

x_offset,y_offset=dilated.shape;
x_offset=(int)(0.5*(h-x_offset))
y_offset=(int)(0.5*(w-y_offset))
blank_image[y_offset:y_offset+dilated.shape[0], x_offset:x_offset+dilated.shape[1]] = dilated
# dilated= dilated * blank_image;
dilated = blank_image
cv2.imshow("dilated",blank_image)
# cv2.waitKey(0)#here
showfigure(imageTogray,cleaned,medfilt2,dilated);


ret,labels= cv2.connectedComponents(dilated,4)
print("ret is")
print(ret)
print("labels is ")
print(labels)
# getting mask with connectComponents
# ret, labels = cv2.connectedComponents(dilated,4 )
for label in range(1,ret):
    mask = np.array(labels, dtype=np.uint8)
    mask[labels == label] = 255
    cv2.imshow('component',mask)
   # cv2.waitKey(0)  # here

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# getting ROIs with findContours
contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
# for cnt in contours:
#     (x,y,w,h) = cv2.boundingRect(cnt)
#     ROI = image[y:y+h,x:x+w]
#     cv2.imshow('ROI', ROI)
#     cv2.waitKey(0)

cv2.destroyAllWindows()


if(ret > 1):
    print('stone detected')
else:
    print('no stone detected ')

distances = [2]
angles = [0, np.pi/2]
properties = ['energy', 'homogeneity']

glcm = greycomatrix(dilated,
                    distances=distances,
                    angles=angles,
                    symmetric=True,
                    normed=True)

feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
print(entropy(dilated))
print(feats)

root.mainloop()
