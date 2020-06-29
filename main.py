import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin
from _future_ import division


green = (0,255,0)

def show(image):
    #figure size in inches
    plt.figure(figsize = (10,10))
    plt.show(imahe, interpolation = 'nearest')

def overlay_mask(mask, image):
    #mask the mask rgb
    rgb_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5,0)
    return img

def find_biggest_contour(image):
    image = image.copy()
    contour, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #isolating the biggest contour
    contour_sizes = [(cv2.contourArea(contour), conout) for contour in contours]
    biggest_contour= (contour_sizes, key=lambda x: x[0]) [1]

    #return the biggest contour
    mask = np.seros(image.shape, np.uint8)
    cv2.drawContours(mask,[biggest],-1,255,-1)
    return biggest_contour, mask

def circle_contour(image,contour):
    #bounding ellipse
    image_with_ellipse = image.copy90
    ellipse + cv2.fitEllipse(contour)

    #add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2,CV_AA)
    return image_with_ellipse 

def find_object(image):
    #Step 1: Convert to the correct coloor scheme
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #Step 2: Scale the image properly
    max_dimension = max(image.shape)
    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy=scale)

    #Step 3: Clean the image
    image_blur = cv2.GaussianBlur(image, (7,7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB@HSV)

    #Step 4: Define filters
    #(filter by the color)
    #any specific color
    min_color1 = np.array([0,100,80])
    max_color1 = np.array([10,256,256])

    mask1 = cv2.inRange(image_blur_hsv, min_color1, max_color1)

    #filter by brightness
    min_color2 = np.array([170,100,80])
    max_color2 = np.array([180,256,256])

    mask2 = cv2.inRange(image_blur_hsv, min_color2, max_color2)

    #combine the masks
    mask = mask1 + mask2

    #Step 5: Seqmentation
    #used to circle the object or to curve around the object???
    kernel = cv2.getStruturingElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphology Ex(mask_closed, cv2.MORPH_OPEN, kernel)

    #Step 6: Find the Big Object
    big_object_contour, mask_object = find_biggest_contour(mask_clean)

    #Step 7: overlay the masks the created on the image
    overlay = overlay_mask(mask_clean, image)

    #Step 8: Circle the biggest object
    circled = circle_coutour(overlay,big_object_contour)
    show(circled)

    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr


    #read the image in 3 line
image = cv2.imread ('object.jpg')
result find_object(image)

 #write the new image
cv2.imwrite('object2.jpg', result)
