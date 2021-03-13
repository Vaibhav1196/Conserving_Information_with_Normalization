'''

Method to execute the program:-   python rescale.py --image /path/to/image.png

'''

import numpy as np
import cv2
import argparse


def convolve(image,K):

    # grab the spatial dimensions of the image and kernel
    (iH,iW) = image.shape[:2]
    (kH,kW) = K.shape[:2]

    
    # Pad the image by copying the borders 
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Create an output array to store the output convolved image 
    output = np.zeros((iH,iW), dtype="float")


    # Loop over the image and perform the convolution 
    for y in np.arange(pad, iH+pad):
    
        for x in np.arange(pad, iW+pad):

            # Extract the Region of interest in the image by slicing
            # Perform the convolution 
            # Store the convolved pixel in the output
            roi = image[y-pad:y+pad+1,x-pad:x+pad+1]
            k = (roi*K).sum()
            output[y-pad, x-pad] = k



    # This is the section involving the post processing after convolution 
    # First we perform clipping , that is values < 0 = 0 and values > 255 = 255
    # Next we perform linear scaling with [min,max] => [0,255]
    # Finally we rescale the pixel intensities by multiplying with 255 
    output[output < 0] = 0
    output[output > 255] = 255
    output = (output-0)/(255-0)

    output = (output*255).astype("uint8")

    # Finally we return the output image to the calling function

    return output

########################################################################################################



# Load the input image and convert it to grayscale


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

########################################################################################################


# The Sobel kernels can detect edge like regions along both x and y axis:


# Construct the sobel x-axis kernel
sobelX = np.array(([-1,0,1], [-2,0,2], [-1,0,1]), dtype="int")

# Construct the sobel y-axis kernel
sobelY = np.array(([-1,-2,-1], [0,0,0], [1,2,1]), dtype="int")


# The kernel bank consists of all the filters that we would like to apply on the image 
kernelBank = (("sobel_x", sobelX), ("sobel_y", sobelY))


########################################################################################################


# This is the main Code where we loop through each kernel in kernel bank and apply it in the image
# We will use our user defined convolve method for convolution
# Also we will use the inbuilt cv2.filter2D for convolution , so that we can compare our method with opencv's method

   
for (kernelName, K) in kernelBank:

    # Apply the kernel to grayscale image using both our custom 'convolve' and CV2's 'filter2D' function

    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray,K)
    opencvOutput = cv2.filter2D(gray,-1, K)


    # show the output images 
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)

    
    cv2.waitKey(0)

    cv2.destroyAllWindows()

########################################################################################################








































