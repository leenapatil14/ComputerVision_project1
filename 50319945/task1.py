"""
Image Filtering
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with image filtering and familiarize you with 'tricks', e.g., padding, commonly used by computer vision 'researchers'.

Please complete all the functions that are labelled with '# TODO'. Steps to complete those functions are provided to make your lives easier. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
are building blocks you could use when implementing the functions labelled with 'TODO'.

I strongly suggest you read the function 'zero_pad' and 'crop' that are defined in 'utils.py'. You will need them!

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""

import argparse
import copy
import os

import cv2
import numpy as np

import utils

# low_pass filter and high-pass filter
low_pass = [[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]
high_pass = [[-1/8, -1/8, -1/8], [-1/8, 2, -1/8], [-1/8, -1/8, -1/8]]


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task1.jpg",
        help="path to the image"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="high-pass",
        choices=["low-pass", "high-pass"],
        help="type of filter"
    )
    parser.add_argument(
        "--result-saving-dir",
        dest="rs_dir",
        type=str,
        default="./results/",
        help="directory to which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")

    cv2.imwrite(img_saving_path, img)
     
def convolve2d(img, kernel):
    """Convolves a given image and a given kernel.

    Steps:
        (1) flips the kernel
        (2) pads the image # IMPORTANT
            this step handles pixels along the border of the image, and ensures that the output image is of the same size as the input image
        (3) calucates the convolved image using nested for loop

    Args:
        img: nested list (int), image.
        kernel: nested list (int), kernel.

    Returns:
        img_conv: nested list (int), convolved image.
    """
    # TODO: implement this function.
    flipped_kernel=utils.flip2d(kernel)
    #number of row and cols padding will, be equal to length of rows and columns minus 1 respectively
    padding_rows=len(kernel)-1
    padding_cols=len(kernel[0])-1
    padded_image=utils.zero_pad(img,padding_rows,padding_cols)
    #copied the padded image to a temporary variable to modify the convolution changes
    temp_array=copy.deepcopy(padded_image)
    for i in range(len(padded_image)-1):
        for j in range(len(padded_image[0])-1):
            #created patches of each matrices which overlap with kernel
            xmin=i
            xmax=i+len(flipped_kernel)
            ymin=j
            ymax=j+len(flipped_kernel[0])
            patch = padded_image[xmin: xmax]
            patch = [row[ymin: ymax] for row in patch]
            #multiplied each element of patch with corresponding element of flipped kernel and calculated the end summation
            multiplied=utils.elementwise_mul(patch,flipped_kernel)
            summation=0
            for m in range(len(multiplied)):
                for n in range(len(multiplied[0])):
                    summation+=multiplied[m][n]
            temp_array[i][j]=summation
    #set min and max x and y for croping the padded image
    min_x=padding_rows-1
    max_x=len(padded_image)-(padding_rows+1)
    min_y=padding_cols-1
    max_y=len(padded_image[0])-(padding_cols+1)
    temp_array=utils.crop(temp_array,min_x,max_x,min_y,max_y)
    return temp_array
    #raise NotImplementedError

def main():
    args = parse_args()

    img = read_image(args.img_path)

    if args.filter == "low-pass":
        kernel = low_pass
    elif args.filter == "high-pass":
        kernel = high_pass
    else:
        raise ValueError("Filter type not recognized.")

    if not os.path.exists(args.rs_dir):
        os.makedirs(args.rs_dir)

    filtered_img = convolve2d(img, kernel)
    write_image(filtered_img, os.path.join(args.rs_dir, "{}.jpg".format(args.filter)))


if __name__ == "__main__":
    main()
