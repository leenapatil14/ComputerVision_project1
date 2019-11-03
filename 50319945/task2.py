"""
Template Matching
(Due date: Sep. 25, 3 P.M., 2019)

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).

Please complete all the functions that are labelled with '# TODO'. When implementing those functions, comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in 'utils.py'
and the functions you implement in 'task1.py' are of great help.

Do NOT modify the code provided to you.
Do NOT use ANY API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import ANY library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    sum_Template=0
    template_rows=len(template)
    template_cols=len(template[0])
    patch_rows=len(template)
    patch_cols=len(template[0])
    
    #calculate mean value of template matrix
    for t1, row in enumerate(template):
        for t2, current_val_t in enumerate(row):
            sum_Template+=current_val_t
    mean_Template=sum_Template/(template_rows*template_cols)
    sum_patch=0
    
    #calculate mean value of patch matrix
    for p1, row in enumerate(patch):
        for p2, current_val_p in enumerate(row):
            sum_patch+=current_val_p
    mean_patch=sum_patch/(patch_rows*patch_cols)    
    
    #calculate NCC from patch and template matrices
    sum_temp=0
    temp_sqr=0
    patch_sqr=0
    for m, rowt in enumerate(template):
        for n, template_value in enumerate(rowt):
            target_patch_diff=patch[m][n]-mean_patch
            target_template_diff=template_value-mean_Template
            sum_temp+=(target_template_diff)*(target_patch_diff)
            temp_sqr+=(target_template_diff)**2
            patch_sqr+=(target_patch_diff)**2
    for_sqrt=temp_sqr*patch_sqr      
    res=sum_temp/for_sqrt**0.5
            
    return res
            
    #raise NotImplementedError

def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    # TODO: implement this function.
    patching_rows=len(img)-len(template)+1
    patching_cols=len(img[0])-len(template[0])+1
    
    max_value=0
    x=0
    y=0
    for i in range(patching_rows):
        for j in range(patching_cols):
            xmin=i
            xmax=i+len(template)
            ymin=j
            ymax=j+len(template[0])
            patch = utils.crop(img,xmin,xmax,ymin,ymax) 
            current_temp=norm_xcorr2d(patch,template)
            if(max_value<current_temp):
                max_value=current_temp
                x=i
                y=j
           
    return x,y,max_value
           
    
    #raise NotImplementedError

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)
    
    x, y, max_value = match(img, template)
     # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
