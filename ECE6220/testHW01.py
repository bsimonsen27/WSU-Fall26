# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:49:07 2026

@author: brend
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Create a simple 5x5 test image
    im = np.array([
        [1,  2,  3,  4,  5],
        [6,  7,  8,  9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ])
    
    # Translate the image 1 pixel right and 1 pixel down
    translated = image_translate(im, 1, 1)
    
    print("Original image:")
    print(im)
    
    print("\nTranslated image:")
    print(translated)
    
    # Scale by 2 in both directions
    #scaled = image_scaling(im, 2, 2)
    scaled = image_scaling(im, 0.5, 0.5)
    
    print("Original:")
    print(im)
    
    print("\nScaled:")
    print(scaled)

    rotated = image_rotation(im, 90)
    
    print("Original:")
    print(im)
    
    print("\nRotated 90 degrees:")
    print(rotated)
    
    sv = 1

    vertical = image_shear_vertical(im, sv)
    
    print("\nVertical shear (sv = 1):")
    print(vertical)
    
    sh = 1

    horizontal = image_shear_horizontal(im, sh)
    
    print("\nHorizontal shear (sh = 1):")
    print(horizontal)
    
def image_shear_vertical(im, sv):
    """Shear an image vertically by sv.

    Sets the values of pixels that are translated out of the image bounds to 0.

    Parameters
    ----------
    im : ndarray
        2D array representing the image, pixel values from 0 to 255.
    sv : float
        Shear factor.

    Returns
    -------
    ndarray
        2D array representing the sheared image, pixel values from 0 to 255.
    """

    # YOUR CODE HERE
    # vertical shear array
    S = np.array([
        [1,0,0],
        [sv,1,0],
        [0,0,1]
    ])
    # inverse array
    S_inv = np.linalg.inv(S)
    # create new image 
    new_img = np.zeros_like(im)
    #image dimensions
    width = im.shape[1]
    height = im.shape[0]
    # loop through rows
    for y_prime in range(height):
        # loop through columns
        for x_prime in range(width):
            # output pixel coordinate
            new_pix = np.array([x_prime,y_prime,1])
            # matrix multiplication
            old_pix = S_inv @ new_pix
            # nearest neighbor interpolation
            x = int(round(old_pix[0]))
            y = int(round(old_pix[1]))
            # check if in bounds of image
            if (0 <= x < width) and (0 <= y < height):
                # copy new pixel
                new_img[y_prime,x_prime] = im[y,x]
    return new_img
    
def image_shear_horizontal(im, sh):
    """Shear an image horizontally by sh.

    Sets the values of pixels that are translated out of the image bounds to 0.

    Parameters
    ----------
    im : ndarray
        2D array representing the image, pixel values from 0 to 255.
    sh : float
        Shear factor.

    Returns
    -------
    ndarray
        2D array representing the sheared image, pixel values from 0 to 255.
    """

    # YOUR CODE HERE
    # horizontal shear array
    S = np.array([
        [1,sh,0],
        [0,1,0],
        [0,0,1]
    ])
    # inverse array
    S_inv = np.linalg.inv(S)
    # create new image 
    new_img = np.zeros_like(im)
    #image dimensions
    width = im.shape[1]
    height = im.shape[0]
    # loop through rows
    for y_prime in range(height):
        # loop through columns
        for x_prime in range(width):
            # output pixel coordinate
            new_pix = np.array([x_prime,y_prime,1])
            # matrix multiplication
            old_pix = S_inv @ new_pix
            # nearest neighbor interpolation
            x = int(round(old_pix[0]))
            y = int(round(old_pix[1]))
            # check if in bounds of image
            if (0 <= x < width) and (0 <= y < height):
                # copy new pixel
                new_img[y_prime,x_prime] = im[y,x]
    return new_img
    
def image_rotation(im, theta):
    """Rotate an image by angle degrees.

    Sets the values of pixels that are translated out of the image bounds to 0.

    Parameters
    ----------
    im : ndarray
        2D array representing the image, pixel values from 0 to 255.
    angle : float
        Angle in degrees.

    Returns
    -------
    ndarray
        2D array representing the rotated image, pixel values from 0 to 255.
    """

    # YOUR CODE HERE
    height = im.shape[0]
    width = im.shape[1]
    #image center
    center_x = (width-1)/2
    center_y = (height-1)/2
    # convert to radians
    theta = np.deg2rad(theta)
    # move center to origin
    T1 = np.array([
        [1,0,-center_x],
        [0,1,-center_y],
        [0,0,1]
    ])
    # rotate matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0,0,1]
    ])
    #move origin back to image center
    T2 = np.array([
        [1,0,center_x],
        [0,1,center_y],
        [0,0,1]
    ])
    # make a single transform matrix
    T = T2 @ R @ T1
    # invert T
    T_inv = np.linalg.inv(T)
    # create a new image
    new_img = np.zeros_like(im)
    # loop through the rows
    for y_prime in range(height):
        # loop through the columns
        for x_prime in range(width):
            # output pixel coordinates
            new_pix = np.array([x_prime, y_prime, 1])
            # find pixel from original image
            old_pix = T_inv @ new_pix
            # nearest-neighbor interpolation
            x = int(round(old_pix[0]))
            y = int(round(old_pix[1]))
            # check if in bounds of image
            if (0 <= x < width) and (0 <= y < height):
                # copy pixel to new image
                new_img[y_prime,x_prime] = im[y,x]
    return new_img

def image_scaling(im, cx, cy):
    """Scale an image by cx in the x-direction and cy in the y-direction.

    Sets the values of pixels that are translated out of the image bounds to 0.

    Parameters
    ----------
    im : ndarray
        2D array representing the image, pixel values from 0 to 255.
    cx : float
        x-scaling factor.
    cy : int
        y-scaling factor.

    Returns
    -------
    ndarray
        2D array representing the scaled image, pixel values from 0 to 255.
    """

    # YOUR CODE HERE
    S = np.array([
        [cx,0,0],
        [0,cy,0],
        [0,0,1]
    ])
    # invert matrix
    S_inv = np.linalg.inv(S)
    new_img = np.zeros_like(im)  # create an empty image
    height = im.shape[0]
    width = im.shape[1]

    # loop through the rows
    for y_prime in range(height):
        # loop through the columns
        for x_prime in range(width):
            new_pix = np.array([x_prime, y_prime, 1])
            old_pix = S_inv @ new_pix    # matrix multiplication
            x = int(round(old_pix[0]))
            y = int(round(old_pix[1]))
            # check if pixels are within bounds of image
            if (0 <= x < width) and (0 <= y < height):
                # copy pixels to the new image
                new_img[y_prime, x_prime] = im[y,x]
    return new_img

def image_translate(im, tx, ty):
    """Translate an image by tx pixels in the x-direction and ty pixels in the y-direction.

    Sets the values of pixels that are translated out of the image bounds to 0.

    Parameters
    ----------
    im : ndarray
        2D array representing the image, pixel values from 0 to 255.
    tx : int
        x-translation in pixels.
    ty : int
        y-translation in pixels.

    Returns
    -------
    ndarray
        2D array representing the translated image, pixel values from 0 to 255.
    """

    # YOUR CODE HERE
    T = np.array([
        [1,0,tx],
        [0,1,ty],
        [0,0,1]
    ])
    T_inv = np.linalg.inv(T)     # invert the matrix
    new_img = np.zeros_like(im)  # create an empty image
    height = im.shape[0]
    width = im.shape[1]

    # loop through the rows
    for y_prime in range(height):
        # loop through the columns
        for x_prime in range(width):
            new_pix = np.array([x_prime, y_prime, 1])
            old_pix = T_inv @ new_pix    # matrix multiplication
            x = int(round(old_pix[0]))
            y = int(round(old_pix[1]))
            # check if pixels are within bounds of image
            if (0 <= x < width) and (0 <= y < height):
                # copy pixels to the new image
                new_img[y_prime, x_prime] = im[y,x]
    return new_img

if __name__ == "__main__":
    main()