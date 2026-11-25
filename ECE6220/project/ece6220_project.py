# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:09:48 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
import time

def main():
    im_spillway = imageio.v2.imread("spillway-dark.tif")
    # im_alley = imageio.v2.imread("header-greyscale.png")
    im_crowd = imageio.v2.imread("crowd.tif")
    im_lax = imageio.v2.imread("lax.tif")
    im_peppers = imageio.v2.imread("peppers.tif")
    im_plane = imageio.v2.imread("plane.tif")
    
    
    # test all images
    run_it(im_spillway, "Spillway")
    # run_it(im_alley,"Alley")
    run_it(im_crowd, "Crowd")
    run_it(im_lax, "LAX")
    run_it(im_peppers, "Peppers")
    run_it(im_plane, "Plane")
    
    
    
def run_it(im, name):
    print(f"\nRunning all HE techniques for {name}")
    
    #plot original image
    plt.figure()
    plt.imshow(im, cmap="gray")
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    # GHE work
    start = time.perf_counter()
    eq_im = image_equal(im)
    end=time.perf_counter()
    elapsed_time = end-start
    print(f"Execution time of GHE: {elapsed_time:.6f} seconds")
    plot_help(im, eq_im, "Global HE")
    
    #AHE work
    start = time.perf_counter()
    ahe_im = adaptive_histogram(im, 2)
    end=time.perf_counter()
    elapsed_time = end-start
    print(f"Execution time of AHE: {elapsed_time:.6f} seconds")
    plot_help(im, ahe_im, "Adaptive HE")
    
    # AHE with interpolation
    start = time.perf_counter()
    eq_im_ahe_inter = adaptive_histogram_interpolated(im, 2)
    end=time.perf_counter()
    elapsed_time = end-start
    print(f"Execution time of AHE Interpolate: {elapsed_time:.6f} seconds")

    plot_help(im, eq_im_ahe_inter, "AHE with Interpolation")
    
    # CLAHE work
    start = time.perf_counter()
    eq_im_clahe = contrast_limited_ahe(
        im,
        2,
        2.0
    )
    end=time.perf_counter()
    elapsed_time = end-start
    print(f"Execution time of CLAHE: {elapsed_time:.6f} seconds")
    
    plot_help(im, eq_im_clahe, "CLAHE")
    
    # BBHE work
    start = time.perf_counter()
    eq_im_bbhe = brightness_preserving_bhe(im)    
    end=time.perf_counter()
    elapsed_time = end-start
    print(f"Execution time of BBHE: {elapsed_time:.6f} seconds")
    
    plot_help(im, eq_im_bbhe, "BBHE")
    
    # DSHE work
    start = time.perf_counter()
    eq_im_dshe = dualistic_subimage_he(im)
    end=time.perf_counter()
    elapsed_time = end-start
    print(f"Execution time of DSIHE: {elapsed_time:.6f} seconds")
    plot_help(im, eq_im_dshe, "DSIHE")
    
    print(f"Comleted all HE techniques for {name}\n")


def dualistic_subimage_he(im):
    # calculate median intensity
    median_intensity = np.median(im)

    # divide image into lower and upper portions
    lower_half = im <= median_intensity
    upper_half = im > median_intensity

    # initialize output
    im_out = np.zeros_like(im)

    # create lower and upper portions
    lower = im[lower_half]
    upper = im[upper_half]

    # equalize lower portion
    if lower.size > 0:
        hist_lower = image_histogram(lower)
        cdf_lower = np.cumsum(hist_lower)
        cdf_min_lower = cdf_lower[np.flatnonzero(hist_lower)[0]]

        if lower.size == np.count_nonzero(lower == lower[0]):
            eq_lower = lower.copy()
        else:
            eq_lower = np.round(
                (cdf_lower[lower] - cdf_min_lower) /
                (1 - cdf_min_lower) * median_intensity
            ).astype(int)

        im_out[lower_half] = eq_lower

    # equalize upper portion
    if upper.size > 0:
        hist_upper = image_histogram(upper)
        cdf_upper = np.cumsum(hist_upper)
        cdf_min_upper = cdf_upper[np.flatnonzero(hist_upper)[0]]

        if upper.size == np.count_nonzero(upper == upper[0]):
            eq_upper = upper.copy()
        else:
            eq_upper = np.round(
                (cdf_upper[upper] - cdf_min_upper) /
                (1 - cdf_min_upper) *
                (255 - median_intensity - 1)
                + median_intensity + 1
            ).astype(int)

        im_out[upper_half] = eq_upper

    # calculate equalized histograms
    if lower.size > 0:
        hist_lower_eq = image_histogram(eq_lower)

    if upper.size > 0:
        hist_upper_eq = image_histogram(eq_upper)

    # plot histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(hist_lower)
    plt.title("Original Lower Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(hist_lower_eq)
    plt.title("Equalized Lower Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(hist_upper)
    plt.title("Original Upper Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(hist_upper_eq)
    plt.title("Equalized Upper Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.tight_layout()
    plt.show()

    return im_out
 


def brightness_preserving_bhe(im):
    # calculate mean intensity
    mean_intensity = np.floor(np.mean(im)).astype(int)

    # divide image into lower and upper portions
    lower_half = im <= mean_intensity
    upper_half = im > mean_intensity

    # initialize output
    im_out = np.zeros_like(im)

    # create lower and upper portions
    lower = im[lower_half]
    upper = im[upper_half]

    # equalize lower portion
    if lower.size > 0:
        hist_lower = image_histogram(lower)
        cdf_lower = np.cumsum(hist_lower)
        cdf_min_lower = cdf_lower[np.flatnonzero(hist_lower)[0]]

        if lower.size == np.count_nonzero(lower == lower[0]):
            eq_lower = lower.copy()
        else:
            eq_lower = np.round(
                (cdf_lower[lower] - cdf_min_lower) /
                (1 - cdf_min_lower) * mean_intensity
            )

        im_out[lower_half] = eq_lower

    # equalize upper portion
    if upper.size > 0:
        hist_upper = image_histogram(upper)
        cdf_upper = np.cumsum(hist_upper)
        cdf_min_upper = cdf_upper[np.flatnonzero(hist_upper)[0]]

        if upper.size == np.count_nonzero(upper == upper[0]):
            eq_upper = upper.copy()
        else:
            eq_upper = np.round(
                (cdf_upper[upper] - cdf_min_upper) /
                (1 - cdf_min_upper) *
                (255 - mean_intensity - 1)
                + mean_intensity + 1
            )

        im_out[upper_half] = eq_upper

    # calculate equalized histograms
    if lower.size > 0:
        hist_lower_eq = image_histogram(eq_lower)

    if upper.size > 0:
        hist_upper_eq = image_histogram(eq_upper)

    # plot histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(hist_lower)
    plt.title("Original Lower Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(hist_lower_eq)
    plt.title("Equalized Lower Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(hist_upper)
    plt.title("Original Upper Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(hist_upper_eq)
    plt.title("Equalized Upper Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.grid()

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 5))

    plt.plot(hist_lower, label="Original Lower", color="blue")
    plt.plot(hist_upper, label="Original Upper", color="blue")
    plt.plot(hist_lower_eq, label="Equalized Lower",color="red")
    plt.plot(hist_upper_eq, label="Equalized Upper",color="red")
    
    plt.xlabel("Intensity")
    plt.ylabel("Normalized Frequency")
    plt.title("BBHE Histogram Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    return im_out


    

def contrast_limited_ahe(im, divisions=1, clip_limit=2.0):

    # get image dimensions
    rows, cols = im.shape
    nbins = 256

    # determine number of tiles
    n_tiles = 2**divisions

    if n_tiles > min(rows, cols):
        raise ValueError("Too many divisions for image dimensions")

    # determine tile boundaries
    row_edges = np.linspace(0, rows, n_tiles + 1, dtype=int)
    col_edges = np.linspace(0, cols, n_tiles + 1, dtype=int)

    # create lookup table for each tile
    luts = np.zeros((n_tiles, n_tiles, nbins), dtype=np.uint8)

    for i in range(n_tiles):
        for j in range(n_tiles):

            # extract tile
            tile = im[row_edges[i]:row_edges[i+1],
                      col_edges[j]:col_edges[j+1]]

            # calculate histogram
            hist = np.bincount(tile.ravel(), minlength=nbins)

            # calculate actual clipping threshold
            threshold = max(1, int(clip_limit * tile.size / nbins))

            # clip and redistribute histogram
            hist = clip_histogram(hist, threshold)

            # calculate CDF
            cdf = np.cumsum(hist)

            # calculate equalization lookup table
            nonzero = np.flatnonzero(hist)
            cdf_min = cdf[nonzero[0]]

            if tile.size == cdf_min:
                lut = np.arange(nbins)
            else:
                lut = np.round(
                    (cdf - cdf_min) /
                    (tile.size - cdf_min) * (nbins - 1)
                )

            luts[i, j] = np.clip(lut, 0, 255).astype(np.uint8)

    # interpolate between neighboring tiles
    row_centers = (row_edges[:-1] + row_edges[1:]) / 2
    col_centers = (col_edges[:-1] + col_edges[1:]) / 2

    y = np.arange(rows)
    x = np.arange(cols)

    # find neighboring tile indices
    y1 = np.clip(np.searchsorted(row_centers, y), 0, n_tiles-1)
    x1 = np.clip(np.searchsorted(col_centers, x), 0, n_tiles-1)

    y0 = np.maximum(y1 - 1, 0)
    x0 = np.maximum(x1 - 1, 0)

    # calculate interpolation weights
    wy = np.divide(
        y - row_centers[y0],
        row_centers[y1] - row_centers[y0],
        out=np.zeros(rows, dtype=float),
        where=(y1 != y0)
    )

    wx = np.divide(
        x - col_centers[x0],
        col_centers[x1] - col_centers[x0],
        out=np.zeros(cols, dtype=float),
        where=(x1 != x0)
    )

    wy = np.clip(wy, 0, 1)
    wx = np.clip(wx, 0, 1)

    # get equalized values from neighboring tiles
    im_float = im.astype(np.uint8)

    v00 = luts[y0[:, None], x0[None, :], im_float]
    v01 = luts[y0[:, None], x1[None, :], im_float]
    v10 = luts[y1[:, None], x0[None, :], im_float]
    v11 = luts[y1[:, None], x1[None, :], im_float]

    # bilinear interpolation
    eq_im = (
        v00 * (1-wy[:, None]) * (1-wx[None, :]) +
        v01 * (1-wy[:, None]) * wx[None, :] +
        v10 * wy[:, None] * (1-wx[None, :]) +
        v11 * wy[:, None] * wx[None, :]
    )

    return np.clip(np.round(eq_im), 0, 255).astype(np.uint8)

    

def clip_histogram(hist, clip_limit):

    # calculate total excess pixels
    excess = np.sum(np.maximum(hist - clip_limit, 0))

    # clip histogram
    hist = np.minimum(hist, clip_limit)

    # redistribute excess pixels evenly
    nbins = len(hist)
    batch = excess // nbins
    residual = excess % nbins

    hist += batch

    # distribute remaining pixels
    if residual > 0:
        step = max(nbins // residual, 1)
        indices = np.arange(0, nbins, step)[:residual]
        hist[indices] += 1

    return hist



def adaptive_histogram_interpolated(im, division_num=1):
    M, N = im.shape

    # Determine number of tiles
    n_tiles = 2**division_num

    # Determine tile boundaries
    row_bounds = np.linspace(0, M, n_tiles + 1, dtype=int)
    col_bounds = np.linspace(0, N, n_tiles + 1, dtype=int)

    # Store the transformation lookup table for each tile
    transformations = np.zeros((n_tiles, n_tiles, 256))

    # Calculate the HE transformation for every tile
    for i in range(n_tiles):
        for j in range(n_tiles):
            tile = im[row_bounds[i]:row_bounds[i+1],
                      col_bounds[j]:col_bounds[j+1]]

            hist = image_histogram(tile)
            cdf = np.cumsum(hist)

            # Find first nonzero histogram bin
            cdf_min = cdf[np.flatnonzero(hist)[0]]

            # Calculate transformation
            if np.all(tile == tile.flat[0]):
                transformations[i, j] = np.arange(256)
            else:
                transformations[i, j] = np.round(
                    (cdf - cdf_min) / (1 - cdf_min) * 255
                )

    # Initialize output image
    im_out = np.zeros_like(im, dtype=np.uint8)

    # Calculate the center of each tile
    row_centers = (row_bounds[:-1] + row_bounds[1:] - 1) / 2
    col_centers = (col_bounds[:-1] + col_bounds[1:] - 1) / 2

    # Process each pixel
    for y in range(M):
        for x in range(N):

            # Find the two neighboring tile centers vertically
            r1 = np.searchsorted(row_centers, y, side='right') - 1
            r1 = np.clip(r1, 0, n_tiles - 1)
            r2 = min(r1 + 1, n_tiles - 1)

            # Find the two neighboring tile centers horizontally
            c1 = np.searchsorted(col_centers, x, side='right') - 1
            c1 = np.clip(c1, 0, n_tiles - 1)
            c2 = min(c1 + 1, n_tiles - 1)

            # Calculate interpolation weights
            if r1 == r2:
                wy = 0
            else:
                wy = (y - row_centers[r1]) / (
                    row_centers[r2] - row_centers[r1]
                )

            if c1 == c2:
                wx = 0
            else:
                wx = (x - col_centers[c1]) / (
                    col_centers[c2] - col_centers[c1]
                )

            # Keep weights within [0, 1]
            wy = np.clip(wy, 0, 1)
            wx = np.clip(wx, 0, 1)

            # Original pixel intensity
            pixel = im[y, x]

            # Apply the four neighboring transformations
            T11 = transformations[r1, c1, pixel]
            T12 = transformations[r1, c2, pixel]
            T21 = transformations[r2, c1, pixel]
            T22 = transformations[r2, c2, pixel]

            # Bilinear interpolation
            top = (1 - wx) * T11 + wx * T12
            bottom = (1 - wx) * T21 + wx * T22

            result = (1 - wy) * top + wy * bottom

            im_out[y, x] = np.clip(np.round(result), 0, 255)

    return im_out


def adaptive_histogram(im, division_num=1):
    # divide image into tiles
    M,N = im.shape
    # how many tiles will we have
    n_tiles = 2**division_num
    # determine tile size
    row_size = M // n_tiles
    col_size = N // n_tiles
    # initialize new image
    im_out = np.zeros_like(im)
    # iterate through the tiles
    for i in range(n_tiles):
        for j in range(n_tiles):
            # extract tile ij from the image
            tile = im[i*row_size:(i+1)*row_size,
                      j*col_size:(j+1)*col_size]
            # apply HE
            eq_tile = image_equal(tile)
            # insert eq_tile into new image
            im_out[i*row_size:(i+1)*row_size,
                      j*col_size:(j+1)*col_size] = eq_tile
    
    return im_out

def image_histogram(im):
    flat_im = im.flatten().astype(int)
    # create array to store pixel values
    values = np.zeros(256)
    # inspect each pixel value in the image
    for pixel in flat_im:
        # increment the number of pixels with that value
        values[pixel] += 1

    #normalize
    values = values / im.size
    return values

def image_equal(im):
    # step 1: Compute histogram
    # step 2: normalize
    hist = image_histogram(im)
    # step 3: computer CDF T(k)
    cdf = np.cumsum(hist)
    # step 4: scale s = round(T(k) * (L - 1))
    s = (cdf * 255).astype(int)  # 255 = L - 1 where L = 256
    # step 5 : map
    return s[im]

def plot_help(orig_im, eq_im, hist_type):
    # calculate histograms
    orig_hist = image_histogram(orig_im)
    eq_hist = image_histogram(eq_im)
    
    # plot histogram
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(orig_hist)
    plt.title("Original Image Histogram")
    plt.xlabel("pixel intensity")
    plt.ylabel("frequency")
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(eq_hist)
    plt.title(f"{hist_type} histogram")
    plt.xlabel("pixel intensity")
    plt.ylabel("frequency")
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # show images
    # plt.figure()
    # plt.imshow(orig_im, cmap="gray")
    # plt.title('Original Image')
    # plt.axis('off')
    # plt.show()
    
    plt.imshow(eq_im, cmap="gray")
    plt.title(f'{hist_type} Image')
    plt.axis('off')
    plt.show()
    
    return

if __name__ == "__main__":
    main()