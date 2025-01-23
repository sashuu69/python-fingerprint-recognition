import numpy as np

def normalise(img, mean, std):
    normed = (img - np.mean(img)) / (np.std(img))
    return normed

def ridge_segment(im, blksze, thresh):
    rows, cols = im.shape
    
    # Normalise the image to have zero mean and unit standard deviation
    im = normalise(im, 0, 1)
    
    # Determine new dimensions that are multiples of `blksze`
    new_rows = int(blksze * np.ceil(float(rows) / blksze))
    new_cols = int(blksze * np.ceil(float(cols) / blksze))
    
    # Create padded images
    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))
    
    # Copy original image into padded image
    padded_img[:rows, :cols] = im
    
    # Compute standard deviation for each block
    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i:i + blksze, j:j + blksze]
            stddevim[i:i + blksze, j:j + blksze] = np.std(block) * np.ones(block.shape)
    
    # Trim padded stddevim to original image size
    stddevim = stddevim[:rows, :cols]
    
    # Create mask based on threshold
    mask = stddevim > thresh
    
    # Calculate mean and standard deviation for ridge regions
    mean_val = np.mean(im[mask])
    std_val = np.std(im[mask])
    
    # Normalise the image using the mask
    normim = (im - mean_val) / std_val
    
    return normim, mask
