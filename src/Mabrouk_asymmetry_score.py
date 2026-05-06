import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.transform import rotate

def Mabrouk_find_axis(mask):
    """Finds the axis at which the boundary is closest to the centroid,
    as done in the 2020 Mabrouk method of finding asymmetry score"""

    #Find the centroid of our mask
    props = regionprops(mask.astype(int))[0]
    Yc, Xc = props.centroid

    #Extract boundary pixels
    boundary = find_boundaries(mask, mode='outer')
    ys, xs = np.where(boundary)

    #vector from centroid to each boundary
    dy = ys - Yc
    dx = xs - Xc

    #angles and distance of each boundary pixel
    angles = np.arctan2(dy, dx)
    angles_deg = (np.degrees(angles) +360) % 360

    distance = np.sqrt(dx*dx + dy*dy)

    #binning by angle and keeping the closest boundary
    bins = np.arange(360)
    min_dist = np.full(360, np.inf)

    for a, d in zip(angles_deg.astype(int), distance):
        if d < min_dist[a]:
            min_dist[a] = d
    
    #Extracting the axis with the closest boundary
    closest_axes = np.where(min_dist == np.min(min_dist))[0]

    return closest_axes

def reflect_mask(mask, angle_deg):
    """Reflects the mask across an axis passing through the centroid at angle_deg"""

    #Rotate so axis becomes vertical
    rot = rotate(mask, -angle_deg, order=0, preserve_range=True)
    rot = rot >0.5

    #reflect horizontally
    refl = np.fliplr(rot)

    #rotate back
    refl_back = rotate(refl, angle_deg, order=0, preserve_range=True)
    refl_back = refl_back > 0.5

    return refl_back

def delta_A(mask, reflected):
    """Computes the non-overlapping area"""

    xor = np.logical_xor(mask, reflected)
    return xor.sum()

def Mabrouk_asymmetry(image, mask):
    """Computes the asymmetry score using the method from the 2020 Mabrouk research paper on skin lesion diagnosis"""

    #Find axis closest to centroid
    axis1 = int(Mabrouk_find_axis(mask)[0])
    axis2 = (axis1 + 90) % 360

    #Reflect across both axes
    refl1 = reflect_mask(mask, axis1)
    refl2 = reflect_mask(mask, axis2)

    #compute delta A for each axis
    A = mask.sum()
    dA1 = delta_A(mask, refl1)
    dA2 = delta_A(mask, refl2)

    #Compute asymmetry index for each axis
    AI1 = dA1 / A
    AI2 = dA2 / A

    #Final asymmetry score (0, 1, or 2)
    score = int(AI1 > 1) + int(AI2 > 1)

    return {
        "mabrouk_asymmetry_score": score
    }
