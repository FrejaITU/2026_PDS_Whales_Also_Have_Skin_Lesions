import numpy as np
from skimage.measure import perimeter

def get_compactness(image, mask):
    """Compute Polsby-Popper compactness score."""
    
    area = np.sum(mask)
    perim = perimeter(mask)
    
    if perim == 0:
        return {"Polsby-Popper": 0}
    
    compact = (4 * np.pi * area) / (perim ** 2)
    
    return {
        "Polsby-Popper": compact
}