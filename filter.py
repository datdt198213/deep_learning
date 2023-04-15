import numpy as np 
from scipy import ndimage 
from typing import List 

def apply_gaussian_filter(signal: List[float], sigma: float) -> List[float]: 
    # Convert the list to a NumPy array 
    arr = np.array(signal) 
     # Apply the Gaussian filter using ndimage 
    filtered = ndimage.gaussian_filter(arr, sigma=sigma) 
     # Convert the filtered array back to a list and return it 
    return filtered.tolist()