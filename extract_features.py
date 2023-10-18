import numpy as np
import tensorflow as tf
import cv2 
from skimage.feature import hog


class extract_image:
    
    def __init__(self):
        pass
    """
    def SIFT(image,
             hessian_val=800):
        SURF(SURF)
    """
        
        
    
    def HOG (image,resize=(64,128),
             orientation=8,
             pixels_per_cell=(16,16),
             cells_per_block=(2,2),
             visualize=False,
             channel_axis=None,
             feature_vector=False
             ):
        
        """
        image           : Input array image (M,N,[C])
        orientations    : int (Number of orientations)
        pixels_per_cell : (int,int)-tuple (Size (in pixels) within a cell)
        cells_per_block : (int,int)-tuple (Number of cells in each block)
        block-norm      : str ['L1','L1-sqrt','L2','L2-Hys'], (JUST ONE) Block normlalization using L1-norm
                         
                         L1     : Normalization L1-norm
                         L1-sqrt: Normalization using L1-norm, followed by square root
                         L2     : Normalization using L2-norm
                         L2-hys : Normalization using L2-norm which is followed by limiting the maximum 
                                values to 0.2(Hys stands for hysteresis) and renormalization using L2-norm.
        visualize       : Bool value (Return an image of the HOG)
        transform_sqrt : Bool value (Normalization of the image before the processing of the HOG.
                                     DO NOT use for each negative values)
        channel_axis    : int or None (If None is grayscale, otherweise RGB image) 
        feature_vector  : Bool Value (return a vector features.)
        
        """
        resize=cv2.resize(image,resize)
        fd,hog_image=hog(image=resize,
                         orientations=orientation,
                         pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block,
                         visualize=visualize,
                         channel_axis=channel_axis,
                         feature_vector=feature_vector)
        return hog_image
        
    def ColorHist(image,mask=None,histSize=[256],ranges=[0,256],accumulate=False):
        
        """
        Extracting the composition of the image color
        """
        
        red=cv2.calcHist([image],
                         [0],
                         mask=mask,
                         histSize=histSize,
                         ranges=ranges,
                         accumulate=accumulate)
        green=cv2.calcHist([image],
                         [1],
                         mask=mask,
                         histSize=histSize,
                         ranges=ranges,
                         accumulate=accumulate)
        blue=cv2.calcHist([image],
                         [2],
                         mask=mask,
                         histSize=histSize,
                         ranges=ranges,
                         accumulate=accumulate)
        return red,green,blue