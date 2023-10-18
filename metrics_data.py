from math import log10,sqrt
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import jaccard_score
import numpy as np
import pandas as pd
import regex as re


class metrics():
    def __init__(self, data1,data2) :
        """
        
        self.array1  : (Arr) Training Data (Single data image)
        
        self.array2  : (Arr) Testing data (Single Data Image)   
        
        """
        self.array1=data1
        self.array2=data2
    
    def MSE(self):
        height,width=self.array1.shape[:2]
        err=np.sum(np.square(np.subtract(self.array1,self.array2)))
        mse=err/(float(height*width))
        return round(mse,3)
                    
    
    def PSNR(self):
        MAX_PIXEL=255
        if self.MSE()==0:
            return np.inf
        psnr_value=(20*log10(MAX_PIXEL))-10*log10(self.MSE())
        return round(psnr_value,3)
    
    def similarity(self,channel_axis,bool_full):
        """
        self.array1  : (Arr)Training Data (Single data image)
        
        self.array2  : (Arr)Testing data (Single Data Image)
        
        channel_axis : (int) If None, the image is assumed to be a grayscale (single channel) image. Otherwise, this parameter indicates which axis of the array corresponds to channels.
        
        full         : (Boolean Value) This map provides information about how SSIM varies across different regions of the images, highlighting areas of higher or lower similarity
                        a. True  : The function will return a full SSIM map, where each pixel in the map represents the SSIM score calculated for a local neighborhood around that pixel. This map provides information about how the structural similarity varies across different regions of the images.
                        b. False : The will return a single SSIM score that represents the overall structural similarity between the two images.
        """
        ssims_value=compare_ssim(self.array1,
                                 self.array2,
                                 channel_axis=channel_axis,
                                 full=bool_full)
        
        return ssims_value
        
    def accuracy(self):
        total_pixels=np.prod(self.array1.shape)
        matching_image=np.sum(self.array1==self.array2)
        val_accuracy=matching_image/total_pixels
        return round(val_accuracy,3)
    
    def IoU(self):
        jac=jaccard_score (self.array1,self.array2)
        return jac
        
        
        
        