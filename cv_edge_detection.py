import cv2
import numpy as np
import pandas as pd
import metrics_data as met
import matplotlib.pyplot as plt
from itertools import chain


class edge_detection():
    
    
    
    def __init__(self,origin,percentage,size):
        """
		origin                       : (Array) Consisting of image
        percentage                   : (Float) Size of training and testing data
		N                            : (Int) Looping system for finding the best score from range 0 to N-1
		"""
        self.image=origin
        self.p=percentage
        self.N=size
        self.kirsch_kernel_north = np.array([[-3, -3, -3],
                                [-3,  0, -3],
                                [ 5,  5,  5]], dtype=np.float32)
        self.kirsch_kernel_northeast = np.array([[-3, -3,  5],
                                    [-3,  0,  5],
                                    [-3, -3,  5]], dtype=np.float32)
                 
        self.kirsch_kernel_east=np.array([[-3,  5,  5],
                                 [-3,  0,  5],
                                 [-3, -3, -3]], dtype=np.int32)
        
        self.kirsch_kernel_southeast = np.array([[ 5,  5,  5],
                                      [-3,  0, -3],
                                      [-3, -3, -3]], dtype=np.int32)
        
        self.kirsch_kernel_south = np.array([[ 5,  5,  5],
                                      [-3,  0, -3],
                                      [-3, -3, -3]], dtype=np.int32) 
        self.kirsch_kernel_southwest = np.array([[-3,  5, -3],
                                    [-3,  0, -3],
                                    [ 5,  5, -3]], dtype=np.int32)
        
        self.kirsch_kernel_west = np.array([[-3, -3, -3],
                                 [ 5,  0, -3],
                                 [ 5,  5, -3]], dtype=np.int32)
        
        self.kirsch_kernel_northwest = np.array([[-3, -3, -3],
                                      [-3,  0, -3],
                                      [ 5,  5,  5]], dtype=np.int32)
        
        self.prewitt_horizontal=np.array([[-1, 0, 1],
                                          [-1, 0, 1],
                                          [-1, 0, 1]], dtype=np.float32)
        
        self.prewitt_vertical =np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]], dtype=np.float32) 
        
    
        
    def canny_cv(self, threshold_tuple_list,kernel_choice, is_gaussian_blur=False):

        """
        
		threshold_tuple_list   : (Array) Consisting of tuples element
        kernel_choice          : (Int) [1,2]
		is_gaussian_blur       : (Bool) True if the image data processes into gaussian first, else otherwise
		"""
        threshold=threshold_tuple_list
        ori_image=self.image.copy()
        kernel={1:(3,3),2:(5,5)}
        kernel_select=kernel[kernel_choice]
        
        
        #Define result summary
        score_eval=[]
        
        
        #define metric
       
        mse=[]
        similar=[]
        accuracy=[]
        psnr=[]
        
        #define size and val_score
        
        size_index=round(self.N*len(self.image))
        val_arr=[]
        
        if is_gaussian_blur:
            for par in threshold:
                
                #select the pair of parameters
                
                par1,par2=par
                for i in range(self.N):
                    
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    
                    #Gaussian Blur
                    
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img], 
                                                  kernel_select, 
                                                  0))
                        Y.append(ori_image[id_img])
                        
                    #Edge detection and its evaluation
                    
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.Canny(im_x,threshold1=par1,threshold2=par2)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=[str(par),np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                #gathering all the evaluation
            df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
            return df_score           
                    
        else:
            for par in threshold:
                #select the pair of parameters
                par1,par2=par
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.Canny(im_x,threshold1=par1,threshold2=par2)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=[str(par),np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
            df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
            return df_score           
                            
                    
                        
                        
         
    

    def prewitt_cv(self,orientation_kernel,kernel_choice,is_gaussian_blur=False):
        """
        orientation_kernel     : (Int) (Horizontal, Vertical) = (0,1)
        kernel_choice          : (Int) [1,2] for 1 :(3,3) and 2:(5,5)
		is_gaussian_blur       : (Bool) True if the image data processes into gaussian first, else otherwise
        """
        
        #Define Variable
        ori_image=self.image.copy()
        kernel={1:(3,3),2:(5,5)}
        kernel_select=kernel[kernel_choice]
        
        #Define result summary
        score_eval=[]
        
        #define metric
        mse=[]
        similar=[]
        accuracy=[]
        psnr=[]
        
        #define size and val_score
        
        size_index=round(self.N*len(self.image))
        val_arr=[]
        
        if (orientation_kernel==0):
            
            #This is Horizontal Prewitt's Edge Detection
            kernel_mat=np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=np.float32)
            if (is_gaussian_blur):
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kernel_mat)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Horizontal + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kernel_mat)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Horizontal + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score       
        
        elif (orientation_kernel==1):
            
            #This is Vertical Prewitt's Edge Detection
            
            kernel_mat=np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]], dtype=np.float32)
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kernel_mat)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Vertical + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kernel_mat)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Vertical + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score       
        else:
            raise ValueError("Error inputting the orientation_kernel ([1,2]) and is_gaussian_blur ([False,True])")
        
        
       
    def kirsch_cv(self,orientation_kernel,kernel_choice,is_gaussian_blur=False):

        """
        orientation_kernel     : (Int) [0,1,2,3,4,5,6,7]
                                 0 = kirsch_kernel_north
                                 1 = kirsch_kernel_northeast
                                 2 = kirsch_kernel_east
                                 3 = kirsch_kernel_south_east
                                 4 = kirsch_kernel_south
                                 5 = kirsch_kernel_south_west
                                 6 = kirsch_kernel_west
                                 7 = kirsch_kernel_north_west
		is_gaussian_blur       : (Bool) True if the image data processes into gaussian first, else otherwise
        kernel_choice          : (Int) [1,2] for 1 :(3,3) and 2:(5,5)
        """
        
        #Define Variable
        ori_image=self.image.copy()
        kernel={1:(3,3),2:(5,5)}
        kernel_select=kernel[kernel_choice]
        
        #Define result summary
        score_eval=[]
        
        #define metric
        mse=[]
        similar=[]
        accuracy=[]
        psnr=[]
        
        #define size and val_score
        
        size_index=round(self.N*len(self.image))
        val_arr=[]
        
        if (orientation_kernel==0):
            
            #This is  Kirsch's North Edge Detection
            kirsch_kernel_north = np.array([[-3, -3, -3],
                                [-3,  0, -3],
                                [ 5,  5,  5]], dtype=np.float32)
            if (is_gaussian_blur):
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_north)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["North + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_north)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["North + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score       
        
        elif (orientation_kernel==1):
            
            #This is Kirsch's Notheast Edge Detection
            
            kirsch_kernel_northeast = np.array([[-3, -3,  5],
                                    [-3,  0,  5],
                                    [-3, -3,  5]], dtype=np.float32)
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_northeast)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Northeast + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_northeast)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Northeast + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
        elif(orientation_kernel==2):
        
            #This is Kirsch's East Edge Detection
         
            kirsch_kernel_east=np.array([[-3,  5,  5],
                                 [-3,  0,  5],
                                 [-3, -3, -3]], dtype=np.int32)
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_east)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["East + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_east)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["East + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
                              
        elif (orientation_kernel==3):
            #This is Kirsch's Southeast Edge Detection
            
            kirsch_kernel_southeast = np.array([[ 5,  5,  5],
                                      [-3,  0, -3],
                                      [-3, -3, -3]], dtype=np.int32) 
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_southeast)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Southeast + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_southeast)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Southeast + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score 
            
                  
        elif (orientation_kernel==4):
            #This is Kirsch's South Edge Detection
            
            kirsch_kernel_south = np.array([[ 5,  5,  5],
                                      [-3,  0, -3],
                                      [-3, -3, -3]], dtype=np.int32) 
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_south)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["South + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_south)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["South + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score  
        elif (orientation_kernel==5):
            
            #This is Kirsch's Southwest Edge Detection
            
            kirsch_kernel_southwest = np.array([[-3,  5, -3],
                                    [-3,  0, -3],
                                    [ 5,  5, -3]], dtype=np.int32) 
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_southwest)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Southwest + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_southwest)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Southwest + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
        elif (orientation_kernel==6):
            
            #This is Kirsch's West Edge Detection
            
            kirsch_kernel_west = np.array([[-3, -3, -3],
                                 [ 5,  0, -3],
                                 [ 5,  5, -3]], dtype=np.int32) 
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_west)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["West + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_west)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["West + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score    

        elif (orientation_kernel==7):
            #This is Kirsch's Northwest Edge Detection
            
            kirsch_kernel_northwest = np.array([[-3, -3, -3],
                                      [-3,  0, -3],
                                      [ 5,  5,  5]], dtype=np.int32) 
            if is_gaussian_blur:
                
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,high=len(self.image)-1,size=size_index)
                    for id_img in index:
                        X.append(cv2.GaussianBlur(ori_image[id_img],kernel_select,0))
                        Y.append(ori_image[id_img])
                    
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=cv2.filter2D(im_x, -1, kirsch_kernel_northwest)
                        
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Northwest + Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score
            
            else:
                for i in range(self.N):
                    #Iterative process for N-th selection image
                    X=[]
                    Y=[]
                    index=np.random.randint(low=0,
                                            high=len(self.image)-1,
                                            size=size_index)
                    for id_img in index:
                        X.append(ori_image[id_img])
                        Y.append(ori_image[id_img])
                        
                    #Edge detection and its evaluation
                    for im_x,im_y in zip(X,Y):
                        im_y=im_y=cv2.filter2D(im_x, -1, kirsch_kernel_northwest)
                        fun_met=met.metrics(im_x,im_y)
                        
                        mse.append(fun_met.MSE())
                        psnr.append(fun_met.PSNR())
                        similar.append(fun_met.similarity(channel_axis=None,bool_full=False))
                        accuracy.append(fun_met.accuracy())
                        
                    eval=["Northwest + Non Gaussian Blur",np.mean(mse),np.mean(psnr),np.mean(similar),np.mean(accuracy)]
                    score_eval.append(eval)
                    
                #gathering all the evaluation
                df_score=pd.DataFrame(score_eval,columns=['parameters','mse','psnr','similar','accuracy'])           
                return df_score          
        else:
            raise ValueError("Error inputting the orientation_kernel ([1,2]) and is_gaussian_blur ([False,True])")
    
    def visualize (self, X, Y, title, nrows=4, ncols=4):
        random_idx=np.arange(len(X))
        Y_viz=Y[0]
        Y_viz_1=Y[1]
        Y_viz_2=Y[2]
        random_idx=np.repeat([random_idx], repeats=4, axis=0)
        
        
        
        fig,axe=plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,12))
        plt.suptitle('Image Data')
        for rows in range(random_idx.shape[0]):
            for cols in range(random_idx.shape[1]):
                if (rows==0):
                    axe[rows,cols].imshow(X[random_idx[rows,cols]],cmap='gray')
                    val_rand=random_idx[rows,cols]
                    axe[rows,cols].set_title('Before Edge Detection:%d' %val_rand)
                    axe[rows,cols].grid(False)
                elif(rows==1):
                    axe[rows,cols].imshow(Y_viz[random_idx[rows,cols]],cmap='gray')
                    val_rand=random_idx[rows,cols]
                    axe[rows,cols].set_title('%s' %title[0] +  ' Image:%d' %val_rand)
                    axe[rows,cols].grid(False)
                elif(rows==2):
                    axe[rows,cols].imshow(Y_viz_1[random_idx[rows,cols]],cmap='gray')
                    val_rand=random_idx[rows,cols]
                    axe[rows,cols].set_title('%s' %title[1] +  ' Image:%d' %val_rand)
                    axe[rows,cols].grid(False)    
                else:
                    axe[rows,cols].imshow(Y_viz_2[random_idx[rows,cols]],cmap='gray')
                    val_rand=random_idx[rows,cols]
                    axe[rows,cols].set_title('%s' %title[2] +  ' Image:%d' %val_rand)
                    axe[rows,cols].grid(False) 
            
        plt.tight_layout()
        plt.show()

    def visualize_kirsch(self,orientation_kernel):
        
        kernel={1:(3,3),2:(5,5)}
        idx=[691,2320,2647,5076]
        X=[]
        for i in idx:
            X.append(self.image.copy()[i])
        if(orientation_kernel==0):
            #kirsch_kernel_north
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_north))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_north))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_north))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-north','kirsch-north(3,3)','kirsch-north(5,5)']
            
            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
            
        elif(orientation_kernel==1):
            #kirsch_kernel_northeast
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_northeast))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_northeast))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_northeast))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-northeast','kirsch-northeast(3,3)','kirsch-northeast(5,5)']
            
                        
            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
            
        elif(orientation_kernel==2):
            #kirsch_kernel_east
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_east))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_east))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_east))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-east','kirsch-east(3,3)','kirsch-east(5,5)']

            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
            
        elif(orientation_kernel==3):
            #kirsch_kernel_southeast
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_southeast))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_southeast))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_southeast))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-southeast','kirsch-southeast(3,3)','kirsch-southeast(5,5)']

            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
            
        elif(orientation_kernel==4):
            #kirsch_kernel_south
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_south))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_south))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_south))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-south','kirsch-south(3,3)','kirsch-south(5,5)']

            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
        
        elif(orientation_kernel==5):
            #kirsch_kernel_southwest
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_southwest))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_southwest))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_southwest))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-southwest','kirsch-southwest(3,3)','kirsch-southwest(5,5)']

            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
        
        elif(orientation_kernel==6):
            #kirsch_kernel_west
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_west))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_west))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_west))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-west','kirsch-west(3,3)','kirsch-west(5,5)']

            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
        
        
        elif(orientation_kernel==7):
            #kirsch_kernel_northwest
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.kirsch_kernel_northwest))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.kirsch_kernel_northwest))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.kirsch_kernel_northwest))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['kirsch-northwest','kirsch-northwest(3,3)','kirsch-northwest(5,5)']

            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
        
        else:
            raise ValueError("Please check your input the orientation_kernel. Orientation kernel between 0 and 7")
    
    def visualize_prewitt(self,orientation_kernel):
        kernel={1:(3,3),2:(5,5)}
        idx=[691,2320,2647,5076]
        X=[]
        for i in idx:
            X.append(self.image.copy()[i])
        if(orientation_kernel==0):
            #prewitt_horizontal
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.prewitt_horizontal))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.prewitt_horizontal))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.prewitt_horizontal))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['prewitt-horizontal','prewitt-horizontal(3,3)','prewitt-horizontal(5,5)']
            
            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
        elif (orientation_kernel==1):
            #prewitt_vertical
            
            Y_viz=[]
            Y_viz_1=[]
            Y_viz_2=[]
            for j,imX in enumerate(X):
                viz_1=cv2.GaussianBlur(imX,kernel[1],0)
                viz_2=cv2.GaussianBlur(imX,kernel[2],0)
                Y_viz.append(cv2.filter2D(imX, -1, self.prewitt_vertical))
                Y_viz_1.append(cv2.filter2D(viz_1, -1,self.prewitt_vertical))
                Y_viz_2.append(cv2.filter2D(viz_2,-1,self.prewitt_vertical))
            Y_new=[Y_viz,Y_viz_1,Y_viz_2] 
            title=['prewitt-vertical','prewitt-vertical(3,3)','prewitt-vertical(5,5)']
            self.visualize(X=X,
                           Y=Y_new,
                           title=title)
        
        else:
            raise ValueError("Please check your input the orientation_kernel. Orientation kernel between 0 and 7")
    
    def visualize_canny(self,threshold_tuple_list):
        kernel={1:(3,3),2:(5,5)}
        idx=2647
        X=[]
        X.append(self.image.copy()[idx])
        Y_viz=[]
        Y_viz_1=[]
        Y_viz_2=[]
        for id,par in enumerate(threshold_tuple_list):
            par1,par2=par
            viz_1=cv2.GaussianBlur(X,kernel[1],0)
            viz_2=cv2.GaussianBlur(X,kernel[2],0)
            Y_viz.append(cv2.Canny(X,threshold1=par1,threshold2=par2))
            Y_viz_1.append(cv2.Canny(viz_1,threshold1=par1,threshold2=par2))
            Y_viz_2.append(cv2.Canny(viz_2,threshold1=par1,threshold2=par2))
        
        title=np.empty((3,len(threshold_tuple_list)),dtype=int)
        for row in title.shape[0]:
            for col in title.shape[1]:
                if (row==0):
                    title[row,col]='canny-cv '+str(par)
                elif(row==1):
                    title[row,col]='canny-cv(3,3) '+str(par)
                else:
                    title[row,col]='canny-cv(5,5) '+str(par)
        
        
        fig,axe=plt.subplots(nrows=3,ncols=len(threshold_tuple_list),figsize=(12,12))
        plt.suptitle('Image Data')
        for rows in range(3):
            for cols in range(len(threshold_tuple_list)):
                if (rows==0):
                    axe[rows,cols].imshow(X,cmap='gray')
                    val_rand=idx
                    axe[rows,cols].set_title('Before Edge Detection:%d' %val_rand)
                    axe[rows,cols].grid(False)
                elif(rows==1):
                    axe[rows,cols].imshow(Y_viz[cols],cmap='gray')
                    val_rand=idx
                    axe[rows,cols].set_title('%s' %title[rows,cols] +  ' Image:%d' %val_rand)
                    axe[rows,cols].grid(False)
                elif(rows==2):
                    axe[rows,cols].imshow(Y_viz_1[cols],cmap='gray')
                    val_rand=idx
                    axe[rows,cols].set_title('%s' %title[rows,cols] +  ' Image:%d' %val_rand)
                    axe[rows,cols].grid(False)    
                else:
                    axe[rows,cols].imshow(Y_viz_2[cols],cmap='gray')
                    val_rand=idx
                    axe[rows,cols].set_title('%s' %title[2] +  ' Image:%d' %val_rand)
                    axe[rows,cols].grid(False) 
            
        plt.tight_layout()
        plt.show()
                    
            
            
                
            
    
        