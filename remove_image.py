import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import cv2
from PIL import Image
from torchvision.transforms import functional as F
import metrics_data as metrics
import numpy as np

class isolate_object:
    
    def __init__ (self):
        self.fail_isolate=0
        self.completed_isolate=0
    
    def MASK_RCNN_Restnet50(self):
        model=maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model
    
    def masking_image(self, model_mask,image):
        model = model_mask
        isolated_image=[]
        val_metric=[]
        tensor_image=F.to_tensor(image)
        tensor_image=tensor_image.unsqueeze(0)
        with torch.no_grad():
            predictions=model(tensor_image)
        masks_predict=predictions[0]['masks']
        get_mask=masks_predict.squeeze(1).cpu().numpy()
        if (np.size(get_mask)==0):
            self.fail_isolate=self.fail_isolate+1
            return image
        else:
            for idx,sub in enumerate(get_mask):
                isolate_object=np.zeros_like(image)
                for channel in range(3):
                    isolate_object[:,:,channel]=image[:,:,channel]*sub
                isolated_image.append(isolate_object)
                met=metrics.metrics(image,isolate_object)
                val_metric.append(met.PSNR())
                self.completed_isolate=self.completed_isolate+1
            return isolated_image[np.argmax(val_metric)]
            
        
        
        
        
        
        
        