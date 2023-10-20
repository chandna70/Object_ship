import torch,os
import pandas as pd
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from albumentations.pytorch.transforms import ToTensorV2
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from detecto.utils import normalize_transform

def Augmented_Image1():
    transform_custom=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1, 2.5)),
        transforms.ColorJitter(brightness=(0.55,1),
                               contrast=(0,1)),
        transforms.ElasticTransform(alpha=90.0)
        ])
    return transform_custom

def Augmented_Image2():
    transform_custom=transforms.Compose([
        transforms.ToPILImage(),transforms.Resize(800),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1, 2.5)),
        transforms.ColorJitter(brightness=(0.55,1),
                               contrast=(0,1)),
        transforms.ElasticTransform(alpha=90.0),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        ])
    return transform_custom


def FRCNN_RESNET50_Model(NUM_CLASSES):
    feature_maps=resnet50(pretrained=True)
    feature_maps.out_channels=2048
    
    rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model = FasterRCNN(backbone=feature_maps,
                       num_classes=NUM_CLASSES,  # Define the number of classes in your dataset
                    rpn_anchor_generator=rpn_anchor_generator)
    model.transform=Augmented_Image() 
    
    return model

def Detecto_File_CSV(label_path,save_to):
    utils.xml_to_csv(label_path,save_to)
    path_file=save_to
    dataset=pd.read_csv(path_file,encoding='utf-8',sep=',')
    return dataset
    
def Detecto_train(dataset_csv,image_path,categories):
    df_training=core.Dataset(label_data=dataset_csv,
                             image_folder=image_path)
                             #transform=Augmented_Image2())
    #dataloader=core.DataLoader(df_training)
    #print("Here your dataloader",dataloader)
    multiclass=categories
    Classifier=core.Model(multiclass)
    history=Classifier.fit(dataset=df_training,
                           epochs=5,
                           verbose=True,learning_rate=0.05)
    return history, Classifier