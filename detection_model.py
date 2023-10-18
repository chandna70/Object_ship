import torch
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from CustomeAugment import CustomRCNNTransform



def Augmented_Image():
    transform_custom=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(3,3),sigma=(0.1, 2.5)),
        transforms.ColorJitter(brightness=(0.55,1),
                               contrast=(0,1)),
        transforms.ElasticTransform(alpha=90.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
    return transform_custom


def FRCNN_RESNET50_Model(NUM_CLASSES):
    feature_maps=resnet50(pretrained=True)
    feature_maps.out_channels=2048
    
    rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)
    model = FasterRCNN(CustomRCNNTransform
                       (model=feature_maps,transform=Augmented_Image()),
                       num_classes=NUM_CLASSES,  # Define the number of classes in your dataset
                    rpn_anchor_generator=rpn_anchor_generator) 
    model.eval()
    return model
    