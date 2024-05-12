import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_segmentation(num_classes, pretrained=True):
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        num_classes=num_classes,
        box_detections_per_img=300,
    )

    return model
