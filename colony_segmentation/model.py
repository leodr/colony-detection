from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_segmentation(num_classes, pretrained=True):
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        box_detections_per_img=300,
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # increase the number of boxes that can be detected
    model.roi_heads.detections_per_img = 300

    return model
