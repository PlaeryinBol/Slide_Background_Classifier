import timm
import torch.nn as nn
import torch.nn.functional as F

import config

MODELS = {
    'efficientnet': 'tf_efficientnetv2_s.in21k',
    'convnext': 'convnext_base.fb_in22k_ft_in1k',
    'swin': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'vit': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k'
}


class ClassificationModel(nn.Module):
    def __init__(self, model_name, apply_softmax=False):
        super(ClassificationModel, self).__init__()
        self.model_name = model_name
        self.apply_softmax = apply_softmax
        self.create_model()

    def create_model(self):
        self.model = timm.create_model(MODELS[self.model_name], pretrained=True, num_classes=config.NUM_CLASSES)

    def set_model(self, model_name):
        self.model_name = MODELS[model_name]
        self.create_model()

    def forward(self, x):
        result = self.model(x)
        if self.apply_softmax:
            result = F.softmax(result, dim=1)
        return result
