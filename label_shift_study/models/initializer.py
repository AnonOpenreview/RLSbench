import logging

import antialiased_cnns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from label_shift_study.utils import load

logger = logging.getLogger("label_shift")

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x


def initialize_model(model_name, dataset_name, num_classes, featurize=False, in_features=None,  pretrained = False,  pretrained_path = None):
    """
    Initializes models according to the config
        Args:
            - model_name: name of the model
            - dataset_name: name of the dataset
            - num_classes (int): number of classes in the dataset
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)
    """

    if model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnetb0') and "cifar" in dataset_name:
        from label_shift_study.models.cifar_efficientnet import EfficientNetB0
        from label_shift_study.models.cifar_resnet import ResNet18, ResNet34, ResNet50, ResNet101

        arch = {
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "efficientnet_b0": EfficientNetB0
        }

        featurizer = arch[model_name](num_classes=1000, features=True)

        if pretrained:       
            assert pretrained_path is not None, "Must provide pretrained_path if pretrained=True"      
            load(featurizer, pretrained_path) 
        
        d_out = getattr(featurizer, "linear").in_features
        featurizer.d_out = d_out
        classifier = nn.Linear(d_out, num_classes)
        model = (featurizer, classifier)
    

        if not featurize: 
            model = nn.Sequential(*model)


    elif model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'efficientnet_b0'):

        featurizer = initialize_torchvision_model(
            name=model_name,
            d_out=None, 
            pretrained=pretrained)
            
        classifier = nn.Linear(featurizer.d_out, num_classes)
        model = (featurizer, classifier)

        # if pretrained_type in ('swav'): 
            # load(model[0], pretrained_path) 

        if not featurize:
            model = nn.Sequential(*model)


    elif model_name in ('cliprn50', 'cliprn101', 'clipvitb16', 'clipvitb32', 'clipvitl14'):

        from label_shift_study.models.clip import ClipRN50, ClipRN101, ClipViTB16, ClipViTB32, ClipViTL14

        arch = {
            "cliprn50": ClipRN50,
            "cliprn101": ClipRN101,
            "clipvitb16": ClipViTB16,
            "clipvitb32": ClipViTB32, 
            'clipvitl14': ClipViTL14
        }
        
        model = arch[model_name](num_classes=num_classes)

        if not featurize:
            model = nn.Sequential(*model)
            

    elif model_name == 'logistic_regression':
        assert not featurize, "Featurizer not supported for logistic regression"
        model = nn.Linear(in_features=in_features, out_features=num_classes)

    else:
        raise ValueError(f'Model: {model_name} not recognized.')

    return model



def initialize_torchvision_model(name, d_out, pretrained=True):

    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        constructor_name = name
        last_layer_name = 'fc'
    elif name in  ('efficientnet_b0'):
        constructor_name = name
        last_layer_name = 'classifier'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(antialiased_cnns, constructor_name)
    model = constructor(pretrained=pretrained)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model

