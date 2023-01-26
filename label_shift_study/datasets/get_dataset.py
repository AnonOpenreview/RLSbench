import logging

import numpy as np
import torch
import torchvision

from label_shift_study.datasets.data_utils import *
from label_shift_study.label_shift_utils import *
from robustness.tools import folder
from robustness.tools.breeds_helpers import (make_entity13, make_entity30,
                                             make_living17, make_nonliving26)
from robustness.tools.helpers import get_label_mapping
from torchvision.datasets import ImageFolder as torch_ImageFolder
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.unlabeled.fmow_unlabeled_dataset import FMoWUnlabeledDataset 
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.datasets.wilds_dataset import WILDSSubset

logger = logging.getLogger("label_shift")

def get_cifar10(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.8, seed=42):
    
    root_dir = f"{root_dir}/cifar10"

    cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
                    "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
                    "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
    severities = [1, 2, 3, 4 ,5]

    CIFAR10 = dataset_with_targets(torchvision.datasets.CIFAR10)

    if source or (target and target_split==0): 

        trainset = CIFAR10(root=root_dir, train=True, download=False, transform=None)
        
        source_train_idx, source_test_idx = split_idx(trainset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset = Subset(trainset, source_train_idx, transform = transforms['source_train'])

        source_testset = Subset(trainset, source_test_idx, transform = transforms['source_test'])

        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")


    if target:

        # CIFAR 10 v1
        if target_split == 0:

            targetset = CIFAR10(root=root_dir, train=False, download=False, transform=None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)
    
            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

            # logger.debug(f"Train indices {target_train_idx} and test indicies {target_test_idx}")

        # CIFAR 10 v2
        elif target_split == 1: 

            target_trainset = CIFAR10v2(root=f"{root_dir}/cifar10v2/", train=True, download=False,\
            transform=transforms['target_train'])

            target_testset = CIFAR10v2(root=f"{root_dir}/cifar10v2/", train=False, download=False,\
            transform=transforms['target_test'])

        # CIFAR 10 C
        elif target_split < 97: 
            split = target_split - 2
            cifar_c_idx = split//5 
            severity_idx = split%5 + 1

            targetset = CIFAR_C(root=f"{root_dir}/cifar10c/", data_type=cifar_c[cifar_c_idx], severity=severity_idx, transform=None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else: 
            raise ValueError("Unlabeled_split must be between 0 and 97 for CIFAR10 dataset")


        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}
    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset

    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets


def get_cifar100(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    root_dir = f"{root_dir}/cifar100"


    cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
                    "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
                    "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
    severities = [1, 2, 3, 4 ,5]

    CIFAR100 = dataset_with_targets(torchvision.datasets.CIFAR100)


    if source or (target and target_split==0): 
        trainset = CIFAR100(root=root_dir, train=True, download=True, transform=None)
        
        source_train_idx, source_test_idx = split_idx(trainset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset = Subset(trainset, source_train_idx, transform = transforms['source_train'])

        # source_testset = CIFAR100(root=root_dir, train=False, download=False, transform=transforms['source_test'])
        source_testset = Subset(trainset, source_test_idx, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:

        # CIFAR 100 v1
        if target_split == 0: 
            targetset = CIFAR100(root=root_dir, train=False, download=False, transform=None)
            
            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        
        # CIFAR 100 C
        elif target_split < 96:
            split = target_split - 1
            cifar_c_idx = split//5 
            severity_idx = split%5 + 1

            targetset = CIFAR_C(root=f"{root_dir}/cifar100c/", data_type=cifar_c[cifar_c_idx], severity=severity_idx, transform=None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        
        else:
            raise ValueError("Unlabeled_split must be between 0 and 96 for CIFAR100 dataset")
        
        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")

    datasets = {}
    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets

def get_fmow(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    # FMoWUnlabeledDataset = dataset_with_get_item(FMoWUnlabeledDataset)

    # import pdb; pdb.set_trace()
    dataset = FMoWDataset(download=False, root_dir=root_dir)
    unlabeled_dataset = FMoWUnlabeledDataset(download=False, root_dir=root_dir)
    # def get_groups(testset, group_idx): 
    #     groups = dataset._eval_groupers['region'].metadata_to_group(testset.metadata_array)

    #     idx = np.where(groups==group_idx)[0]
    #     return WILDSSubset(testset, idx, None)

    if source: 
        source_trainset = dataset.get_subset('train', transform = transforms['source_train'])
        source_testset = dataset.get_subset('id_test', transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target: 
        if target_split == 0:
            target_trainset = unlabeled_dataset.get_subset('train_unlabeled', transform = transforms['target_train'], load_y=True)
            target_testset = dataset.get_subset('id_val', transform = transforms['target_test'])

            target_ydist = calculate_marginal(np.array(target_trainset.y_array), num_classes)
            target_test_idx = get_resampled_indices(np.array(target_testset.y_array), num_classes, target_ydist, seed=seed)
            
            target_testset = Subset(target_testset, target_test_idx)

            # target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            # target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            # target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])


        elif target_split == 1: 
            target_trainset = unlabeled_dataset.get_subset('val_unlabeled', transform = transforms['target_train'], load_y=True)

            target_testset = dataset.get_subset('val', transform = transforms['target_test'])

            target_ydist = calculate_marginal(np.array(target_trainset.y_array), num_classes)
            target_test_idx = get_resampled_indices(np.array(target_testset.y_array), num_classes, target_ydist, seed=seed)
            
            target_testset = Subset(target_testset, target_test_idx)

            # target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            # target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            # target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        
        elif target_split == 2:
            target_trainset = unlabeled_dataset.get_subset('test_unlabeled', transform = transforms['target_train'], load_y=True)

            target_testset = dataset.get_subset('test', transform = transforms['target_test'])

            target_ydist = calculate_marginal(np.array(target_trainset.y_array), num_classes)
            target_test_idx = get_resampled_indices(np.array(target_testset.y_array), num_classes, target_ydist, seed=seed)
            
            target_testset = Subset(target_testset, target_test_idx)
            # targetset = dataset.get_subset('test', transform = None)

            # target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            # target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            # target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # TODO: Add target_split grouped by region

        else:
            raise ValueError("Unlabeled_split must be between 0 and 2 for FMoW dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")
        
    datasets = {}
    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    return datasets


def get_rxrx1(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    
    dataset = RxRx1Dataset(download=False, root_dir=root_dir)
    
    if source or (target and target_split == 0):
        source_trainset = dataset.get_subset('train', transform = transforms['source_train'])
        
        sourceset = dataset.get_subset('id_test', transform = None)

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.4, seed=seed)

        source_testset = Subset(sourceset, source_idx, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:
        if target_split == 0:
            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])
        
        elif target_split == 1:

            targetset = dataset.get_subset('val', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        
        elif target_split == 2:
            targetset = dataset.get_subset('test', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        

        else: 
            raise ValueError("Unlabeled_split must be between 0 and 2 for RxRx1 dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}
    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets


def get_iwildcams(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    
    dataset = IWildCamDataset(download=False, root_dir=root_dir)

    if source: 
        source_trainset = dataset.get_subset('train', transform = transforms['source_train'])
        source_testset = dataset.get_subset('id_test', transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target: 
        if target_split == 0:
            targetset = dataset.get_subset('id_val', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])


        elif target_split == 1: 
            targetset = dataset.get_subset('val', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        
        elif target_split == 2:
            targetset = dataset.get_subset('test', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else:
            raise ValueError("Unlabeled_split must be between 0 and 2 for IWildCam dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}
    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    return datasets


def get_camelyon(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    
    dataset = Camelyon17Dataset(download=False, root_dir=root_dir)
    
    if source or (target and target_split == 0):
        sourceset = dataset.get_subset('train', transform = None)

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset = Subset(sourceset, source_idx, transform = transforms['source_test'])

        source_testset = dataset.get_subset('id_val', transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:
        if target_split == 0:
            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])
        
        elif target_split == 1:

            targetset = dataset.get_subset('val', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        
        elif target_split == 2:
            targetset = dataset.get_subset('test', transform = None)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])
        

        else: 
            raise ValueError("Unlabeled_split must be between 0 and 2 for Camelyon dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")

    datasets = {}
    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset
    
    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets


def get_domainnet(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):

    root_dir = f"{root_dir}/domainnet"

    ImageFolder = dataset_with_targets(torch_ImageFolder)

    if source or (target and target_split == 0):
    
        sourceset = ImageFolder(f"{root_dir}/real")

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset_idx, source_testset_idx = split_idx(sourceset.y_array[source_idx], num_classes, source_frac=split_fraction, seed=seed)

        source_trainset_idx, source_testset_idx = source_idx[source_trainset_idx], source_idx[source_testset_idx]

        source_trainset = Subset(sourceset, source_trainset_idx, transform = transforms['source_train'])
        source_testset = Subset(sourceset, source_testset_idx, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:

        if target_split == 0:
            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])
        
        # Clipart 
        elif target_split == 1:

            targetset = ImageFolder(f"{root_dir}/clipart")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # Sketch
        elif target_split == 2:
            
            targetset = ImageFolder(f"{root_dir}/sketch")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # Paint 
        elif target_split == 3:

            targetset = ImageFolder(f"{root_dir}/painting")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # Quickdraw
        elif target_split == 4:
                
                targetset = ImageFolder(f"{root_dir}/quickdraw")
    
                target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)
    
                target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
                target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else:
            raise ValueError("Unlabeled_split must be between 0 and 4 for DomainNet dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}

    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets


def get_entity13(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    return get_breeds("entity13", source, target, root_dir, target_split, transforms, num_classes, split_fraction, seed)


def get_entity30(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):

    return get_breeds("entity30", source, target, root_dir, target_split, transforms, num_classes, split_fraction, seed)

def get_living17(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):

    return get_breeds("living17", source, target, root_dir, target_split, transforms, num_classes, split_fraction, seed)

def get_nonliving26(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):

    return get_breeds("nonliving26", source, target, root_dir, target_split, transforms, num_classes, split_fraction, seed)


def get_breeds(dataset=None, source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):

    root_dir = f"{root_dir}/imagenet/"

    if dataset == "living17": 
        ret = make_living17(f"{root_dir}/imagenet_hierarchy/", split="good")
    elif dataset == "entity13":
        ret = make_entity13(f"{root_dir}/imagenet_hierarchy/", split="good")
    elif dataset == "entity30":
        ret = make_entity30(f"{root_dir}/imagenet_hierarchy/", split="good")
    elif dataset == "nonliving26":
        ret = make_nonliving26(f"{root_dir}/imagenet_hierarchy/", split="good")


    ImageFolder = dataset_with_targets(folder.ImageFolder)
    
    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])  
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])  

    if source or (target and target_split==0): 

        sourceset = ImageFolder(f"{root_dir}/imagenetv1/train/", label_mapping=source_label_mapping)

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset = Subset(sourceset, source_idx, transform = transforms['source_train'])
        source_testset = ImageFolder(f"{root_dir}/imagenetv1/val/", label_mapping=source_label_mapping, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target: 

        if target_split == 0:

            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])

        elif target_split == 1:

            targetset = ImageFolder(f"{root_dir}/imagenetv1/train/", label_mapping=target_label_mapping)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        elif target_split == 2:

            targetset = ImageFolder(f"{root_dir}/imagenetv2/imagenetv2-matched-frequency-format-val", label_mapping=source_label_mapping)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])


        elif target_split == 3:
            
            targetset = ImageFolder(f"{root_dir}/imagenetv2/imagenetv2-matched-frequency-format-val", label_mapping=target_label_mapping)

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else: 
            raise ValueError("target_split must be between 0 and 3 for BREEDs dataset") 

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")

    
    dataset = {}

    if source and target: 
        dataset['source_train'] = source_trainset
        dataset['source_test'] = source_testset
        dataset['target_train'] = target_trainset
        dataset['target_test'] = target_testset
    
    elif source:
        dataset['source_train'] = source_trainset
        dataset['source_test'] = source_testset
    
    elif target:
        dataset['target_train'] = target_trainset
        dataset['target_test'] = target_testset

    return dataset

def get_visda(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.8, seed=42):

    root_dir = f"{root_dir}/visda/"

    ImageFolder = dataset_with_targets(torch_ImageFolder)

    if source or (target and target_split == 0):
    
        sourceset = ImageFolder(f"{root_dir}/train")

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset_idx, source_testset_idx = split_idx(sourceset.y_array[source_idx], num_classes, source_frac=split_fraction, seed=seed)

        source_trainset_idx, source_testset_idx = source_idx[source_trainset_idx], source_idx[source_testset_idx]

        source_trainset = Subset(sourceset, source_trainset_idx, transform = transforms['source_train'])
        source_testset = Subset(sourceset, source_testset_idx, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:

        if target_split == 0:
            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])
        
        # Validation
        elif target_split == 1:

            targetset = ImageFolder(f"{root_dir}/validation")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # Test
        elif target_split == 2:
            
            targetset = ImageFolder(f"{root_dir}/test")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else:
            raise ValueError("Unlabeled_split must be between 0 and 2 for Visda dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}

    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets

def get_officehome(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):
    
    root_dir = f"{root_dir}/officehome/"

    ImageFolder = dataset_with_targets(torch_ImageFolder)

    if source or (target and target_split == 0):
    
        sourceset = ImageFolder(f"{root_dir}/Product/")

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset_idx, source_testset_idx = split_idx(sourceset.y_array[source_idx], num_classes, source_frac=split_fraction, seed=seed)

        source_trainset_idx, source_testset_idx = source_idx[source_trainset_idx], source_idx[source_testset_idx]

        source_trainset = Subset(sourceset, source_trainset_idx, transform = transforms['source_train'])
        source_testset = Subset(sourceset, source_testset_idx, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:

        if target_split == 0:
            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])
        
        # Real
        elif target_split == 1:

            targetset = ImageFolder(f"{root_dir}/RealWorld")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # ClipArt
        elif target_split == 2:
            
            targetset = ImageFolder(f"{root_dir}/Clipart")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # Art 
        elif target_split == 3:
            
            targetset = ImageFolder(f"{root_dir}/Art")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else:
            raise ValueError("Unlabeled_split must be between 0 and 4 for OfficeHome dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}

    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets


def get_office31(source = True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.2, seed=42):

    root_dir = f"{root_dir}/office31/"
    
    ImageFolder = dataset_with_targets(torch_ImageFolder)

    if source or (target and target_split == 0):
    
        sourceset = ImageFolder(f"{root_dir}/amazon/")

        source_idx, target_idx = split_idx(sourceset.y_array, num_classes, source_frac=0.8, seed=seed)

        source_trainset_idx, source_testset_idx = split_idx(sourceset.y_array[source_idx], num_classes, source_frac=split_fraction, seed=seed)

        source_trainset_idx, source_testset_idx = source_idx[source_trainset_idx], source_idx[source_testset_idx]

        source_trainset = Subset(sourceset, source_trainset_idx, transform = transforms['source_train'])
        source_testset = Subset(sourceset, source_testset_idx, transform = transforms['source_test'])
        logger.debug(f"Size of source data; train {len(source_trainset)} and test {len(source_testset)}")

    if target:

        if target_split == 0:
            target_train_idx, target_test_idx = split_idx(sourceset.y_array[target_idx], num_classes, source_frac=split_fraction, seed=seed)

            target_train_idx, target_test_idx = target_idx[target_train_idx], target_idx[target_test_idx]

            target_trainset = Subset(sourceset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(sourceset, target_test_idx, transform = transforms['target_test'])
        
        # Webcam
        elif target_split == 1:

            targetset = ImageFolder(f"{root_dir}/webcam")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        # DSLR
        elif target_split == 2:
            
            targetset = ImageFolder(f"{root_dir}/dslr")

            target_train_idx, target_test_idx = split_idx(targetset.y_array, num_classes, source_frac=split_fraction, seed=seed)

            target_trainset = Subset(targetset, target_train_idx, transform = transforms['target_train'])
            target_testset = Subset(targetset, target_test_idx, transform = transforms['target_test'])

        else:
            raise ValueError("Unlabeled_split must be between 0 and 2 for Office-31 dataset")

        logger.debug(f"Target split {target_split} with size of target data; train {len(target_trainset)} and test {len(target_testset)}")


    datasets = {}

    if source and target:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    elif source:
        datasets['source_train'] = source_trainset
        datasets['source_test'] = source_testset
    
    elif target:
        datasets['target_train'] = target_trainset
        datasets['target_test'] = target_testset

    return datasets    

