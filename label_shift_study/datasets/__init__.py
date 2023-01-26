from label_shift_study.datasets.data_utils import *
from label_shift_study.datasets.get_dataset import *
from label_shift_study.remote_data_utils import *

benchmark_datasets = [
    'camelyon',
    'iwildcam',
    'fmow',
    # 'rxrx1',
    'cifar10',
    'cifar100',
    'domainnet',
    'entity13', 
    'entity30',
    'living17',
    'nonliving26',
    'office31',
    'officehome',
    'visda'
]

supported_datasets = benchmark_datasets 

dataset_map = { 
    "cifar10" : get_cifar10, 
    "cifar100" : get_cifar100,
    'office31' : get_office31,
    'officehome' : get_officehome,
    'visda' : get_visda,
    'domainnet' : get_domainnet,
    'entity13' : get_entity13,
    'entity30' : get_entity30,
    'living17': get_living17,
    'nonliving26': get_nonliving26,
    'fmow': get_fmow,
    'iwildcam': get_iwildcams,
    'rxrx1': get_rxrx1,
    'camelyon': get_camelyon
}

dataset_file = {
    "cifar10": "cifar10.tar.gz", 
    "cifar100": "cifar100.tar.gz", 
    "camelyon": "camelyon17_v1.0.tar.gz", 
    "fmow": "fmow_v1.1.tar.gz", 
    "iwildcam": "iwildcam_v2.0.tar.gz", 
    "officehome": "officehome.tar.gz",
    "office31": "office31.tar.gz", 
    "visda": "visda.tar.gz", 
    "entity13": "imagenet.tar.gz",
    "entity30": "imagenet.tar.gz",
    "living17": "imagenet.tar.gz",
    "nonliving26": "imagenet.tar.gz",
    "domainnet": "domainnet.tar.gz", 
}

def get_dataset(dataset, source=True, target = False, root_dir = None, target_split = None, transforms = None, num_classes = None, split_fraction=0.8, seed=42, remote=False, remote_client=None, remote_bucket=None):
    """
    Returns the appropriate dataset
    Input:
        dataset: name of the dataset
        source: whether to return the source dataset
        target: whether to return the target dataset
        download: whether to download the dataset
        root_dir: root directory of the dataset
        target_split: which split of the target set to return
        transforms: transforms to apply to the dataset
        num_classes: number of classes in the dataset
    Output:
        dataset: labeled dataset (if unlabeled is False) or labeled and unlabeled dataset (if unlabeled is True)
    """

    if remote: 
        if not os.path.exists(root_dir): 
            os.makedirs(root_dir)
        
        if not os.path.exists(os.path.join(root_dir, dataset_file[dataset][:-7])):        
            download_dir_and_extract(f"data/{dataset_file[dataset]}", f"{root_dir}/{dataset_file[dataset]}", remote_bucket, remote_client)


    return dataset_map[dataset](source, target, root_dir, target_split, transforms, num_classes, split_fraction, seed)


