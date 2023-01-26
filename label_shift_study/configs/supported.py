# see initialize_*() functions for correspondence=
# See algorithms/initializer.py
algorithms = [
    'ERM' , 'IW-ERM', 
    'ERM-aug', 'IW-ERM-aug',
    'ERM-oracle', 'IW-ERM-oracle',
    'ERM-adv',
    'DANN', 'CDANN', 
    'IW-DANN', 'IW-CDANN', 
    'IS-DANN', 'IS-CDANN',
    'COAL', 'IW-COAL',
    'SENTRY', 'IW-SENTRY',
    'FixMatch', 'IW-FixMatch', 'IS-FixMatch', 
    'PseudoLabel', 'IW-PseudoLabel',
    'NoisyStudent', 'IS-NoisyStudent',
    'CORAL', 'IW-CORAL',
    'BN_adapt', 'BN_adapt-adv' , 'IS-BN_adapt', 'IS-BN_adapt-adv',
    'TENT', 'IS-TENT',
    ]

label_shift_adapt = ['MLLS', 'true', 'RLLS', 'None', 'baseline']


# See transforms.py
transforms = ['image_base', 'image_resize_and_center_crop', 'image_none', 'rxrx1', 'clip']
additional_transforms = ['randaugment', 'weak', ]

# See models/initializer.py
models = [ 'resnet18', 'resnet34', 'resnet50', 'resnet101',
         'densenet121', 'clipvitb32', 'clipvitb16', 'clipvitl14', 'efficientnet_b0']

# Pre-training type
pretrainining_options = ['clip', 'imagenet', 'swav' , 'rand']

# See optimizer.py
optimizers = ['SGD', 'Adam', 'AdamW']

# See scheduler.py
schedulers = ['linear_schedule_with_warmup', 'cosine_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR', 'FixMatchLR', 'MultiStepLR']

# See losses.py
losses = ['cross_entropy', 'cross_entropy_logits']
