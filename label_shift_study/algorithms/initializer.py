import logging

from label_shift_study.algorithms.BN_adapt import BN_adapt
from label_shift_study.algorithms.BN_adapt_adv import BN_adapt_adv
from label_shift_study.algorithms.CDAN import CDAN
from label_shift_study.algorithms.COAL import COAL
from label_shift_study.algorithms.CORAL import CORAL
from label_shift_study.algorithms.DANN import DANN
from label_shift_study.algorithms.ERM import ERM
from label_shift_study.algorithms.ERM_Adv import ERM_Adv
from label_shift_study.algorithms.fixmatch import FixMatch
from label_shift_study.algorithms.noisy_student import NoisyStudent
from label_shift_study.algorithms.pseudolabel import PseudoLabel
from label_shift_study.algorithms.SENTRY import SENTRY
from label_shift_study.algorithms.TENT import TENT

logger = logging.getLogger("label_shift")

def initialize_algorithm(config, datasets, dataloader):

    logger.info(f"Initializing algorithm {config.algorithm} ...")

    source_dataset = datasets['source_train']
    trainloader_source = dataloader['source_train']

    # Other config
    n_train_steps = len(trainloader_source) * config.n_epochs // config.gradient_accumulation_steps

    if config.algorithm in ('ERM-rand', 'ERM-imagenet', 'ERM-clip', 'ERM-aug-rand' , 'ERM-aug-imagenet', 'ERM-swav', 'ERM-oracle-rand', 'ERM-oracle-imagenet', 
                            'IW-ERM-rand', 'IW-ERM-imagenet', 'IW-ERM-clip', 'IW-ERM-aug-rand', 'IW-ERM-aug-imagenet', 'IW-ERM-swav', 'IW-ERM-oracle-rand', 'IW-ERM-oracle-imagenet'):
        algorithm = ERM(
            config=config,
            dataloader = trainloader_source, 
            loss_function=config.loss_function,
            n_train_steps=n_train_steps)

    elif config.algorithm in ('ERM-adv'):
        algorithm = ERM_Adv(
            config=config,
            dataloader = trainloader_source, 
            loss_function=config.loss_function,
            n_train_steps=n_train_steps)

    elif config.algorithm in ('DANN', 'IW-DANN', 'IS-DANN'): 

        algorithm = DANN(
            config=config,
            dataloader = trainloader_source, 
            loss_function=config.loss_function,
            n_train_steps=n_train_steps, 
            n_domains=2,
            **config.dann_kwargs)

    elif config.algorithm in ('CDANN', 'IW-CDANN', 'IS-CDANN'):
        
        algorithm = CDAN(
            config=config,
            dataloader = trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            n_domains=2,
            **config.cdan_kwargs)

    elif config.algorithm in ('FixMatch', 'IS-FixMatch'):
        
        algorithm = FixMatch(
            config=config,
            dataloader = trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.fixmatch_kwargs)
        
    elif config.algorithm in ('PseudoLabel', 'IW-PseudoLabel'):

        algorithm = PseudoLabel(
            config=config,
            dataloader = trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.pseudolabel_kwargs)

    elif config.algorithm in ('NoisyStudent', 'IS-NoisyStudent'):

        algorithm = NoisyStudent(
            config=config,
            dataloader = trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.noisystudent_kwargs)
    
    elif config.algorithm in ('COAL', 'IW-COAL'):
        
        algorithm = COAL(
            config=config,
            dataloader = trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.coal_kwargs)

    elif config.algorithm in ('SENTRY', 'IW-SENTRY'):
        
        algorithm = SENTRY(
            config=config,
            dataloader = trainloader_source,
            loss_function=config.loss_function,
            n_train_steps=n_train_steps,
            **config.sentry_kwargs)

    elif config.algorithm in ('CORAL', 'IW-CORAL'):
        
        algorithm = CORAL(config=config)

    elif config.algorithm in ('BN_adapt', 'IS-BN_adapt'):
        
        algorithm = BN_adapt(config=config)

    elif config.algorithm in ('BN_adapt-adv', 'IS-BN_adapt-adv'):
        
        algorithm = BN_adapt_adv(config=config)

    elif config.algorithm in ('TENT', 'IS-TENT'):

        algorithm = TENT(config=config)

    else:
        raise ValueError(f"Algorithm {config.algorithm} not recognized")

    return algorithm

