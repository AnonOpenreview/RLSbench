algorithm_defaults = {
    'ERM': {
        'use_source': True,
        'use_target': False,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'save_every': 1
    },
    'ERM-aug': {
        'use_source': True,
        'use_target': False,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'additional_train_transform': 'randaugment', 
        'randaugment_n': 2,
        'save_every': 1
    },
    'ERM-oracle': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': True,
        'source_balanced': False,
        'additional_train_transform': 'randaugment',
        'randaugment_n': 2,
    },
    'IW-ERM': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
    },
    'IW-ERM-aug': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'additional_train_transform': 'randaugment',
        'randaugment_n': 2,
    },
    'IW-ERM-oracle': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': True,
        'source_balanced': False,
        'additional_train_transform': 'randaugment',
        'randaugment_n': 2,
    },
    'ERM-adv': {
        'use_source': True,
        'use_target': False,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'additional_train_transform': 'randaugment', 
        'randaugment_n': 2,
        'save_every': 1
    },
    # Domain Alignment methods 
    'DANN': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'save_every': 100
    },
    'IS-DANN': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'save_every': 100
    },
    'CDANN': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'save_every': 100
    },
    'IW-DANN': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'IW-CDANN': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },
    'IS-CDANN': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    },

    # Domain Alignment methods with Pseudo Labeling
    'COAL': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'coal_kwargs': { 
            'self_training_threshold': 0.7, 
            'alpha': 0.1,
        },
        'save_every': 100
    },
    'IW-COAL': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'coal_kwargs': { 
            'self_training_threshold': 0.7
        }
    },
    'SENTRY' : {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'scheduler': 'FixMatchLR',
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'sentry_kwargs': {
            'lambda_src': 1.0, 
            'lambda_unsup': 0.1,
            'lambda_ent': 1.0,
        },
        'save_every': 100
    },
    # 'IW-SENTRY' : {
    #     'use_source': True,
    #     'use_target': True,
    #     'use_unlabeled_y': False,
    #     'source_balanced': False,
    #     'randaugment_n': 2,
    #     'scheduler': 'FixMatchLR',
    #     'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
    #     'sentry_kwargs': {
    #         'lambda_src': 1.0, 
    #         'lambda_unsup': 0.1,
    #         'lambda_ent': 1.0,
    #     }
    # },

    # Pseudo Labeling Methods  
    'FixMatch': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'scheduler': 'FixMatchLR',
        'scheduler_kwargs': {},
        'fixmatch_kwargs': {
            'self_training_lambda': 1.0,
            'target_align': False,
            'self_training_threshold': 0.9, 
            },
        'save_every': 100
    },
    'IS-FixMatch': {
        'use_source': True, 
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'scheduler': 'FixMatchLR',
        'scheduler_kwargs': {},
        'fixmatch_kwargs': {
            'self_training_lambda': 1.0,
            'target_align': False,
            'self_training_threshold': 0.9, 
            }
    },
    'PseudoLabel': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples        
        'scheduler': 'FixMatchLR',
        'scheduler_kwargs': {},
        'pseudolabel_kwargs': {
            'self_training_lambda': 1.0,  
            'pseudolabel_T2': 0.4, 
            'self_training_threshold': 0.7, 
            'target_align': False, 
        },
        'save_every': 100
    },
    'IW-PseudoLabel': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'scheduler': 'FixMatchLR',
        'scheduler_kwargs': {},
        'pseudolabel_kwargs': {
            'self_training_lambda': 1.0,  
            'pseudolabel_T2': 0.4, 
            'self_training_threshold': 0.7, 
            'target_align': True, 
        },
        'save_every': 100
    },
    'NoisyStudent': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'scheduler': 'FixMatchLR',
        'scheduler_kwargs': {},
        'noisystudent_kwargs': {
            'noisystudent_dropout_rate': 0.5, 
            'noisystudent_add_dropout': True,
            'iterations': 2,
        },
        'save_every': 100
    },  
    'IS-NoisyStudent': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'randaugment_n': 2,
        'additional_train_transform': 'randaugment',     # Apply strong augmentation to labeled & unlabeled examples
        'scheduler': 'FixMatchLR',
        'scheduler_kwargs': {},
        'noisystudent_kwargs': {
            'noisystudent_dropout_rate': 0.5, 
            'noisystudent_add_dropout': True,
            'iterations': 2,
        }
    },

    # Test time adaptation methods
    'CORAL' : {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    },

    'IW-CORAL' : {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    },

    'BN_adapt': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    },
    'IS-BN_adapt': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    },
    'BN_adapt-adv': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    },
    'IS-BN_adapt-adv': {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    },
    'TENT' : {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    }, 
    'IS-TENT' : {
        'use_source': True,
        'use_target': True,
        'use_unlabeled_y': False,
        'source_balanced': False,
        'test_time_adapt': True,
        'use_source_model': True,
    }
}