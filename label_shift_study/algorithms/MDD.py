import logging
from typing import Dict, List

from label_shift_study.algorithms.single_model_algorithm import \
    SingleModelAlgorithm
from label_shift_study.losses import initialize_loss
from label_shift_study.models.initializer import initialize_model
from label_shift_study.models.mdd_net import MDDNet, get_mdd_loss
from label_shift_study.optimizer import initialize_optimizer_with_model_params
from label_shift_study.utils import concat_input

logger = logging.getLogger("label_shift")


class MDD(SingleModelAlgorithm):
    """
    Domain-adversarial training of neural networks.

    Original paper:
        @inproceedings{dann,
          title={Domain-Adversarial Training of Neural Networks},
          author={Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand and Lempitsky},
          booktitle={Journal of Machine Learning Research 17},
          year={2016}
        }
    """

    def __init__( self, config, dataloader, loss_function, n_train_steps,  **kwargs):
        
        logger.info("Initializing MDD models")

        self.use_target_marginal = False

        loss = initialize_loss(loss_function)

        # Initialize model
        featurizer, classifier = initialize_model(
            model_name = config.model, 
            dataset_name = config.dataset,
            num_classes = config.num_classes,
            featurize = True, 
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
        )

        model = MDDNet(featurizer, config.num_classes)
        
        parameters_to_optimize =  model.get_parameter_list()
        
        self.optimizer = initialize_optimizer_with_model_params(config, parameters_to_optimize)

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )

        # Algorithm hyperparameters
        self.srcweight = kwargs["srcweight"]
        self.num_classes = config.num_classes

    def process_batch(self, batch, unlabeled_batch=None, target_marginal=None, source_marginal = None, target_average=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - y_pred (Tensor): model output for batch 
                - domains_true (Tensor): true domains for batch and unlabeled batch
                - domains_pred (Tensor): predicted domains for batch and unlabeled batch
        """
        # Forward pass
        x, y_true, = batch[:2]
        
        if unlabeled_x is not None:
            unlabeled_x = unlabeled_batch[:1]

            # Concatenate examples and true domains
            x_cat = concat_input(x, unlabeled_x)
            _ , y_pred, _, y_pred_adv = self.model(x_cat)
            

            return {
                "y_true": y_true,
                "y_pred": y_pred, 
                "y_pred_adv": y_pred_adv
            }
        else: 
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            _, y_pred, _, _ = self.model(x)

            return {
                "y_true": y_true,
                "y_pred": y_pred,
            }

    def objective(self, results):
        
        return get_mdd_loss(results["y_pred"], results["y_pred_adv"], results["y_true"], self.loss, self.srcweight)
