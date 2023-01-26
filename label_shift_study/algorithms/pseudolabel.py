import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from label_shift_study.algorithms.single_model_algorithm import \
    SingleModelAlgorithm
from label_shift_study.losses import initialize_loss
from label_shift_study.models.initializer import initialize_model
from label_shift_study.models.model_utils import linear_probe
from label_shift_study.scheduler import LinearScheduleWithWarmupAndThreshold
from label_shift_study.utils import (concat_input, detach_and_clone, move_to,
                                     pseudolabel_multiclass_logits)

logger = logging.getLogger("label_shift")


class PseudoLabel(SingleModelAlgorithm):
    """
    PseudoLabel.
    This is a vanilla pseudolabeling algorithm which updates the model per batch and incorporates a confidence threshold.

    Original paper:
        @inproceedings{lee2013pseudo,
            title={Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks},
            author={Lee, Dong-Hyun and others},
            booktitle={Workshop on challenges in representation learning, ICML},
            volume={3},
            number={2},
            pages={896},
            year={2013}
            }
    """
    def __init__(self, config, dataloader, loss_function, n_train_steps, **kwargs):

        logger.info("Initializing PseudoLabel models")

        model = initialize_model(
            model_name = config.model, 
            dataset_name = config.dataset,
            num_classes = config.num_classes,
            featurize = True, 
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
        )

        if config.algorithm.startswith("IW"):
            self.use_target_marginal = True
        else:
            self.use_target_marginal = False

        if config.source_balanced or self.use_target_marginal:
            loss = initialize_loss(loss_function, reduction='none')
        else:
            loss = initialize_loss(loss_function)

        if config.pretrained : 
            model = linear_probe(model, dataloader, device= config.device, progress_bar=config.progress_bar)

        model = nn.Sequential(*model)
        
        # initialize module
        super().__init__(
            config=config,
            model=model,
            loss=loss,
            n_train_steps=n_train_steps,
        )

        # algorithm hyperparameters
        self.lambda_scheduler = LinearScheduleWithWarmupAndThreshold(
            max_value=kwargs['self_training_lambda'],
            step_every_batch=True, # step per batch
            last_warmup_step=0,
            threshold_step=kwargs['pseudolabel_T2'] * n_train_steps
        )

        self.schedulers.append(self.lambda_scheduler)
        self.confidence_threshold = kwargs['self_training_threshold']
        self.target_align = kwargs['target_align']
        self.process_pseudolabels_function = pseudolabel_multiclass_logits

        self.source_balanced = config.source_balanced
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
                - unlabeled_y_pseudo (Tensor): pseudolabels on the unlabeled batch, already thresholded 
                - unlabeled_y_pred (Tensor): model output on the unlabeled batch, already thresholded 
        """
        # Labeled examples
        x, y_true = batch[:2]
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        n_lab = y_true.shape[0]

        # package the results
        results = {
            'y_true': y_true,
        }

        if self.use_target_marginal and target_marginal is not None:
            results['im_weights'] = torch.divide(torch.tensor(target_marginal).to(self.device),\
                    torch.tensor(source_marginal).to(self.device))

        if self.source_balanced and source_marginal is not None:
            results['source_marginal'] = torch.tensor(source_marginal).to(self.device)


        if unlabeled_batch is not None:
            if self.target_align and target_average is not None: 
                alignment_dist = torch.divide(torch.tensor(target_marginal).to(self.device), torch.tensor(target_average).to(self.device))

            x_unlab = unlabeled_batch[0]
            x_unlab = move_to(x_unlab, self.device)

            x_cat = concat_input(x, x_unlab)
            outputs = self.get_model_output(x_cat)
            unlabeled_output = outputs[n_lab:]
            
            if self.target_align and target_average is not None :
                unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, _ = self.process_pseudolabels_function(
                    unlabeled_output,
                    self.confidence_threshold, 
                    alignment_dist
                )
            else:
                unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, _ = self.process_pseudolabels_function(
                    unlabeled_output,
                    self.confidence_threshold
                )

            results['y_pred'] = outputs[:n_lab]
            results['unlabeled_y_pred'] = unlabeled_y_pred
            results['unlabeled_y_pseudo'] = detach_and_clone(unlabeled_y_pseudo)
        else:
            results['y_pred'] = self.get_model_output(x)
            pseudolabels_kept_frac = 0

        results['pseudolabels_kept_frac'] = pseudolabels_kept_frac        

        return results

    def objective(self, results):

        # Labeled loss
        classification_loss = self.loss(
            results['y_pred'], results['y_true']
        )

        if self.use_target_marginal: 
            classification_loss = torch.mean(classification_loss*results["im_weights"][results["y_true"]])

        elif self.source_balanced: 
            classification_loss = torch.mean(classification_loss/results["source_marginal"][results["y_true"]]/ self.num_classes)


        # Pseudolabeled loss
        if 'unlabeled_y_pseudo' in results:
            loss_output = self.loss(
                results['unlabeled_y_pred'],
                results['unlabeled_y_pseudo'],
            )

            if self.use_target_marginal or self.source_balanced:
                loss_output = torch.mean(loss_output)

            consistency_loss = loss_output * results['pseudolabels_kept_frac']
        else:
            consistency_loss = 0


        return classification_loss + self.lambda_scheduler.value * consistency_loss
