from ultralytics.models.yolo.detect import DetectionTrainer
from plausbility_functions import evaluate_plausability
from ultralytics.utils import (RANK, DEFAULT_CFG)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.model import Model
from ultralytics.models import yolo
import wandb
import torch

# ISSUES:
# During validation, cant compute gradients not matter what I do.

class PGTYOLO(Model):
    """
    PGT modified YOLO (You Only Look Once) object detection model. Only
    supports detection task.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        return {
            'detect': {
                'model': PGModel,
                'trainer': PGTrainer,
                'validator': yolo.detect.DetectionValidator, # Replace with PGValidator
                'predictor': yolo.detect.DetectionPredictor, }}

class PGModel(DetectionModel):
    """
        YOLOv8 detection model that utlizes Plausbility Guided Training
    """
    def __init__(self, pc, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        self.pc = pc
        super().__init__(cfg, ch, nc, verbose)


    def loss(self, batch, preds=None):
        """
        Custom loss function that utilizes a plausbility loss
        """

        # Initialize criterion if not already done
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
            # wandb.run.config.train.update({'pc': self.pc})

        if isinstance(self.pc, float):
            self.pc = torch.tensor(self.pc).to(batch['img'].device)

        # Make imgs require gradients so we can use gradients for attribution
        imgs = batch['img'].requires_grad_(self.training and not self.pc == 0)

        # Compute loss
        preds = self.forward(imgs) if preds is None else preds
        loss = self.criterion(preds, batch)

        if self.training and not self.pc == 0: # Can only compute gradients during training, not validation
            p_loss = self.get_plausbility_loss(preds, batch)
            wandb.log({'train/plausbility_loss': p_loss})
            imgs.requires_grad = False
            loss = (loss[0] - (self.pc * p_loss), loss[1])

        return loss
    

    def get_plausbility_loss(self, preds, batch):
        """
        Get the plausbility loss for a batch of images and predictions

        Args:
            preds: The predictions from the model
            batch: The batch of images and targets

        Returns:
            plausbility_loss: The plausbility loss for the batch
        """

        # Parse preds to get pred scores
        pred_scores = self.get_pred_scores(preds)

        # Compute gradients of pred scores w.r.t. images
        gradients = torch.autograd.grad(pred_scores, 
                                        batch['img'], 
                                        grad_outputs=torch.ones_like(pred_scores),
                                        retain_graph=True)[0].detach().float()
        
        # Rearange targets in batch index groups
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['bboxes']), 1)  
        targets = self.process_targets(targets, gradients)

        # Compute plausbility scores per batch index
        plausability_scores = []
        for gradient, target in zip(gradients, targets):
            plausability_scores.append(evaluate_plausability(gradient.unsqueeze(0), target).mean())
        plausability_scores = torch.stack(plausability_scores)
        
        return plausability_scores.mean() # Return mean plausbility score for batch

    def get_pred_scores(self, preds):
        # This is ripped from the v8DetectionLoss
        feats = preds[1] if isinstance(preds, tuple) else preds
        _, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.model[-1].no, -1) for xi in feats], 2).split(
            (self.model[-1].reg_max * 4, self.model[-1].nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        return pred_scores
    

    def process_targets(self, targets, gradients):
        """
        Rearange targets in batch index groups
        """
        # TODO: This is a hacky way to do this, but it works for now
        # Convert this to batch tensor operations
        processed_targets = []

        for j in torch.unique(targets[:, 0]):
            filtered_targets = []
            for i in range(len(targets)):
                if int(targets[i][0]) == j:
                    filtered_targets.append(targets[i][-4:].to(gradients[0].device))
            processed_targets.append(torch.stack(filtered_targets))

        return processed_targets

    
class PGTrainer(DetectionTrainer):
    """
        A class extending the DetectTrainer class for plausbility guided training.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # Get pc from cfg and remove from cfg
        if 'pc' in overrides:
            self.pc = overrides['pc']
            del overrides['pc']
        else:
            self.pc = 0.1

        super().__init__(cfg, overrides, _callbacks)



    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = PGModel(cfg=cfg, pc=self.pc, nc=self.data['nc'], verbose=verbose and RANK == -1)
        
        if weights:
            model.load(weights)

        return model
