from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import (RANK, DEFAULT_CFG)
from plausbility_functions import vanilla_grad_attributions
import torch

# ISSUES:
# During validation, cant compute gradients not matter what I do.

class PGModel(DetectionModel):
    def loss(self, batch, preds=None):
        """
        Custom loss function that utilizes a plausbility loss

        Args:
            batch (dict): batch of data
            preds (tuple, optional): tuple of predictions. Defaults to None.

        Returns:
            loss: loss value
        """

        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        imgs = batch['img'].requires_grad_(self.training)

        preds = self.forward(imgs) if preds is None else preds
        loss = self.criterion(preds, batch)

        if self.training: # Can only compute gradients during training, not validation
            p_loss = self.get_plausbility_loss(preds, batch)
            imgs.requires_grad = False


        return loss
    
    def get_plausbility_loss(self, preds, batch):
        pred_scores = self.get_pred_scores(preds)

        gradients = torch.autograd.grad(pred_scores, 
                                        batch['img'], 
                                        grad_outputs=torch.ones_like(pred_scores),
                                        retain_graph=True)[0].detach().float()

        # Targets has more then 1 bbox for each image
        # So run through each list of batch targets and average their plausbility scores
        
        print('NAN?: ', torch.isnan(gradients).any().item())



        # Sum across color channels
    
    def get_pred_scores(self, preds):
        # This is ripped from the v8DetectionLoss
        feats = preds[1] if isinstance(preds, tuple) else preds
        _, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.model[-1].no, -1) for xi in feats], 2).split(
            (self.model[-1].reg_max * 4, self.model[-1].nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        return pred_scores
    
    
class PGTrainer(DetectionTrainer):
    """
        A class extending the DetectTrainer class for plausbility guided training.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, pc=0.1, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.pc = pc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = PGModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model