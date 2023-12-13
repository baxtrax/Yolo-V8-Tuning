from ultralytics.models.yolo.detect import DetectionTrainer
from plausbility_functions import evaluate_plausability
from ultralytics.utils import (RANK, DEFAULT_CFG)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.model import Model
from ultralytics.models import yolo
import wandb
import torch

import numpy as np
import matplotlib.pyplot as plt

def imshow(img, save_path=None):
    try:
        npimg = img.cpu().detach().numpy()
    except:
        npimg = img
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg)
    if save_path != None:
        plt.savefig(str(str(save_path) + ".png"))

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
            current = wandb.config.train
            current['pc'] = self.pc
            wandb.config.update({'train': current})

        # Make imgs require gradients so we can use gradients for attribution
        imgs = batch['img'].requires_grad_(self.training and not self.pc == 0)

        # Compute loss
        preds = self.forward(imgs) if preds is None else preds
        loss = self.criterion(preds, batch)

        if self.training and not self.pc == 0: # Can only compute gradients during training, not validation
            p_loss = self.get_plausbility_loss(preds, batch)
            p_loss *= self.pc # Weight plausbility loss

            imgs.requires_grad = False # should this be true?

            if RANK in (-1, 0):
                wandb.log({'train/plausbility_loss': p_loss})


            # Undo batch scaling, add in new loss, and reapply batch scaling
            # Ideally this would actually be a part of the overall loss function
            # This was quicker to implement though
            batch_size = imgs.shape[0]
            loss = (((loss[0]/batch_size) + p_loss)*batch_size, loss[1])

        return loss

    #plt.show()

    def get_plausbility_loss(self, preds, batch, debug=False):
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
                                        retain_graph=True)[0].detach().float().abs()
        
        # Rearange targets in batch index groups
        # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['bboxes']), 1)  
        # targets = self.process_targets(targets, gradients)

        target_bboxes = batch['bboxes']        

        # Swap to xyxy from v8Detectionloss?
        def corners_coords(center_xywh):
            center_x, center_y, w, h = torch.unbind(center_xywh, dim=1)
            x = center_x - w/2
            y = center_y - h/2
            return torch.stack([x, y, x+w, y+h], dim=1)

        img_height, img_width = batch['img'].shape[2:]
        # Convert to xyxy, scale to image size, and round down to nearest integer
        xyxy_scaled = (corners_coords(target_bboxes) * torch.tensor([img_width, 
                                                                    img_height, 
                                                                    img_width, 
                                                                    img_height])).int()

        target_idxes = batch['batch_idx'].int()
        
        coords_map = torch.zeros_like(gradients, dtype=torch.bool)
        # rows = np.arange(co.shape[0])
        x1, x2 = xyxy_scaled[:,1], xyxy_scaled[:,3]
        y1, y2 = xyxy_scaled[:,0], xyxy_scaled[:,2]
        
        for ic in range(xyxy_scaled.shape[0]): # potential for speedup here with torch indexing instead of for loop
            coords_map[target_idxes[ic], :,x1[ic]:x2[ic],y1[ic]:y2[ic]] = True

        # debug=True
        if debug:
            for i in range(len(coords_map)):
                # coords_map3ch = torch.cat([coords_map[i], coords_map[i], coords_map[i]], dim=0)
                test_bbox = torch.zeros_like(batch['img'][i])
                test_bbox[coords_map[i]] = batch['img'][i][coords_map[i]]
                imshow(test_bbox, save_path='figs/test_bbox')
                imshow(batch['img'][i], save_path='figs/im0')
                imshow(gradients[i], save_path='figs/attr')
        
        IoU_num = (torch.sum(gradients[coords_map]))
        IoU_denom = torch.sum(gradients)
        IoU = (IoU_num / IoU_denom)
        plausability_scores = IoU
        
        return 1-plausability_scores # Return mean plausbility score for batch

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
        # Converts from idx, xyxy to grousp of xyxy per idx
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
