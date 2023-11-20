# Reference for custom trainers
# https://docs.ultralytics.com/usage/engine/

# Main training loop in ultralytics is at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py#L282

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

class PGModel(DetectionModel):
    def loss(self, batch, preds=None):
        """
        Compute loss. Utilize PG loss here

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        # imgs = batch['img'].requires_grad = True # We need the img gradients for attribution
        imgs = batch['img']

        # self.zero_grad() # May not need?
        preds = self.forward(imgs) if preds is None else preds

        # preds[1] = torch.Size([16, 144, 80, 80]) [Batches, ??] (Head P5)
        # preds[0] = torch.Size([16, 144, 40, 40]) [Batches, ??] (Head P4)
        # preds[2] = torch.Size([16, 144, 20, 20]) [Batches, ??] (Head P3)
        
        # Loss is utilized at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py#L337
        loss = self.criterion(preds, batch) # Tuple (self.loss, self.loss_items)?
        return loss

    def calc_plausibility():
        pass


class PGTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        return PGModel()