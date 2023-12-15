import wandb
import torch
import copy
from datetime import datetime
from ultralytics.utils.torch_utils import de_parallel
import ultralytics

try:
    import dill as pickle
except ImportError:
    import pickle

def on_train_start(trainer):
    wandb.config.train = vars(trainer.args)

# Log metrics for learning rate, and "metrics" (mAP etc. and val losses)
def on_fit_epoch_end(trainer):
    wandb.log({**trainer.lr,
               **trainer.metrics})
    
    if trainer.epoch % 10 == 0 or trainer.epoch == trainer.epochs:
        save_model(trainer)
    
# Log metrics for training loss
def on_train_batch_end(trainer):
    wandb.log({'train/box_loss': trainer.loss_items[0],
               'train/cls_loss': trainer.loss_items[1],
               'train/dfl_loss': trainer.loss_items[2]})
    
# Saves the model checkpoint every, stolen from Wandb's integration for yolo8
def save_model(trainer):
    current_args = vars(trainer.args)
    current_args['pc'] = trainer.pc

    model_checkpoint_artifact = wandb.Artifact(
        f"run_{wandb.run.id}_model", "model", metadata=current_args
    )

    checkpoint_dict = {
        "epoch": trainer.epoch,
        "best_fitness": trainer.best_fitness,
        "model": copy.deepcopy(de_parallel(trainer.model)).half(),
        "ema": copy.deepcopy(trainer.ema.ema).half(),
        "updates": trainer.ema.updates,
        "optimizer": trainer.optimizer.state_dict(),
        "train_args": current_args,
        "date": datetime.now().isoformat(),
        "version": ultralytics.__version__,
    }

    checkpoint_path = trainer.wdir / f"epoch{trainer.epoch}.pt"
    torch.save(checkpoint_dict, checkpoint_path, pickle_module=pickle)
    model_checkpoint_artifact.add_file(checkpoint_path)
    wandb.log_artifact(
        model_checkpoint_artifact, aliases=[f"epoch_{trainer.epoch}"]
    )
