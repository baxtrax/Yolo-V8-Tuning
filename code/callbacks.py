import wandb
def on_train_start(trainer):
    wandb.config.train = vars(trainer.args)

# Log metrics for learning rate, and "metrics" (mAP etc. and val losses)
def on_fit_epoch_end(trainer):
    wandb.log({**trainer.lr,
               **trainer.metrics})
    
# Log metrics for training loss
def on_train_batch_end(trainer):
    wandb.log({'train/box_loss': trainer.loss_items[0],
               'train/cls_loss': trainer.loss_items[1],
               'train/dfl_loss': trainer.loss_items[2]})
