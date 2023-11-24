import yaml
import argparse
from ultralytics import YOLO
from trainers import PGTrainer
import os

def main():
    args = setup_argparser()

    # Load config file
    if args.train is not None:
        with open(args.train, 'r') as f:
            train_cfg = yaml.safe_load(f)
        train_args = train_cfg.get('train_args', {})
        custom_args = train_cfg.get('custom_args', {})

    trainer = PGTrainer(pc=custom_args.get('pc', 0.1), 
                        overrides=train_args)
    trainer.train()

    # model.train(data=train_cfg['data'], trainer=PGTrainer, name='drone_real_train', **train_args)



def setup_argparser():
    """
    Sets up the argument parser for the main script.

    Returns:
        args: The arguments parsed from the command line.
    """
    parser = argparse.ArgumentParser(description='Research Setup and Configuration')
    parser.add_argument('--cfg_overall', '-co', type=str, help='Path to overall config file')
    parser.add_argument('--train', '-t', type=str, help='Train the model with path to config file')
    parser.add_argument('--val', '-v', type=str, help='Validate the model with path to config file')

    args = parser.parse_args()

        # Check for conditional requirements
    if not args.train and not args.val:
        parser.error("Either --train or --val must be specified")
    
    return args


if __name__ == '__main__':
    main()