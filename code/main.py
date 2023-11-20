import yaml
import argparse
from ultralytics import YOLO
from trainers import PGTrainer
import os

def main():
    args = setup_argparser()

    # Print current working directory
    print('Current working directory: ', os.getcwd())
    print(args.train)

    # Load config files
    if args.train is not None:
        with open(args.train, 'r') as f:
            train_cfg = yaml.safe_load(f)
        train_args = train_cfg.get('args', {})

    if args.val is not None:
        with open(args.val, 'r') as f:
            val_cfg = yaml.safe_load(f)
        val_args = val_cfg.get('args', {})

    model = YOLO('yolov8n.pt')

    model.train(data=train_cfg['data'], trainer=PGTrainer, name='drone_real_train', **train_args) 
    # model.val(data=val_cfg['data'], name='drone_real_val', **val_args)


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