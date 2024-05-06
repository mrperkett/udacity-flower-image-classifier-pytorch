#! /usr/bin/env python3
import argparse
import logging
import os.path

import torch
from torch import nn, optim

from utils import train


def parse_args():
    """
    Parse command line arguments.
    """
    # TODO: modify hidden_units to allow a list of integers to be passed
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", type=str,
                        help="Path to the directory containing the 'train', 'valid', and 'test' folders containing the jpeg images")
    parser.add_argument("--save_dir", type=str, default=os.getcwd(), required=False,
                        help="Path to the directory in which to save the model checkpoint file when training completes.")
    parser.add_argument("--arch", type=str, default="vgg13", required=False,
                        choices=["alexnet", "densenet121", "resnet18", "vgg13", "vgg16"],
                        help="Pretrained model architecture from which to start training")
    parser.add_argument("--learning_rate", type=float, default=0.01, required=False,
                        help="Learning rate during model training")
    parser.add_argument("--hidden_units", type=int, default=512, required=False,
                        help="Number of units in the hidden layer")
    parser.add_argument("--epochs", type=int, default=1, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--dropout", type=float, default=None, required=False,
                        help="Dropout rate for hidden layer during training (in range (0.0, 1.0))")
    parser.add_argument("--gpu", default=False, action="store_true",
                        help="Train using GPU")
    parser.add_argument("--category_names", type=str, default=None, required=False,
                        help="JSON input file specifying the category label to class label (e.g. '1': 'pink primrose')")
    args = parser.parse_args()
    
    # set checkpoint_file_path based on args.save_dir
    args.checkpoint_file_path = os.path.join(args.save_dir, "checkpoint.pt")

    return args


def main():
    """
    Main script
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # get command line arguments and train the model
    args = parse_args()
    train(args)
    
    return


if __name__ == "__main__":
    main()