"""
GIF Generation Script for Galen
Author: Siddhant Sharma, 2019
"""
__author__ = "Siddhant Sharma"


import argparse
from galen.utils import plot as plot_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate GIF for Galen training images")
    parser.add_argument("--dir", required = False, default = "./training/gif/", help = "Directory to training images")
    args = vars(parser.parse_args())
    print("Generating GIF...")
    plot_utils.generate_gif(args['dir'])
    print("Done!")
