
# main.py
# Refactored script for mask processing and JSON generation
# Organized into functions and uses argparse for input arguments

import cv2
import numpy as np
import uuid

import os
import argparse
from scene_processor import process_scene

def main():
        parser = argparse.ArgumentParser(description="Process mask images and generate JSON files.")
        parser.add_argument('--input', type=str, required=True, help='Input root directory containing scenes')
        parser.add_argument('--output', type=str, required=True, help='Output directory for JSON files')
        args = parser.parse_args()
        process_scene(args.input, args.output)

if __name__ == "__main__":
    main()
    # Extract lanes
