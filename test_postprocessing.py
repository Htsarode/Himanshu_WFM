#!/usr/bin/env python3
"""
Test script to verify the postprocessing functions work correctly
"""
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Import the postprocessing functions from server.py
import sys
import os

# Add current directory to path to import from server.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_postprocessing_functions():
    """Test the postprocessing functions independently"""
    try:
        # Test loading images
        lane_path = "1616102501299_lanes.png"
        drivable_path = "1616102501299_drivableArea.png"
        
        if not os.path.exists(lane_path) or not os.path.exists(drivable_path):
            print("Test images not found!")
            return False
            
        # Read images as binary data (simulating upload)
        with open(lane_path, 'rb') as f:
            lane_data = f.read()
        
        with open(drivable_path, 'rb') as f:
            drivable_data = f.read()
        
        print(f"Lane image size: {len(lane_data)} bytes")
        print(f"Drivable image size: {len(drivable_data)} bytes")
        
        # Test the processing functions
        from server import process_combined_images_server
        
        result_image = process_combined_images_server(lane_data, drivable_data)
        
        print(f"Result image shape: {result_image.shape}")
        
        # Save the result
        pil_image = Image.fromarray(result_image)
        pil_image.save("test_result.png")
        
        print("Test completed successfully! Check test_result.png")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_postprocessing_functions()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")