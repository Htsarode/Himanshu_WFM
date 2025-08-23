
#!/usr/bin/env python3
"""
Simple test to verify the postprocessing server functions work independently
"""
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os

def process_lane_lines(image):
    """Process lane segmentation to extract centerlines from colored lanes"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define blue color range for lanes
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Also check for bright regions that might be lanes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Combine masks
    lane_mask = cv2.bitwise_or(blue_mask, bright_mask)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centerlines = []
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 100:
            continue
            
        # Create a mask for this specific lane
        temp_mask = np.zeros(lane_mask.shape, dtype=np.uint8)
        cv2.fillPoly(temp_mask, [contour], 255)
        
        # Use distance transform to find the centerline
        dist_transform = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
        
        # Find the skeleton points (points with maximum distance from boundary)
        # Get the ridge/skeleton by finding local maxima in distance transform
        skeleton_mask = np.zeros(temp_mask.shape, dtype=np.uint8)
        
        # Apply morphological thinning to get skeleton
        kernel = np.ones((3,3), np.uint8)
        temp = temp_mask.copy()
        
        # Iterative thinning to get skeleton
        while True:
            eroded = cv2.erode(temp, kernel)
            temp_opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            temp_subset = cv2.subtract(eroded, temp_opened)
            skeleton_mask = cv2.bitwise_or(skeleton_mask, temp_subset)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        # If skeleton is too sparse, use distance transform ridge
        if cv2.countNonZero(skeleton_mask) < 5:
            # Find points with high distance values (centerline)
            _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
            threshold = max_val * 0.8
            skeleton_mask = (dist_transform >= threshold).astype(np.uint8) * 255
        
        # Extract skeleton points and create a single line
        skeleton_points = np.where(skeleton_mask > 0)
        if len(skeleton_points[0]) > 2:
            # Convert to (x,y) format
            points = list(zip(skeleton_points[1], skeleton_points[0]))
            
            # Sort points to create a continuous line (simple approach)
            if len(points) > 1:
                # Find the two endpoints (points with only one neighbor)
                points_array = np.array(points)
                
                # Sort points by y-coordinate first, then x-coordinate
                sorted_indices = np.lexsort((points_array[:, 0], points_array[:, 1]))
                sorted_points = points_array[sorted_indices]
                
                # Create a simplified line by taking every nth point to avoid too dense lines
                step = max(1, len(sorted_points) // 20)  # Limit to about 20 points max
                simplified_points = sorted_points[::step]
                
                # Convert back to contour format
                centerline = simplified_points.reshape(-1, 1, 2).astype(np.int32)
                centerlines.append(centerline)
    
    return centerlines

def process_drivable_area(image):
    """Process drivable area to extract accurate boundaries without smoothing"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for drivable areas (green/orange)
    # Green range
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Orange range
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    drivable_mask = cv2.bitwise_or(green_mask, orange_mask)
    
    # NO smoothing - keep the raw mask to preserve accurate boundaries
    # Apply minimal morphological operations only to remove noise
    kernel = np.ones((2,2), np.uint8)
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours with detailed approximation to preserve curves and breakages
    contours, _ = cv2.findContours(drivable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    boundaries = []
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 1000:
            continue
            
        # Use minimal approximation to preserve the accurate shape
        # Reduce epsilon significantly to maintain more detail
        epsilon = 0.001 * cv2.arcLength(contour, True)  # Much smaller epsilon for accuracy
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        boundaries.append(polygon)
    
    return boundaries

def process_combined_images_server(lane_image_data, drivable_image_data):
    """Process lane and drivable area images and combine them for server"""
    # Convert image data to numpy arrays
    lane_image = np.array(Image.open(BytesIO(lane_image_data)))
    drivable_image = np.array(Image.open(BytesIO(drivable_image_data)))
    
    # Convert RGB to BGR for OpenCV
    if len(lane_image.shape) == 3 and lane_image.shape[2] == 3:
        lane_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2BGR)
    if len(drivable_image.shape) == 3 and drivable_image.shape[2] == 3:
        drivable_image = cv2.cvtColor(drivable_image, cv2.COLOR_RGB2BGR)
    
    # Check if images have the same dimensions
    if lane_image.shape != drivable_image.shape:
        print("Warning: Images have different dimensions, resizing...")
        # Resize drivable image to match lane image
        drivable_image = cv2.resize(drivable_image, (lane_image.shape[1], lane_image.shape[0]))
    
    # Create a black BGR image for drawing
    height, width = lane_image.shape[:2]
    result_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process lane lines (convert colored lane regions to single lines)
    print("Processing lane lines...")
    lane_lines = process_lane_lines(lane_image)
    print(f"Found {len(lane_lines)} lane lines")
    
    # Draw lane lines in green (thin centerlines)
    for line in lane_lines:
        if len(line) > 1:
            cv2.polylines(result_image, [line], isClosed=False, color=(0, 255, 0), thickness=3)
    
    # Process drivable area boundaries
    print("Processing drivable area boundaries...")
    drivable_boundaries = process_drivable_area(drivable_image)
    print(f"Found {len(drivable_boundaries)} drivable area boundaries")
    
    # Draw drivable area boundaries in blue
    for boundary in drivable_boundaries:
        cv2.polylines(result_image, [boundary], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Convert back to RGB for PIL
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image

def test_server_function():
    """Test the server processing function"""
    try:
        # Test loading images
        lane_path = "1616102501299_lanes.png"
        drivable_path = "1616102501299_drivableArea.png"
        
        if not os.path.exists(lane_path) or not os.path.exists(drivable_path):
            print("❌ Test images not found!")
            return False
            
        # Read images as binary data (simulating upload)
        with open(lane_path, 'rb') as f:
            lane_data = f.read()
        
        with open(drivable_path, 'rb') as f:
            drivable_data = f.read()
        
        print(f"✅ Lane image size: {len(lane_data)} bytes")
        print(f"✅ Drivable image size: {len(drivable_data)} bytes")
        
        # Test the processing function
        result_image = process_combined_images_server(lane_data, drivable_data)
        
        print(f"✅ Result image shape: {result_image.shape}")
        
        # Save the result
        pil_image = Image.fromarray(result_image)
        pil_image.save("server_test_result.png")
        
        print("✅ Test completed successfully! Check server_test_result.png")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_server_function()
    if success:
        print("\n🎉 All server function tests passed!")
        print("The postprocessing integration should work correctly.")
    else:
        print("\n💥 Server function tests failed!")
