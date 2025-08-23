import cv2
import numpy as np
import argparse
import os
import glob

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
        # Skip small contours - increased threshold to filter out very small lanes
        if cv2.contourArea(contour) < 500:  # Increased from 100 to 500 for bigger lanes only
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
    
    # Apply smoothing morphological operations for better polygon quality
    kernel = np.ones((5,5), np.uint8)  # Larger kernel for more smoothing
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_OPEN, kernel)
    
    # Additional Gaussian blur for smoother boundaries
    drivable_mask = cv2.GaussianBlur(drivable_mask, (7, 7), 0)
    
    # Re-threshold after blurring to get clean binary mask
    _, drivable_mask = cv2.threshold(drivable_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours with detailed approximation to preserve curves and breakages
    contours, _ = cv2.findContours(drivable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundaries = []
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 1000:
            continue
            
        # Use better approximation for smoother polygons
        # Increase epsilon for more smoothing and fewer vertices
        epsilon = 0.005 * cv2.arcLength(contour, True)  # Increased epsilon for smoother polygons
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        boundaries.append(polygon)
    
    return boundaries

def process_combined_images(lane_path, drivable_path, output_path):
    """Process lane and drivable area images and combine them"""
    # Load both images as color images
    lane_image = cv2.imread(lane_path, cv2.IMREAD_COLOR)
    drivable_image = cv2.imread(drivable_path, cv2.IMREAD_COLOR)
    
    if lane_image is None:
        print(f"Error: could not load lane image from {lane_path}")
        return
    if drivable_image is None:
        print(f"Error: could not load drivable area image from {drivable_path}")
        return
    
    # Check if images have the same dimensions
    if lane_image.shape != drivable_image.shape:
        print("Error: Images must have the same dimensions")
        print(f"Lane image shape: {lane_image.shape}")
        print(f"Drivable image shape: {drivable_image.shape}")
        return
    
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
    
    # Handle directory output path
    if os.path.isdir(output_path):
        output_file = os.path.join(output_path, "combined_result.png")
    else:
        output_file = output_path
    
    # Save result
    cv2.imwrite(output_file, result_image)
    print(f"Saved combined output to {output_file}")

def process_image(input_path, output_path):
    """Legacy function for single image processing"""
    # Load the segmentation mask image
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: could not load image from {input_path}")
        return

    # Create a black BGR image for drawing
    black_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Smooth the entire mask slightly to soften edges
    smoothed_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Get unique class IDs (excluding background)
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]

    for class_id in unique_classes:
        # Binary mask for this class
        binary_mask = (smoothed_mask == class_id).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Skip small contours
            if cv2.contourArea(contour) < 1000:  # adjust threshold as needed
                continue

            # Approximate and smooth the polygon
            epsilon = 0.0005 * cv2.arcLength(contour, True)  # slightly higher for more smoothing
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            # Draw the polygon boundary in blue
            cv2.polylines(black_image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    # Save result
    cv2.imwrite(output_path, black_image)
    print(f"Saved output to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process segmentation masks. Supports single image mode or dual image mode for combining lane and drivable area.")
    
    # Add mutually exclusive group for different modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", nargs=2, metavar=('INPUT', 'OUTPUT'), 
                      help="Single image mode: INPUT_PATH OUTPUT_PATH")
    group.add_argument("--dual", nargs=3, metavar=('LANE', 'DRIVABLE', 'OUTPUT'),
                      help="Dual image mode: LANE_PATH DRIVABLE_PATH OUTPUT_PATH")
    
    args = parser.parse_args()
    
    if args.single:
        input_path, output_path = args.single
        if os.path.isdir(input_path):
            # Folder mode: process all .png, .jpg, .jpeg files in the folder
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            image_extensions = ('*.png', '*.jpg', '*.jpeg')
            input_files = []
            for ext in image_extensions:
                input_files.extend(glob.glob(os.path.join(input_path, ext)))
            if not input_files:
                print(f"No image files found in directory: {input_path}")
                return
            for in_file in input_files:
                base_name = os.path.basename(in_file)
                out_file = os.path.join(output_path, os.path.splitext(base_name)[0] + "_smoothed.png")
                process_image(in_file, out_file)
        else:
            # Single image mode
            # If output_path is a directory, save output with original filename + _smoothed.png
            if os.path.isdir(output_path):
                base_name = os.path.basename(input_path)
                out_file = os.path.join(output_path, os.path.splitext(base_name)[0] + "_smoothed.png")
            else:
                out_file = output_path
            process_image(input_path, out_file)
    
    elif args.dual:
        lane_path, drivable_path, output_path = args.dual
        process_combined_images(lane_path, drivable_path, output_path)

if __name__ == "__main__":
    main()