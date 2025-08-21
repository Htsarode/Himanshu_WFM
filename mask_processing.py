# mask_processing.py
# Functions for processing mask images and extracting polygons/lanes

import numpy as np
from utils import get_node_token, extract_polygons_from_mask
import cv2
import uuid

def process_combined_mask(image_path, offset_x=0, offset_y=0, nodes=None, node_token_map=None):
    """
    Process a mask image and extract polygons and lanes.
    """
    
    mask = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"Could not load image {image_path}")

    if nodes is None:
        nodes = []
    if node_token_map is None:
        node_token_map = {}

    drivable_lower = np.array([200, 0, 0])
    drivable_upper = np.array([255, 50, 50])
    lane_lower = np.array([0, 200, 0])
    lane_upper = np.array([50, 255, 50])

    all_polygons = []
    drivable_areas, lanes = [], []

    polygons = extract_polygons_from_mask(mask, drivable_lower, drivable_upper, offset_x, offset_y, nodes, node_token_map)
    all_polygons.extend(polygons)
    for poly in polygons:
        drivable_areas.append({
            "token": str(uuid.uuid4()),
            "polygon_tokens": [poly["token"]]
        })

    polygons = extract_polygons_from_mask(mask, lane_lower, lane_upper, offset_x, offset_y, nodes, node_token_map)
    all_polygons.extend(polygons)
    for poly in polygons:
        lanes.append({
            "token": str(uuid.uuid4()),
            "polygon_token": poly["token"],
            "lane_type": "CAR",
            "from_edge_line_token": "",
            "to_edge_line_token": "",
            "left_lane_divider_segments": [],
            "right_lane_divider_segments": []
        })

    return {
        "polygon": all_polygons,
        "node": nodes,
        "drivable_area": drivable_areas,
        "lanes": lanes,
        "node_token_map": node_token_map
    }

