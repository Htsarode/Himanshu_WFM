# utils.py
# Utility functions for mask processing

import uuid
import numpy as np


def get_node_token(point, nodes, node_token_map, offset_x=0, offset_y=0):
    """
    Generate a unique token for a node (point) and store it in the node list and token map.
    """
    key = (point[0] + offset_x, point[1] + offset_y)
    if key not in node_token_map:
        token = str(uuid.uuid4())
        node_entry = {
            "token": token,
            "x": round(float(point[0] + offset_x), 2),
            "y": round(float(point[1] + offset_y), 2)
        }
        nodes.append(node_entry)
        node_token_map[key] = token
    return node_token_map[key]


def extract_polygons_from_mask(mask, color_lower, color_upper, offset_x, offset_y, nodes, node_token_map):
    """
    Extract polygons (and holes) for a given color range from the mask image.
    """
    import cv2
    binary_mask = cv2.inRange(mask, color_lower, color_upper)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    polygons = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # only process exterior contours
            if color_lower[1] > 0:  # lane (green)
                epsilon = 0.01 * cv2.arcLength(contour, True)
            else:  # drivable (red)
                epsilon = 0.02 * cv2.arcLength(contour, True)

            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_contour) < 4:
                approx_contour = contour

            exterior_tokens = [
                get_node_token(pt[0], nodes, node_token_map, offset_x, offset_y)
                for pt in approx_contour
            ]

            holes_list = []
            child_index = hierarchy[0][i][2]
            while child_index != -1:
                hole_contour = contours[child_index]
                epsilon_hole = 0.02 * cv2.arcLength(hole_contour, True)
                approx_hole = cv2.approxPolyDP(hole_contour, epsilon_hole, True)
                hole_tokens = [
                    get_node_token(pt[0], nodes, node_token_map, offset_x, offset_y)
                    for pt in approx_hole
                ]
                holes_list.append({"node_tokens": hole_tokens})
                child_index = hierarchy[0][child_index][0]

            polygon_token = str(uuid.uuid4())
            polygons.append({
                "token": polygon_token,
                "exterior_node_tokens": exterior_tokens,
                "holes": holes_list
            })

    return polygons
