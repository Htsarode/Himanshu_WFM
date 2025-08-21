# scene_processor.py
# Functions for processing scenes and saving JSON output

import os
import json
from mask_processing import process_combined_mask


def process_scene(root_dir, output_dir):
    """
    Process all scenes in the root directory and save results as JSON files.
    """
    for split in os.listdir(root_dir):
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for scene_folder in os.listdir(split_path):
            scene_path = os.path.join(split_path, scene_folder)
            if not os.path.isdir(scene_path):
                continue

            for cam_name in os.listdir(scene_path):
                cam_path = os.path.join(scene_path, cam_name)
                if not os.path.isdir(cam_path):
                    continue

                cam_data = {
                    "version": "1.0",
                    "polygon": [],
                    "line": [],
                    "node": [],
                    "drivable_area": [],
                    "road_segment": [],
                    "road_block": [],
                    "lane": [],
                    "ped_crossing": [],
                    "walkway": [],
                    "stop_line": [],
                    "carpark_area": [],
                    "road_divider": [],
                    "lane_divider": [],
                    "traffic_light": [],
                    "canvas_edge": [],
                    "connectivity": [],
                    "arcline_path_3": [],
                    "lane_connector": []
                }

                nodes = []
                node_token_map = {}
                offset_x = 0
                offset_y = 0
                max_height = 0

                for idx, image in enumerate(sorted(os.listdir(cam_path))):
                    if not image.lower().endswith((".png", ".jpg")):
                        continue

                    image_path = os.path.join(cam_path, image)
                    import cv2
                    mask = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if mask is None:
                        print(f"Could not read {image_path}")
                        continue

                    try:
                        result = process_combined_mask(image_path, offset_x, offset_y, nodes, node_token_map)
                        cam_data["polygon"].extend(result["polygon"])
                        cam_data["drivable_area"].extend(result["drivable_area"])
                        cam_data["lane"].extend(result["lanes"])
                        offset_x += mask.shape[1] + 10
                        max_height = max(max_height, mask.shape[0])
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

                cam_data["node"] = nodes
                output_directory = os.path.join(output_dir, split, scene_folder)
                os.makedirs(output_directory, exist_ok=True)
                output_file = os.path.join(output_directory, f"{cam_name}.json")
                with open(output_file, "w") as f:
                    json.dump(cam_data, f, indent=2)
                print(f"Saved JSON for scene {scene_folder} -> {output_file}")
