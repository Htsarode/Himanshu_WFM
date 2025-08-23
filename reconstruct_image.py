import cv2
import json
import numpy as np
import argparse

def reconstruct_polygon(polygon_entry, nodes_dict):
    """Reconstruct polygon (exterior + holes) from node tokens."""
    exterior = [ (nodes_dict[token]["x"], nodes_dict[token]["y"]) for token in polygon_entry["exterior_node_tokens"] ]
    holes = []
    for hole in polygon_entry["holes"]:
        hole_coords = [ (nodes_dict[token]["x"], nodes_dict[token]["y"]) for token in hole["node_tokens"] ]
        holes.append(hole_coords)
    return np.array(exterior, dtype=np.int32), [np.array(h, dtype=np.int32) for h in holes]

def extract_single_image(json_path, image_shape=(1024, 1024, 3)):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build a node dictionary for easy lookup
    nodes_dict = {node["token"]: node for node in data["node"]}

    # Create blank image
    img = np.zeros(image_shape, dtype=np.uint8)

    # ---- Draw drivable areas ----
    for da in data["drivable_area"]:
        for poly_token in da["polygon_tokens"]:
            polygon_entry = next(p for p in data["polygon"] if p["token"] == poly_token)
            exterior, holes = reconstruct_polygon(polygon_entry, nodes_dict)

            cv2.fillPoly(img, [exterior], (255, 0, 0))   # Blue
            for hole in holes:
                cv2.fillPoly(img, [hole], (0, 0, 0))     # Carve hole

    # ---- Draw lanes ----
    for lane in data["lane"]:
        polygon_entry = next(p for p in data["polygon"] if p["token"] == lane["polygon_token"])
        exterior, holes = reconstruct_polygon(polygon_entry, nodes_dict)

        cv2.fillPoly(img, [exterior], (0, 255, 0))       # Green
        for hole in holes:
            cv2.fillPoly(img, [hole], (0, 0, 0))

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct image from JSON annotation.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output image')
    parser.add_argument('--image_shape', type=int, nargs=3, default=[1024, 1902, 3], help='Shape of output image (height width channels)')
    args = parser.parse_args()

    output_img = extract_single_image(args.json_path, image_shape=tuple(args.image_shape))
    cv2.imwrite(args.output_path, output_img)
    print(f"Saved reconstructed image as {args.output_path}")