from fastapi import FastAPI, status, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_2.build_sam import build_sam2, build_sam2_video_predictor
from segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator


import os
import cv2
import time
import torch
import shutil
import zipfile
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from base64 import b64encode, b64decode
import glob

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

# Postprocessing functions from postprocessing.py
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

def process_combined_mask_server(mask_image_data):
    """Process a single combined mask containing both lanes and drivable areas"""
    # Convert image data to numpy array
    mask_image = np.array(Image.open(BytesIO(mask_image_data)))
    
    print(f"Mask image shape: {mask_image.shape}")
    print(f"Mask image dtype: {mask_image.dtype}")
    
    # Ensure the image is in BGR format with 3 channels
    if len(mask_image.shape) == 2:
        # Grayscale image, convert to BGR
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    elif len(mask_image.shape) == 3:
        if mask_image.shape[2] == 4:
            # RGBA image, convert to BGR
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGR)
        elif mask_image.shape[2] == 3:
            # RGB image, convert to BGR
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
    
    print(f"Processed mask image shape: {mask_image.shape}")
    
    # Create a black BGR image for drawing the result
    height, width = mask_image.shape[:2]
    result_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Extract lane annotations (blue color: RGB 0,128,255 -> BGR 255,128,0)
    # Use exact color matching for the specific colors used in frontend
    lane_color_bgr = np.array([255, 128, 0], dtype=np.uint8)  # Exact blue color in BGR
    lane_mask = cv2.inRange(mask_image, lane_color_bgr, lane_color_bgr)
    
    # Extract drivable area annotations (orange color: RGB 255,128,0 -> BGR 0,128,255)  
    drivable_color_bgr = np.array([0, 128, 255], dtype=np.uint8)  # Exact orange color in BGR
    drivable_mask = cv2.inRange(mask_image, drivable_color_bgr, drivable_color_bgr)
    
    # If exact matching doesn't work, try with small tolerance
    if np.sum(lane_mask) == 0:
        lower_lane = np.array([250, 120, 0], dtype=np.uint8)
        upper_lane = np.array([255, 135, 5], dtype=np.uint8)
        lane_mask = cv2.inRange(mask_image, lower_lane, upper_lane)
        
    if np.sum(drivable_mask) == 0:
        lower_drivable = np.array([0, 120, 250], dtype=np.uint8)
        upper_drivable = np.array([5, 135, 255], dtype=np.uint8)
        drivable_mask = cv2.inRange(mask_image, lower_drivable, upper_drivable)
    
    print(f"Lane mask pixels found: {np.sum(lane_mask > 0)}")
    print(f"Drivable mask pixels found: {np.sum(drivable_mask > 0)}")
    
    # Process lane lines (convert colored lane regions to centerlines)
    print("Processing lane lines from combined mask...")
    if np.any(lane_mask):
        from postprocessing import process_lane_lines_from_mask
        lane_lines = process_lane_lines_from_mask(lane_mask)
        print(f"Found {len(lane_lines)} lane lines")
        
        # Draw lane lines in green (thin centerlines)
        for line in lane_lines:
            if len(line) > 1:
                cv2.polylines(result_image, [line], isClosed=False, color=(0, 255, 0), thickness=3)
    
    # Process drivable area boundaries
    print("Processing drivable area boundaries from combined mask...")
    if np.any(drivable_mask):
        from postprocessing import process_drivable_area_from_mask
        drivable_boundaries = process_drivable_area_from_mask(drivable_mask)
        print(f"Found {len(drivable_boundaries)} drivable area boundaries")
        
        # Draw drivable area boundaries in blue
        for boundary in drivable_boundaries:
            cv2.polylines(result_image, [boundary], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Convert back to RGB for PIL
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image

# device = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

use_sam2 = False
if not use_sam2:
    sam_checkpoint = "sam_vit_l_0b3195.pth" # "sam_vit_l_0b3195.pth" or "sam_vit_h_4b8939.pth"
    model_type = "vit_l" # "vit_l" or "vit_h"

    print("Loading model")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    print("Finishing loading")
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)
else:
    sam2_checkpoint = "sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model) # for single image
    
    vid_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint) # for video
    inference_state = None

    mask_generator = SAM2AutomaticMaskGenerator(sam2_model) # seg_everything


app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a palette for video segmentation
import random
palette = [random.randint(0, 255) for _ in range(256*3)]

input_point = []
input_label = []
masks = []

segmented_mask = []
interactive_mask = []
mask_input = None

GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_ZIPBUFFER = None

@app.post("/image")
async def process_images(
    image: UploadFile = File(...)
):
    global segmented_mask, interactive_mask
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER

    segmented_mask = []
    interactive_mask = []

    # Read the image and mask data as bytes
    image_data = await image.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None

    predictor.set_image(GLOBAL_IMAGE)

    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=200,
    )

from XMem import XMem, InferenceCore, image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.set_grad_enabled(False)

if not use_sam2:
    def seg_propagation(video_name, mask_name):
        # default configuration
        config = {
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 5,
            'max_mid_term_frames': 10,
            'max_long_term_elements': 10000,
        }

        network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

        im = Image.open(mask_name).convert('L')
        im.putpalette(palette)
        mask = np.array(im)
        acc = 0
        for i in range(256):
            if np.sum(mask==i) == 0:
                acc += 1
                mask[mask==i] -= acc-1
            else:
                mask[mask==i] -= acc
        print(np.unique(mask))
        num_objects = len(np.unique(mask)) - 1

        st = time.time()
        # torch.cuda.empty_cache()

        processor = InferenceCore(network, config=config)
        processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
        cap = cv2.VideoCapture(video_name)

        # You can change these two numbers
        frames_to_propagate = 1500
        visualize_every = 1

        current_frame_index = 0

        masked_video = []

        with torch.cuda.amp.autocast(enabled=True):
            while (cap.isOpened()):
                # load frame-by-frame
                _, frame = cap.read()
                if frame is None or current_frame_index > frames_to_propagate:
                    break

                # convert numpy array to pytorch tensor format
                frame_torch, _ = image_to_torch(frame, device=device)
                if current_frame_index == 0:
                    # initialize with the mask
                    mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                    # the background mask is not fed into the model
                    prediction = processor.step(frame_torch, mask_torch[1:])
                else:
                    # propagate only
                    prediction = processor.step(frame_torch)

                # argmax, convert to numpy
                prediction = torch_prob_to_numpy_mask(prediction)

                if current_frame_index % visualize_every == 0:
                    visualization = overlay_davis(frame[...,::-1], prediction)
                    masked_video.append(visualization)

                current_frame_index += 1
        ed = time.time()

        print(f"Propagation time: {ed-st} s")

        from moviepy.editor import ImageSequenceClip, AudioFileClip
        
        audio = AudioFileClip(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_dir = f'./XMem/output/{video_name.split("/")[-1].split(".")[0]}.mp4'
        if not os.path.exists('./XMem/output/'):
            os.mkdir('./XMem/output/')
        clip = ImageSequenceClip(sequence=masked_video, fps=fps)
        # Set the audio of the new video to be the audio from the original video
        clip = clip.set_audio(audio)
        clip.write_videofile(output_dir, fps=fps, audio=True)

        return output_dir
else:
    def get_mask(mask, obj_id=None):
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

        # print(color, mask.shape)

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image

    def overlay_mask(image, masks):
        for mask_ in masks:
            alpha = mask_[..., 3:]
            mask_ = mask_[..., :3] * 255
            # print(set(mask_.flatten()), set(alpha.flatten()))
            image = image * (1 - alpha) + mask_ * alpha
        return image
    
    def seg_propagation():
        global VIDEO_NAME, VIDEO_PATH, FPS, inference_state

        st = time.time()

        video_segments = {}  # video_segments contains the per-frame segmentation results
        masked_video = []
        for out_frame_idx, out_obj_ids, out_mask_logits in vid_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            masked_video.append(
                overlay_mask(cv2.imread(f"{VIDEO_PATH}/{out_frame_idx}.jpg")[...,::-1], [get_mask(out_mask, out_obj_id) for out_obj_id, out_mask in video_segments[out_frame_idx].items()])
            )
        
        ed = time.time()

        print(f"Propagation time: {ed-st} s")

        from moviepy.editor import ImageSequenceClip, AudioFileClip

        output_dir = f'./output/{VIDEO_NAME.split("/")[-1].split(".")[0]}.mp4'
        if not os.path.exists('./output/'):
            os.mkdir('./output/')

        AUDIO = AudioFileClip(VIDEO_NAME)
        print(len(masked_video), FPS, AUDIO)

        try:
            clip = ImageSequenceClip(sequence=masked_video, fps=FPS)
            # Set the audio of the new video to be the audio from the original video
            clip = clip.set_audio(AUDIO)
            clip.write_videofile(output_dir, fps=FPS, audio=True)
        except:
            clip = ImageSequenceClip(sequence=masked_video, fps=FPS)
            clip.write_videofile(output_dir, fps=FPS, audio=False)

        return output_dir

VIDEO_NAME = ""
VIDEO_PATH = ""
FPS = 0

@app.post("/video")
async def obtain_videos(
    video: UploadFile = File(...)
):
    # Read the video data as bytes
    video_data = await video.read()

    # Write the video data to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_data)
    temp_file.close()

    print(temp_file.name)

    global VIDEO_NAME, VIDEO_PATH, FPS, inference_state
    if VIDEO_NAME != "":
        os.unlink(VIDEO_NAME)
    VIDEO_NAME = temp_file.name

    if use_sam2:
        VIDEO_PATH = os.path.join('./output', VIDEO_NAME.split("/")[-1].split(".")[0])
        os.makedirs(VIDEO_PATH, exist_ok=True)
        assert os.path.exists(VIDEO_PATH)

        print("VIDEO_PATH", VIDEO_PATH)
        # save the video frames in jpg format
        cap = cv2.VideoCapture(VIDEO_NAME)
        frame_count = 0
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            if frame is None:
                break
            
            cv2.imwrite(f"{VIDEO_PATH}/{frame_count}.jpg", frame)
            frame_count += 1
            # print(f"Succeed in saving frame {frame_count}")

        FPS = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()

        inference_state = vid_predictor.init_state(video_path=VIDEO_PATH)
        vid_predictor.reset_state(inference_state)

    return JSONResponse(
        content={
            "message": "upload video successfully",
        },
        status_code=200,
    )

@app.post("/ini_seg")
async def process_videos(
    ini_seg: UploadFile = File(...)
):
    global VIDEO_NAME, VIDEO_PATH

    ini_seg_data = await ini_seg.read()

    tmp_seg_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_seg_file.write(ini_seg_data)
    tmp_seg_file.close()

    print(tmp_seg_file.name)

    if VIDEO_NAME == "" and VIDEO_PATH == "":
        raise HTTPException(status_code=204, detail="No content")
    
    if not use_sam2:
        res_path = seg_propagation(VIDEO_NAME, tmp_seg_file.name)
    else:
        res_path = seg_propagation()

    os.unlink(tmp_seg_file.name)
    # shutil.rmtree(VIDEO_PATH)
    # os.unlink(VIDEO_NAME)
    # VIDEO_NAME = ""

    # Return a FileResponse with the processed video path
    return FileResponse(
        res_path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{VIDEO_NAME.split("/")[-1].split(".")[0]}.mp4"',
        },
    )

@app.post("/undo")
async def undo_mask():
    global segmented_mask
    # this is not necessary actually because segmented_mask is only maintained but not used
    segmented_mask.pop()

    return JSONResponse(
        content={
            "message": "Clear successfully",
        },
        status_code=200,
    )


from fastapi import Request


@app.post("/click")
async def click_images(
    request: Request,
):  
    global mask_input, interactive_mask, inference_state

    form_data = await request.form()
    type_list = [int(i) for i in form_data.get("type").split(',')]
    click_list = [int(i) for i in form_data.get("click_list").split(',')]
    # x_list = [int(i) for i in form_data.get("x").split(',')]
    # y_list = [int(i) for i in form_data.get("y").split(',')]

    point_coords = np.array(click_list, np.float32).reshape(-1, 2)
    point_labels = np.array(type_list).reshape(-1)

    print(point_coords)
    print(point_labels)

    if (len(point_coords) == 1):
        mask_input = None

    if VIDEO_NAME == "":
        masks_, scores_, logits_ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True,
        )

        best_idx = np.argmax(scores_)
        res = masks_[best_idx]
        mask_input = logits_[best_idx][None, :, :]
    else:
        _, _, out_mask_logits = vid_predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
        )
        # print(out_mask_logits.shape)
        res = (out_mask_logits[0][0] > 0.0).cpu().numpy()

    len_prompt = len(point_labels)
    len_mask = len(interactive_mask)
    if len_mask == 0 or len_mask < len_prompt:
        interactive_mask.append(res)
    else:
        interactive_mask[len_prompt-1] = res

    # Return a JSON response
    res = Image.fromarray(res)
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/finish_click")
async def finish_interactive_click(
    mask_idx: int = Form(...),
):
    global segmented_mask, interactive_mask

    segmented_mask.append(interactive_mask[mask_idx])
    interactive_mask = list()

    return JSONResponse(
        content={
            "message": "Finish successfully",
        },
        status_code=200,
    )
    

@app.post("/rect")
async def rect_images(
    start_x: int = Form(...), # horizontal
    start_y: int = Form(...), # vertical
    end_x: int = Form(...), # horizontal
    end_y: int = Form(...)  # vertical
):
    masks_, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([[start_x, start_y, end_x, end_y]]),
        multimask_output=False
    )
    
    res = Image.fromarray(masks_[0])
    # res.save("res.png")
    print(masks_[0].shape)
    # res.save("res.png")

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/everything")
async def seg_everything():
    """
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
    """
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER
    if type(GLOBAL_MASK) != type(None):
        return JSONResponse(
            content={
                "masks": pil_image_to_base64(GLOBAL_MASK),
                "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
                "message": "Images processed successfully"
            },
            status_code=200,
        )


    masks = mask_generator.generate(GLOBAL_IMAGE)
    assert len(masks) > 0, "No masks found"

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))

    # Create a new image with the same size as the original image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1 # color can only be in range [1, 255]
    
    res = Image.fromarray(img)
    GLOBAL_MASK = res

    # Save the original image, mask, and cropped segments to a zip file in memory
    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(GLOBAL_IMAGE)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Cut out the segmented regions as smaller squares
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
            cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))

            # Create a transparent image with the same size as the cropped image
            transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))

            # Create a mask from the segmentation data and crop it
            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))

            # Combine the cropped image with the transparent image using the mask
            result = Image.composite(cropped.convert("RGBA"), transparent, mask_cropped)

            # Save the result to the zip file
            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())

    # move the file pointer to the beginning of the file so we can read whole file
    zip_buffer.seek(0)
    GLOBAL_ZIPBUFFER = zip_buffer

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(GLOBAL_MASK),
            "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/postprocess")
async def postprocess_images(
    mask_image: UploadFile = File(...),
    image_name: str = Form("image")
):
    """Postprocess combined mask containing both lanes and drivable areas"""
    try:
        # Read the combined mask image data
        mask_data = await mask_image.read()
        
        print(f"Processing combined mask: {mask_image.filename}")
        print(f"Image name: {image_name}")
        
        # Process the combined mask
        result_image = process_combined_mask_server(mask_data)
        
        # Convert result to PIL Image and save to a temporary file
        pil_image = Image.fromarray(result_image)
        
        # Create a temporary file with the proper name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pil_image.save(temp_file.name, format='PNG')
        temp_file.close()
        
        print(f"Saved temporary result to: {temp_file.name}")
        
        # Use the image name for the download filename
        output_filename = f"{image_name}_hdmap.png"
        
        # Return as FileResponse which handles downloads properly
        return FileResponse(
            temp_file.name,
            media_type="image/png",
            filename=output_filename,
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
        
    except Exception as e:
        print(f"Error in postprocess_images: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.get("/assets/{path}/{file_name}", response_class=FileResponse)
async def read_assets(path, file_name):
    return f"assets/{path}/{file_name}"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return read_content('segDrawer.html')



# Deallocate GPU memory on process exit using atexit
import atexit
def cleanup_gpu():
    if torch.cuda.is_available():
        print("[Server Shutdown] Deallocating GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[Server Shutdown] GPU memory deallocated.")
    else:
        print("[Server Shutdown] No GPU to deallocate.")
atexit.register(cleanup_gpu)

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=7860)