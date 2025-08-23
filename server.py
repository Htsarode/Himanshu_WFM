# Start the FastAPI server if run as main

#!/usr/bin/env python3
"""
Enhanced SAM2 Video Annotation Server
Integrates server.py functionality with segDrawer2.html interface
Supports magic tool, rectangular tool, and pencil drawing tool
"""

from fastapi import FastAPI, status, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# Add SAM2 path to Python path
import sys
sys.path.append('/home/arvinds7/Desktop/ML3/sam2')


# --- SAM1 imports ---
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor
# --- SAM2 imports ---
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
from typing import List, Optional
import uvicorn
import random

class EnhancedAnnotationServer:
    def __init__(self):
        self.app = FastAPI(title="Enhanced SAM2 Video Annotation Server", debug=True)
        self.setup_cors()
        # Initialize device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

        # Global state variables
        self.GLOBAL_IMAGE = None
        self.GLOBAL_MASK = None
        self.GLOBAL_ZIPBUFFER = None
        self.VIDEO_NAME = ""
        self.VIDEO_PATH = ""
        self.FPS = 0
        self.inference_state = None

        # Annotation state
        self.input_point = []
        self.input_label = []
        self.masks = []
        self.segmented_mask = []
        self.interactive_mask = []
        self.mask_input = None

        # Initialize both SAM1 and SAM2 models
        self.setup_sam_models()

        # Setup routes
        self.setup_routes()
        
    def setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_sam_models(self):
        """Initialize both SAM1 and SAM2 models"""
        # --- SAM1 ---
        sam1_checkpoint = "sam_vit_l_0b3195.pth"  # Update path if needed
        sam1_type = "vit_l"  # or "vit_h", "vit_b" as available
        sam1_model = sam_model_registry[sam1_type](checkpoint=sam1_checkpoint)
        sam1_model = sam1_model.to(self.device)
        self.sam1_predictor = SamPredictor(sam1_model)

        # --- SAM2 ---
        sam2_checkpoint = "sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        sam2_model = sam2_model.to(self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.vid_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        if hasattr(self.vid_predictor, 'to'):
            self.vid_predictor = self.vid_predictor.to(self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
        print("SAM1 and SAM2 models initialized successfully!")
    
    def setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def read_index():
            """Serve the main HTML interface"""
            return self.read_content('segDrawer 2.html')
        
        @self.app.get("/assets/{path}/{file_name}", response_class=FileResponse)
        async def read_assets(path, file_name):
            """Serve static assets"""
            return f"assets/{path}/{file_name}"
        
        @self.app.post("/image")
        async def process_images(image: UploadFile = File(...)):
            """Process uploaded image for annotation (uses SAM1)"""
            self.segmented_mask = []
            self.interactive_mask = []
            # Read image data
            image_data = await image.read()
            image_data = BytesIO(image_data)
            img = np.array(Image.open(image_data))
            print("Received image shape:", img.shape)
            # Store global image (remove alpha channel if present)
            self.GLOBAL_IMAGE = img[:,:,:-1] if img.shape[2] == 4 else img[:,:,:3]
            self.GLOBAL_MASK = None
            self.GLOBAL_ZIPBUFFER = None
            # Set image for SAM1 predictor
            self.sam1_predictor.set_image(self.GLOBAL_IMAGE)
            return JSONResponse(
                content={"message": "Image received successfully (SAM1)"},
                status_code=200,
            )
        
        @self.app.post("/video")
        async def obtain_videos(
            video: UploadFile = File(...),
            max_frames: int = 100,
            frame_stride: int = 1
        ):
            """Process uploaded video for annotation"""
            try:
                # Read video data
                video_data = await video.read()
                # Write to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(video_data)
                temp_file.close()
                # Clean up previous video
                if self.VIDEO_NAME != "":
                    try:
                        os.unlink(self.VIDEO_NAME)
                    except Exception as e:
                        print(f"Warning: Could not delete previous video: {e}")
                self.VIDEO_NAME = temp_file.name
                self.VIDEO_PATH = os.path.join('./output', self.VIDEO_NAME.split("/")[-1].split(".")[0])
                # Clean up old frames
                if os.path.exists(self.VIDEO_PATH):
                    for f in os.listdir(self.VIDEO_PATH):
                        if f.endswith('.jpg'):
                            try:
                                os.remove(os.path.join(self.VIDEO_PATH, f))
                            except Exception as e:
                                print(f"Warning: Could not delete old frame {f}: {e}")
                else:
                    os.makedirs(self.VIDEO_PATH, exist_ok=True)
                # Extract frames
                cap = cv2.VideoCapture(self.VIDEO_NAME)
                if not cap.isOpened():
                    raise RuntimeError("Failed to open uploaded video file. Please check the file format.")
                frame_count = 0
                while cap.isOpened() and frame_count < max_frames:
                    frame_idx = frame_count * frame_stride
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        if frame_count == 0:
                            raise RuntimeError("No frames could be read from the uploaded video. Please check the file.")
                        break
                    cv2.imwrite(f"{self.VIDEO_PATH}/{frame_count:06d}.jpg", frame)
                    frame_count += 1
                self.FPS = cap.get(cv2.CAP_PROP_FPS) or 25
                cap.release()
                if frame_count == 0:
                    raise RuntimeError("No frames extracted from video. Please upload a valid video file.")
                # Initialize video predictor inference state
                try:
                    self.inference_state = self.vid_predictor.init_state(video_path=self.VIDEO_PATH)
                    if self.inference_state is None:
                        raise RuntimeError("Failed to initialize inference state")
                    self.vid_predictor.reset_state(self.inference_state)
                    print(f"Video processed: {frame_count} frames extracted")
                except Exception as e:
                    print(f"Error initializing inference state: {e}")
                    self.inference_state = None
                    return JSONResponse(
                        content={
                            "error": "Failed to initialize video inference state",
                            "message": str(e)
                        },
                        status_code=500,
                    )
                return JSONResponse(
                    content={
                        "message": f"Video uploaded and processed successfully. {frame_count} frames extracted.",
                        "total_frames": frame_count
                    },
                    status_code=200,
                )
            except Exception as e:
                print(f"Video upload error: {e}")
                return JSONResponse(
                    content={
                        "error": "Video upload or processing failed",
                        "message": str(e)
                    },
                    status_code=500,
                )
            
            return JSONResponse(
                content={"message": "Video uploaded successfully"},
                status_code=200,
            )
        
        @self.app.post("/click")
        async def click_images(request: Request):
            """Handle magic tool clicks"""
            form_data = await request.form()
            # Parse form data - handle both array and comma-separated formats
            type_data = form_data.getlist("type")
            click_data = form_data.getlist("click_list")
            object_type = form_data.get("object_type", None)
            print(f"Debug - type_data: {type_data}")
            print(f"Debug - click_data: {click_data}")
            print(f"Debug - object_type: {object_type}")

            # Convert to numpy arrays
            point_coords = []
            point_labels = []
            # Handle type data - convert from form format
            try:
                if len(type_data) == 1:
                    type_str = type_data[0]
                    if ',' in type_str:
                        type_values = [int(x.strip()) for x in type_str.split(',') if x.strip()]
                    else:
                        type_values = [int(type_str)]
                else:
                    type_values = [int(x) for x in type_data if x]
                point_labels = type_values
                print(f"Debug - parsed labels: {point_labels}")
            except Exception as e:
                print(f"Error parsing type data: {e}")
                return JSONResponse(
                    content={"error": f"Invalid type data format: {e}"},
                    status_code=400,
                )
            # Handle click coordinates
            try:
                coords = []
                if len(click_data) == 1:
                    coord_str = click_data[0]
                    if ',' in coord_str:
                        coords = [float(x.strip()) for x in coord_str.split(',') if x.strip()]
                    else:
                        coords = [float(coord_str)]
                else:
                    for coord_str in click_data:
                        if coord_str:
                            coords.append(float(coord_str))
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        point_coords.append([coords[i], coords[i+1]])
                print(f"Debug - parsed coords: {point_coords}")
            except Exception as e:
                print(f"Error parsing coordinate data: {e}")
                return JSONResponse(
                    content={"error": f"Invalid coordinate data format: {e}"},
                    status_code=400,
                )
            point_coords = np.array(point_coords, dtype=np.float32)
            point_labels = np.array(point_labels, dtype=np.int32)

            # Map object_type to label value (customize as needed)
            label_map = {"lanes": 1, "drivableArea": 2}
            selected_label = label_map.get(object_type, None)


            try:
                # If it's a video, use SAM2 predictor; if image, use SAM1 predictor
                if self.VIDEO_PATH and self.inference_state is not None:
                    # Video: use SAM2 predictor
                    # Set the first frame as image if not already set
                    if self.GLOBAL_IMAGE is None:
                        frame_files = sorted([f for f in os.listdir(self.VIDEO_PATH) if f.endswith('.jpg')])
                        if frame_files:
                            first_frame_path = os.path.join(self.VIDEO_PATH, frame_files[0])
                            first_frame = cv2.imread(first_frame_path)
                            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                            self.GLOBAL_IMAGE = first_frame_rgb
                    # Always set image before prediction
                    self.sam2_predictor.set_image(self.GLOBAL_IMAGE)
                    masks_, scores, logits = self.sam2_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False
                    )
                else:
                    # Image: use SAM1 predictor
                    if self.GLOBAL_IMAGE is not None:
                        self.sam1_predictor.set_image(self.GLOBAL_IMAGE)
                    else:
                        raise Exception("No image loaded for prediction.")
                    masks_, scores, logits = self.sam1_predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False
                    )

                mask_to_use = masks_[0]
                # If object_type is specified, try to filter the mask to only the selected object
                if selected_label is not None:
                    # The mask may contain multiple objects; try to keep only the region corresponding to the selected label
                    # If the model outputs a label map, use it; otherwise, use the mask as is if the click label matches
                    if selected_label in point_labels:
                        # For lanes, apply morphological closing to connect broken segments
                        if object_type == "lanes":
                            kernel = np.ones((15, 15), np.uint8)  # You can adjust the kernel size as needed
                            mask_to_use = cv2.morphologyEx(mask_to_use.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                    else:
                        # No matching label, return empty mask
                        mask_to_use = np.zeros_like(mask_to_use)

                mask_image = Image.fromarray(mask_to_use.astype(np.uint8) * 255)
                self.interactive_mask.append(mask_to_use)
                self.segmented_mask.append(mask_to_use.astype(bool))
                if self.GLOBAL_MASK is None:
                    self.GLOBAL_MASK = mask_image
                else:
                    existing_array = np.array(self.GLOBAL_MASK)
                    new_mask_array = mask_to_use.astype(np.uint8) * 255
                    if existing_array.shape[:2] == new_mask_array.shape[:2]:
                        combined = np.maximum(existing_array, new_mask_array)
                        self.GLOBAL_MASK = Image.fromarray(combined)
                mask_b64 = self.pil_image_to_base64(mask_image)
                print(f"Magic tool: Added mask, total interactive: {len(self.interactive_mask)}, total segmented: {len(self.segmented_mask)}")
                return JSONResponse(
                    content={
                        "masks": mask_b64,
                        "message": "Magic tool prediction successful"
                    },
                    status_code=200,
                )
            except Exception as e:
                print(f"Error in magic tool prediction: {e}")
                return JSONResponse(
                    content={
                        "error": "Magic tool prediction failed",
                        "message": str(e)
                    },
                    status_code=500,
                )
        
        @self.app.post("/ini_seg")
        async def ini_seg(
            startX: int = Form(...),
            startY: int = Form(...),
            endX: int = Form(...),
            endY: int = Form(...),
            frame_idx: int = Form(0)
        ):
            """Initialize segmentation for video (compatible with segDrawer2.html)"""
            if self.inference_state is None:
                return JSONResponse(
                    content={"error": "No video loaded or inference state not initialized"},
                    status_code=400,
                )
            
            try:
                # For video, we use the video predictor instead of image predictor
                box = np.array([startX, startY, endX, endY], dtype=np.float32)
                
                # Add box to video predictor for the specified frame
                _, out_obj_ids, out_mask_logits = self.vid_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=1,  # Use object ID 1
                    box=box,
                )
                
                # Convert mask to PIL image
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                mask_image = Image.fromarray(mask)
                
                # Convert to base64
                mask_b64 = self.pil_image_to_base64(mask_image)
                
                return JSONResponse(
                    content={
                        "masks": mask_b64,
                        "message": "Video segmentation initialized successfully"
                    },
                    status_code=200,
                )
                
            except Exception as e:
                print(f"Error in video initialization: {e}")
                return JSONResponse(
                    content={
                        "error": "Video segmentation initialization failed",
                        "message": str(e)
                    },
                    status_code=500,
                )
        
        @self.app.post("/finish_click")
        async def finish_click(mask_idx: int = Form(...)):
            """Finish magic tool interaction"""
            if self.interactive_mask:
                # Move from interactive to segmented
                self.segmented_mask.extend(self.interactive_mask)
                self.interactive_mask = []
            
            return JSONResponse(
                content={"message": "Click sequence finished"},
                status_code=200,
            )
        
        @self.app.post("/rect")
        async def rect_images(
            start_x: int = Form(...),
            start_y: int = Form(...),
            end_x: int = Form(...),
            end_y: int = Form(...)
        ):
            """Handle rectangular tool"""
            try:
                # Create bounding box
                box = np.array([[start_x, start_y, end_x, end_y]])

                # Check if we're in video mode or image mode
                if self.inference_state is not None and self.VIDEO_PATH:
                    # Video mode - use video predictor (SAM2)
                    _, out_obj_ids, out_mask_logits = self.vid_predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=0,  # Use first frame
                        obj_id=2,  # Use object ID 2 for rectangle
                        box=box[0],  # Remove extra dimension
                    )
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                else:
                    # Image mode - use SAM1 predictor
                    if self.GLOBAL_IMAGE is None:
                        raise ValueError("No image loaded")
                    masks_, scores, logits = self.sam1_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=False
                    )
                    mask = masks_[0]

                # Robustly squeeze/reshape mask to 2D
                mask = np.array(mask)
                while mask.ndim > 2:
                    mask = np.squeeze(mask, axis=0)
                if mask.ndim != 2:
                    raise ValueError(f"Rectangle tool: mask shape after squeeze is invalid: {mask.shape}")
                mask = (mask > 0).astype(np.uint8) * 255

                # Convert mask to PIL image
                mask_image = Image.fromarray(mask)

                # Store segmented mask
                self.segmented_mask.append((mask > 0))

                # Convert to base64
                mask_b64 = self.pil_image_to_base64(mask_image)

                return JSONResponse(
                    content={
                        "masks": mask_b64,
                        "message": "Rectangle tool prediction successful"
                    },
                    status_code=200,
                )

            except RuntimeError as e:
                if "set_image" in str(e):
                    return JSONResponse(
                        content={
                            "error": "No image uploaded. Please upload an image first.",
                            "message": str(e)
                        },
                        status_code=400,
                    )
                else:
                    return JSONResponse(
                        content={
                            "error": "Rectangle tool prediction failed",
                            "message": str(e)
                        },
                        status_code=500,
                    )
            except Exception as e:
                print(f"Error in rectangle tool: {e}")
                return JSONResponse(
                    content={
                        "error": "Rectangle tool prediction failed",
                        "message": str(e)
                    },
                    status_code=500,
                )
        
        @self.app.post("/pencil_draw")
        async def pencil_draw(request: Request):
            """Handle pencil/drawing tool"""
            try:
                form_data = await request.form()

                # Get drawing path data
                path_data = form_data.get("path_data")
                if not path_data:
                    raise ValueError("No path data provided")

                # Parse path data (should be JSON string with coordinates)
                import json
                try:
                    paths = json.loads(path_data)
                except json.JSONDecodeError:
                    raise ValueError("Invalid path data format")

                # Create mask from drawing paths
                if self.GLOBAL_IMAGE is None:
                    raise ValueError("No image loaded")

                h, w = self.GLOBAL_IMAGE.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

                # Draw paths on mask
                for path in paths:
                    if 'points' in path:
                        points = np.array([[int(round(p['x'])), int(round(p['y']))] for p in path['points']], dtype=np.int32)
                        if len(points) > 1:
                            for i in range(len(points) - 1):
                                cv2.line(mask, tuple(points[i]), tuple(points[i+1]), 255, thickness=5)

                # Convert mask to PIL image
                mask_image = Image.fromarray(mask)

                # Store drawn mask for download/propagation
                self.segmented_mask.append(mask.astype(bool))

                # Convert to base64
                mask_b64 = self.pil_image_to_base64(mask_image)

                return JSONResponse(
                    content={
                        "masks": mask_b64,
                        "message": "Pencil drawing successful"
                    },
                    status_code=200,
                )

            except Exception as e:
                print(f"Error in pencil drawing: {e}")
                return JSONResponse(
                    content={
                        "error": "Pencil drawing failed",
                        "message": str(e)
                    },
                    status_code=500,
                )
        
        @self.app.post("/everything")
        async def seg_everything():
            """Segment everything in the image"""
            if self.GLOBAL_MASK is not None:
                return JSONResponse(
                    content={
                        "masks": self.pil_image_to_base64(self.GLOBAL_MASK),
                        "zipfile": b64encode(self.GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
                        "message": "Images processed successfully"
                    },
                    status_code=200,
                )
            
            if self.GLOBAL_IMAGE is None:
                return JSONResponse(
                    content={"error": "No image loaded"},
                    status_code=400,
                )
            
            # Generate all masks
            masks = self.mask_generator.generate(self.GLOBAL_IMAGE)
            if not masks:
                return JSONResponse(
                    content={"error": "No masks found"},
                    status_code=400,
                )
            
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
            print(f"Generated {len(sorted_anns)} masks")
            
            # Create combined mask image
            img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
            for idx, ann in enumerate(sorted_anns):
                img[ann['segmentation']] = (idx % 255) + 1
            
            self.GLOBAL_MASK = Image.fromarray(img)
            
            # Create ZIP file with segments
            zip_buffer = BytesIO()
            PIL_GLOBAL_IMAGE = Image.fromarray(self.GLOBAL_IMAGE)
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for idx, ann in enumerate(sorted_anns):
                    left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
                    cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))
                    
                    transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))
                    mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
                    mask_cropped = mask.crop((left, top, right, bottom))
                    
                    result = Image.composite(cropped.convert("RGBA"), transparent, mask_cropped)
                    
                    result_bytes = BytesIO()
                    result.save(result_bytes, format="PNG")
                    result_bytes.seek(0)
                    zip_file.writestr(f"seg_{idx}.png", result_bytes.read())
            
            zip_buffer.seek(0)
            self.GLOBAL_ZIPBUFFER = zip_buffer
            
            return JSONResponse(
                content={
                    "masks": self.pil_image_to_base64(self.GLOBAL_MASK),
                    "zipfile": b64encode(self.GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
                    "message": "Everything segmented successfully"
                },
                status_code=200,
            )
        
        @self.app.post("/undo")
        async def undo_mask():
            """Undo last annotation"""
            if self.segmented_mask:
                self.segmented_mask.pop()
                return JSONResponse(
                    content={"message": "Undo successful"},
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content={"message": "No mask to undo"},
                    status_code=200,
                )
        
        @self.app.post("/propagate_video")
        async def propagate_video():
            """Propagate annotations to entire video using SAM2"""
            if self.inference_state is None:
                return JSONResponse(
                    content={"error": "No video loaded or inference state not initialized"},
                    status_code=400,
                )
            
            try:
                # Propagate through video
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.vid_predictor.propagate_in_video(self.inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                
                # Create output video with overlays
                frame_paths = sorted([
                    os.path.join(self.VIDEO_PATH, f) 
                    for f in os.listdir(self.VIDEO_PATH) 
                    if f.endswith('.jpg')
                ], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                
                out_frames = []
                for frame_idx, frame_path in enumerate(frame_paths):
                    frame = cv2.imread(frame_path)
                    
                    # Apply masks if available
                    if frame_idx in video_segments:
                        overlay = frame.copy().astype(np.float32)
                        for obj_id, mask in video_segments[frame_idx].items():
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            mask_bin = mask.astype(np.uint8)
                            
                            # Apply colored overlay
                            color = self.get_object_color(obj_id)
                            alpha = 0.3
                            overlay = overlay * (1 - alpha * mask_bin[..., None]) + \
                                     (color * 255 * mask_bin[..., None]) * alpha
                        
                        frame = overlay.astype(np.uint8)
                    
                    out_frames.append(frame)
                
                # Save output video
                if out_frames:
                    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
                    output_dir = './output'
                    output_video_path = os.path.join(output_dir, 'output_video.mp4')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in out_frames], fps=self.FPS)
                    clip.write_videofile(output_video_path, fps=self.FPS, audio=False)
                    
                    print(f"Output video saved to {output_video_path}")
                
                return JSONResponse(
                    content={
                        "message": f"Video propagation completed for {len(out_frames)} frames",
                        "total_frames": len(out_frames)
                    },
                    status_code=200,
                )
                
            except Exception as e:
                print(f"Error in video propagation: {e}")
                return JSONResponse(
                    content={
                        "error": "Video propagation failed",
                        "message": str(e)
                    },
                    status_code=500,
                )
        
        @self.app.get("/download_video")
        async def download_video():
            """Download processed video"""
            video_path = './output/output_video.mp4'
            if os.path.exists(video_path):
                return FileResponse(
                    video_path,
                    media_type="video/mp4",
                    filename="annotated_video.mp4"
                )
            else:
                return JSONResponse(
                    content={"error": "No processed video available"},
                    status_code=404,
                )
        
        @self.app.get("/download_image")
        async def download_image():
            """Download annotated image"""
            try:
                if self.GLOBAL_MASK is not None:
                    # Create download file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    self.GLOBAL_MASK.save(temp_file.name)
                    
                    return FileResponse(
                        temp_file.name,
                        media_type="image/png",
                        filename="annotated_mask.png"
                    )
                else:
                    return JSONResponse(
                        content={"error": "No mask available for download"},
                        status_code=404,
                    )
            except Exception as e:
                return JSONResponse(
                    content={"error": f"Download failed: {str(e)}"},
                    status_code=500,
                )
        
        @self.app.get("/status")
        async def get_status():
            """Get current annotation status for debugging"""
            return JSONResponse(
                content={
                    "global_image_loaded": self.GLOBAL_IMAGE is not None,
                    "global_mask_exists": self.GLOBAL_MASK is not None,
                    "video_loaded": self.VIDEO_PATH != "",
                    "inference_state_ready": self.inference_state is not None,
                    "interactive_masks_count": len(self.interactive_mask),
                    "segmented_masks_count": len(self.segmented_mask),
                    "total_annotations": len(self.interactive_mask) + len(self.segmented_mask)
                },
                status_code=200,
            )
        
        @self.app.get("/download_results")
        async def download_results():
            """Download all annotation results as ZIP"""
            try:
                zip_buffer = BytesIO()
                files_added = 0
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    
                    # Add global combined mask if available
                    if self.GLOBAL_MASK is not None:
                        mask_buffer = BytesIO()
                        self.GLOBAL_MASK.save(mask_buffer, format='PNG')
                        zip_file.writestr("combined_annotation_mask.png", mask_buffer.getvalue())
                        files_added += 1
                        print(f"Added combined mask to ZIP")
                    
                    # Add individual segmented masks (from rectangle and magic tools)
                    for i, mask in enumerate(self.segmented_mask):
                        try:
                            if isinstance(mask, bool) or mask.dtype == bool:
                                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                            else:
                                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                            
                            mask_buffer = BytesIO()
                            mask_img.save(mask_buffer, format='PNG')
                            zip_file.writestr(f"annotation_segment_{i+1}.png", mask_buffer.getvalue())
                            files_added += 1
                            print(f"Added segment {i+1} to ZIP")
                        except Exception as e:
                            print(f"Error adding segment {i+1}: {e}")
                    
                    # Add interactive masks (from magic tool)
                    for i, mask in enumerate(self.interactive_mask):
                        try:
                            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                            mask_buffer = BytesIO()
                            mask_img.save(mask_buffer, format='PNG')
                            zip_file.writestr(f"magic_tool_mask_{i+1}.png", mask_buffer.getvalue())
                            files_added += 1
                            print(f"Added magic tool mask {i+1} to ZIP")
                        except Exception as e:
                            print(f"Error adding magic tool mask {i+1}: {e}")
                    
                    # Add original image if available
                    if self.GLOBAL_IMAGE is not None:
                        try:
                            orig_img = Image.fromarray(self.GLOBAL_IMAGE)
                            img_buffer = BytesIO()
                            orig_img.save(img_buffer, format='PNG')
                            zip_file.writestr("original_image.png", img_buffer.getvalue())
                            files_added += 1
                            print(f"Added original image to ZIP")
                        except Exception as e:
                            print(f"Error adding original image: {e}")
                
                print(f"ZIP created with {files_added} files")
                
                if files_added == 0:
                    return JSONResponse(
                        content={"error": "No annotation data available to download. Please annotate some regions first."},
                        status_code=400,
                    )
                
                zip_buffer.seek(0)
                
                # Create temporary file for download
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                temp_file.write(zip_buffer.getvalue())
                temp_file.close()
                
                print(f"Created temp ZIP file: {temp_file.name}, size: {os.path.getsize(temp_file.name)} bytes")
                
                return FileResponse(
                    temp_file.name,
                    media_type="application/zip",
                    filename="annotation_results.zip"
                )
                
            except Exception as e:
                print(f"Download error: {e}")
                return JSONResponse(
                    content={"error": f"Download failed: {str(e)}"},
                    status_code=500,
                )
        
        @self.app.post("/download")
        async def download_annotated_video(mask: UploadFile = File(None)):
            """Process video with all annotation masks and return annotated video (OpenCV only)"""
            try:
                if self.VIDEO_PATH == "":
                    return JSONResponse(
                        content={"error": "No video loaded. Please upload a video first."},
                        status_code=400,
                    )
                # Combine all masks in segmented_mask and interactive_mask (logical OR)
                binary_mask = None
                all_masks = []
                if mask is not None:
                    mask_content = await mask.read()
                    mask_image = Image.open(BytesIO(mask_content)).convert('L')
                    mask_array = np.array(mask_image)
                    all_masks.append((mask_array > 128).astype(np.uint8))
                if self.segmented_mask:
                    all_masks.extend([(m.astype(np.uint8) if m.dtype != np.uint8 else m) for m in self.segmented_mask])
                if self.interactive_mask:
                    all_masks.extend([(m.astype(np.uint8) if m.dtype != np.uint8 else m) for m in self.interactive_mask])
                if all_masks:
                    combined = np.zeros_like(all_masks[0], dtype=np.uint8)
                    for m in all_masks:
                        combined = np.logical_or(combined, m)
                    binary_mask = combined.astype(np.uint8)
                    print("Using combined annotation mask from all user annotations.")
                else:
                    return JSONResponse(
                        content={"error": "No mask uploaded and no annotation mask available. Please annotate first."},
                        status_code=400,
                    )
                if self.inference_state is not None:
                    try:
                        _, out_obj_ids, out_mask_logits = self.vid_predictor.add_new_mask(
                            inference_state=self.inference_state,
                            frame_idx=0,
                            obj_id=1,
                            mask=binary_mask
                        )
                        print(f"Added mask to video predictor for object {out_obj_ids}")
                        # Propagate the mask through the video
                        video_segments = {}
                        for out_frame_idx, out_obj_ids, out_mask_logits in self.vid_predictor.propagate_in_video(self.inference_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                        print(f"Propagated mask through {len(video_segments)} frames")
                        # Write annotated video using OpenCV only
                        frame_files = sorted([
                            f for f in os.listdir(self.VIDEO_PATH) if f.endswith('.jpg')
                        ], key=lambda x: int(os.path.splitext(x)[0]))
                        if not frame_files:
                            print(f"No frames found in {self.VIDEO_PATH}")
                            return JSONResponse(
                                content={"error": "No frames found in video directory."},
                                status_code=500,
                            )
                        first_frame = cv2.imread(os.path.join(self.VIDEO_PATH, frame_files[0]))
                        if first_frame is None:
                            print(f"Failed to read first frame: {frame_files[0]}")
                            return JSONResponse(
                                content={"error": "Failed to read first frame."},
                                status_code=500,
                            )
                        frame_height, frame_width = first_frame.shape[:2]
                        fps = self.FPS if self.FPS else 25
                        print(f"Output video properties: {fps} FPS, {frame_width}x{frame_height}")
                        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
                        frames_written = 0
                        frame_idx_map = {int(os.path.splitext(f)[0]): f for f in frame_files}
                        max_idx = max(frame_idx_map.keys()) if frame_idx_map else -1
                        for idx in range(max_idx + 1):
                            fname = frame_idx_map.get(idx)
                            if fname is None:
                                print(f"Frame index {idx} missing in frame_files.")
                                continue
                            frame = cv2.imread(os.path.join(self.VIDEO_PATH, fname))
                            if frame is None:
                                print(f"Failed to read frame: {fname}")
                                continue
                            overlayed = False
                            if idx in video_segments:
                                for obj_id, mask in video_segments[idx].items():
                                    if mask is None:
                                        print(f"Warning: mask is None for frame {idx}, obj {obj_id}")
                                        continue
                                    mask_arr = mask.astype(np.uint8)
                                    if mask_arr.size == 0:
                                        print(f"Warning: empty mask for frame {idx}, obj {obj_id}")
                                        continue
                                    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
                                        mask_arr = np.squeeze(mask_arr, axis=0)
                                    if mask_arr.ndim != 2:
                                        print(f"Warning: mask for frame {idx}, obj {obj_id} has invalid shape {mask_arr.shape} after squeeze")
                                        continue
                                    overlay = np.zeros_like(frame)
                                    overlay[mask_arr > 0] = [0, 0, 255]  # Red overlay
                                    alpha = 0.7
                                    frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
                                    overlayed = True
                                if overlayed:
                                    print(f"Applied mask overlay to frame {idx} ({fname})")
                                else:
                                    print(f"No valid mask to overlay for frame {idx} ({fname})")
                            else:
                                print(f"No mask for frame {idx} ({fname})")
                            out_video.write(frame)
                            frames_written += 1
                        out_video.release()
                        print(f"Created annotated video: {temp_video_path}, frames written: {frames_written}")
                        if frames_written == 0:
                            return JSONResponse(
                                content={"error": "No frames were written to the output video. Please check your input video and mask."},
                                status_code=500,
                            )
                        return FileResponse(
                            temp_video_path,
                            media_type="video/mp4",
                            filename="annotated_video.mp4"
                        )
                    except Exception as e:
                        print(f"Video processing error: {e}")
                        return JSONResponse(
                            content={"error": f"Video processing failed: {str(e)}"},
                            status_code=500,
                        )
                else:
                    return JSONResponse(
                        content={"error": "Video inference state not initialized. Please upload a video first."},
                        status_code=400,
                    )
            except Exception as e:
                print(f"Download endpoint error: {e}")
                return JSONResponse(
                    content={"error": f"Processing failed: {str(e)}"},
                    status_code=500,
                )
    
    def read_content(self, file_path: str) -> str:
        """Read file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def pil_image_to_base64(self, image):
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def get_object_color(self, obj_id):
        """Get color for object ID"""
        colors = {
            1: np.array([0.2, 0.4, 1.0]),  # blue
            2: np.array([1.0, 0.5, 0.0]),  # orange
        }
        if obj_id in colors:
            return colors[obj_id]
        else:
            # Use colormap for other objects
            cmap = plt.get_cmap("tab20")
            return np.array(cmap(obj_id % 20)[:3])
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the server"""
        print(f"Starting Enhanced SAM2 Video Annotation Server on http://{host}:{port}")
        print("Features:")
        print("- Magic tool (point-based segmentation)")
        print("- Rectangular tool (bounding box segmentation)")
        print("- Pencil tool (manual drawing)")
        print("- Video propagation with SAM2")
        print("- Automatic segmentation")
        uvicorn.run(self.app, host=host, port=port)


# --- Ensure this is the last code in the file ---
if __name__ == "__main__":
    server = EnhancedAnnotationServer()
    import uvicorn
    print("Starting EnhancedAnnotationServer on http://0.0.0.0:8000")
    uvicorn.run(server.app, host="0.0.0.0", port=8000)
