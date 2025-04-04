import os
import sys
import gradio as gr
import torch
import subprocess
import argparse
import glob

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["GRADIO_TEMP_DIR"] = os.path.join(project_root, "tmp", "gradio")
sys.path.append(project_root)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Diffusion as Shader Web UI")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--share", action="store_true", help="Share the web UI")
parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
parser.add_argument("--model_path", type=str, default="EXCAI/Diffusion-As-Shader", help="Path to model checkpoint")
parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
args = parser.parse_args()

# Use the original GPU ID throughout the entire code for consistency
GPU_ID = args.gpu

# Set environment variables - this used to remap the GPU, but we're removing this for consistency
# Instead, we'll pass the original GPU ID to all commands
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Commented out to ensure consistent GPU ID usage

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    GPU_COUNT = torch.cuda.device_count()
    GPU_NAMES = [f"{i}: {torch.cuda.get_device_name(i)}" for i in range(GPU_COUNT)]
else:
    GPU_COUNT = 0
    GPU_NAMES = ["CPU (CUDA not available)"]
    GPU_ID = "CPU"

DEFAULT_MODEL_PATH = args.model_path
OUTPUT_DIR = args.output_dir

# Create necessary directories
os.makedirs("outputs", exist_ok=True)
# Create project tmp directory instead of using system temp
os.makedirs(os.path.join(project_root, "tmp"), exist_ok=True)
os.makedirs(os.path.join(project_root, "tmp", "gradio"), exist_ok=True)

def save_uploaded_file(file):
    if file is None:
        return None
        
    # Use project tmp directory instead of system temp
    temp_dir = os.path.join(project_root, "tmp")
    
    if hasattr(file, 'name'):
        filename = file.name
    else:
        # Generate a unique filename if name attribute is missing
        import uuid
        ext = ".tmp"
        if hasattr(file, 'content_type'):
            if "image" in file.content_type:
                ext = ".png"
            elif "video" in file.content_type:
                ext = ".mp4"
        filename = f"{uuid.uuid4()}{ext}"
    
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        # Check if file is a FileStorage object or already a path
        if hasattr(file, 'save'):
            file.save(temp_path)
        elif isinstance(file, str):
            # It's already a path
            return file
        else:
            # Try to read and save the file
            with open(temp_path, 'wb') as f:
                f.write(file.read() if hasattr(file, 'read') else file)
    except Exception as e:
        print(f"Error saving file: {e}")
        return None
        
    return temp_path

def create_run_command(args):
    """Create command based on input parameters"""
    cmd = ["python", "demo.py"]
    
    if "prompt" not in args or args["prompt"] is None or args["prompt"] == "":
        args["prompt"] = ""
    if "checkpoint_path" not in args or args["checkpoint_path"] is None or args["checkpoint_path"] == "":
        args["checkpoint_path"] = DEFAULT_MODEL_PATH
    
    # 添加调试输出
    print(f"DEBUG: Command args: {args}")
    
    for key, value in args.items():
        if value is not None:
            # Handle boolean values correctly - for repaint, we need to pass true/false
            if isinstance(value, bool):
                cmd.append(f"--{key}")
                cmd.append(str(value).lower())  # Convert True/False to true/false
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))
    
    return cmd

def run_process(cmd):
    """Run command and return output"""
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    output = []
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        output.append(line)
        if not line:
            break
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code:
        stderr = process.stderr.read()
        print(f"Error: {stderr}")
        raise subprocess.CalledProcessError(return_code, cmd, output="\n".join(output), stderr=stderr)
    
    return "\n".join(output)

# Process functions for each tab
def process_motion_transfer(source, prompt, mt_repaint_option, mt_repaint_image):
    """Process video motion transfer task"""
    try:
        # Save uploaded files
        input_video_path = save_uploaded_file(source)
        if input_video_path is None:
            return None
        
        print(f"DEBUG: Repaint option: {mt_repaint_option}")
        print(f"DEBUG: Repaint image: {mt_repaint_image}")
        
        args = {
            "input_path": input_video_path,
            "prompt": f"\"{prompt}\"",
            "checkpoint_path": DEFAULT_MODEL_PATH,
            "output_dir": OUTPUT_DIR,
            "gpu": GPU_ID
        }
        
        # Priority: Custom Image > Yes > No
        if mt_repaint_image is not None:
            # Custom image takes precedence if provided
            repaint_path = save_uploaded_file(mt_repaint_image)
            print(f"DEBUG: Repaint path: {repaint_path}")
            args["repaint"] = repaint_path
        elif mt_repaint_option == "Yes":
            # Otherwise use Yes/No selection
            args["repaint"] = "true"
        
        # Create and run command
        cmd = create_run_command(args)
        output = run_process(cmd)
        
        # Find generated video files
        output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.mp4"))
        if output_files:
            # Sort by modification time, return the latest file
            latest_file = max(output_files, key=os.path.getmtime)
            return latest_file
        else:
            return None
    except Exception as e:
        import traceback
        print(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        return None

def process_camera_control(source, prompt, camera_motion, tracking_method, override_extrinsics):
    """Process camera control task"""
    try:
        # Save uploaded files
        input_media_path = save_uploaded_file(source)
        if input_media_path is None:
            return None
        
        print(f"DEBUG: Camera motion: '{camera_motion}'")
        print(f"DEBUG: Tracking method: '{tracking_method}'")
        print(f"DEBUG: Override extrinsics: '{override_extrinsics}'")
        
        args = {
            "input_path": input_media_path,
            "prompt": prompt,
            "checkpoint_path": DEFAULT_MODEL_PATH,
            "output_dir": OUTPUT_DIR,
            "gpu": GPU_ID,
            "tracking_method": tracking_method
        }
        
        if camera_motion and camera_motion.strip():
            args["camera_motion"] = camera_motion
            
        # 设置 override_extrinsics 参数
        if override_extrinsics == "Apply on top of extrinsics (preserve original camera)":
            args["override_extrinsics"] = "append"
        else:
            args["override_extrinsics"] = "override"
        
        # Create and run command
        cmd = create_run_command(args)
        output = run_process(cmd)
        
        # Find generated video files
        output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.mp4"))
        if output_files:
            # Sort by modification time, return the latest file
            latest_file = max(output_files, key=os.path.getmtime)
            return latest_file
        else:
            return None
    except Exception as e:
        import traceback
        print(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        return None

def process_object_manipulation(source, prompt, object_motion, object_mask, tracking_method):
    """Process object manipulation task"""
    try:
        # Save uploaded files
        input_image_path = save_uploaded_file(source)
        if input_image_path is None:
            return None
            
        object_mask_path = save_uploaded_file(object_mask)
        
        args = {
            "input_path": input_image_path,
            "prompt": prompt,
            "checkpoint_path": DEFAULT_MODEL_PATH,
            "output_dir": OUTPUT_DIR,
            "gpu": GPU_ID,
            "object_motion": object_motion,
            "object_mask": object_mask_path,
            "tracking_method": tracking_method
        }
        
        # Create and run command
        cmd = create_run_command(args)
        output = run_process(cmd)
        
        # Find generated video files
        output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.mp4"))
        if output_files:
            # Sort by modification time, return the latest file
            latest_file = max(output_files, key=os.path.getmtime)
            return latest_file
        else:
            return None
    except Exception as e:
        import traceback
        print(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        return None

def process_mesh_animation(source, prompt, tracking_video, ma_repaint_option, ma_repaint_image):
    """Process mesh animation task"""
    try:
        # Save uploaded files
        input_video_path = save_uploaded_file(source)
        if input_video_path is None:
            return None
            
        tracking_video_path = save_uploaded_file(tracking_video)
        if tracking_video_path is None:
            return None
        
        args = {
            "input_path": input_video_path,
            "prompt": prompt,
            "checkpoint_path": DEFAULT_MODEL_PATH,
            "output_dir": OUTPUT_DIR,
            "gpu": GPU_ID,
            "tracking_path": tracking_video_path
        }
        
        # Priority: Custom Image > Yes > No
        if ma_repaint_image is not None:
            # Custom image takes precedence if provided
            repaint_path = save_uploaded_file(ma_repaint_image)
            args["repaint"] = repaint_path
        elif ma_repaint_option == "Yes":
            # Otherwise use Yes/No selection
            args["repaint"] = "true"
        
        # Create and run command
        cmd = create_run_command(args)
        output = run_process(cmd)
        
        # Find generated video files
        output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.mp4"))
        if output_files:
            # Sort by modification time, return the latest file
            latest_file = max(output_files, key=os.path.getmtime)
            return latest_file
        else:
            return None
    except Exception as e:
        import traceback
        print(f"Processing failed: {str(e)}\n{traceback.format_exc()}")
        return None

# Create Gradio interface with updated layout
with gr.Blocks(title="Diffusion as Shader") as demo:
    gr.Markdown("# Diffusion as Shader Web UI")
    gr.Markdown("### [Project Page](https://igl-hkust.github.io/das/) | [GitHub](https://github.com/IGL-HKUST/DiffusionAsShader)")
    
    with gr.Row():
        left_column = gr.Column(scale=1)
        right_column = gr.Column(scale=1)

    with right_column:
        output_video = gr.Video(label="Generated Video")

    with left_column:
        source = gr.File(label="Source", file_types=["image", "video"])
        common_prompt = gr.Textbox(label="Prompt", lines=2)
        gr.Markdown(f"**Using GPU: {GPU_ID}**")
        
        with gr.Tabs() as task_tabs:
            # Motion Transfer tab
            with gr.TabItem("Motion Transfer"):
                gr.Markdown("## Motion Transfer")
                
                # Simplified controls - Radio buttons for Yes/No and separate file upload
                with gr.Row():
                    mt_repaint_option = gr.Radio(
                        label="Repaint First Frame",
                        choices=["No", "Yes"],
                        value="No"
                    )
                gr.Markdown("### Note: If you want to use your own image as repainted first frame, please upload the image in below.")
                # Custom image uploader (always visible)
                mt_repaint_image = gr.File(
                    label="Custom Repaint Image", 
                    file_types=["image"]
                )
                
                # Add run button for Motion Transfer tab
                mt_run_btn = gr.Button("Run Motion Transfer", variant="primary", size="lg")
                
                # Connect to process function
                mt_run_btn.click(
                    fn=process_motion_transfer,
                    inputs=[
                        source, common_prompt,
                        mt_repaint_option, mt_repaint_image
                    ],
                    outputs=[output_video]
                )
            
            # Camera Control tab
            with gr.TabItem("Camera Control"):
                gr.Markdown("## Camera Control")

                cc_camera_motion = gr.Textbox(
                    label="Current Camera Motion Sequence",
                    placeholder="Your camera motion sequence will appear here...",
                    interactive=False
                )

                cc_override_extrinsics = gr.Radio(
                    label="Camera Motion Application Mode",
                    choices=["Append (preserve original camera)", "Override (replace original camera) (experimental)"],
                    value="Append (preserve original camera)",
                    info="Controls how camera motion is applied: either replacing the original camera parameters or building upon them. Override is experimental and may not work as expected."
                )
                
                # Use tabs for different motion types
                with gr.Tabs() as cc_motion_tabs:
                    
                    # Translation tab
                    with gr.TabItem("Translation (trans)"):
                    
                        cc_trans_note = gr.Markdown("""
                        **Translation Notes:**
                        - Positive X: Move left, Negative X: Move right
                        - Positive Y: Move up, Negative Y: Move down
                        - Positive Z: Zoom out, Negative Z: Zoom in
                        """)

                        with gr.Row():
                            cc_trans_x = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.05, label="X-axis right- left+")
                            cc_trans_y = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.05, label="Y-axis up- down+")
                            cc_trans_z = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.05, label="Z-axis zoom_out- zoom_in+")
                        
                        with gr.Row():
                            cc_trans_start = gr.Number(minimum=0, maximum=48, value=0, step=1, label="Start Frame", precision=0)
                            cc_trans_end = gr.Number(minimum=0, maximum=48, value=48, step=1, label="End Frame", precision=0)
                        
                        # Add translation button in the Translation tab
                        cc_add_trans = gr.Button("Add Camera Translation", variant="secondary")
                        
                        # Function to add translation motion
                        def add_translation_motion(current_motion, trans_x, trans_y, trans_z, trans_start, trans_end):
                            # Format: trans dx dy dz [start_frame end_frame]
                            frame_range = f" {int(trans_start)} {int(trans_end)}" if trans_start != 0 or trans_end != 48 else ""
                            new_motion = f"trans {trans_x:.2f} {trans_y:.2f} {trans_z:.2f}{frame_range}"
                            
                            # Append to existing motion string with semicolon separator if needed
                            if current_motion and current_motion.strip():
                                updated_motion = f"{current_motion}; {new_motion}"
                            else:
                                updated_motion = new_motion
                            
                            return updated_motion
                        
                        # Connect translation button
                        cc_add_trans.click(
                            fn=add_translation_motion,
                            inputs=[
                                cc_camera_motion,
                                cc_trans_x, cc_trans_y, cc_trans_z, cc_trans_start, cc_trans_end
                            ],
                            outputs=[cc_camera_motion]
                        )
                    
                    # Rotation tab
                    with gr.TabItem("Rotation (rot)"):

                        cc_rot_note = gr.Markdown("""
                        **Rotation Notes:**
                        - X-axis rotation: positive X: pitch down, negative X: pitch up
                        - Y-axis rotation: positive Y: yaw left, negative Y: yaw right
                        - Z-axis rotation: positive Z: roll counter-clockwise, negative Z: roll clockwise
                        """)

                        with gr.Row():
                            cc_rot_axis = gr.Dropdown(choices=["x", "y", "z"], value="y", label="Rotation Axis")
                            cc_rot_angle = gr.Slider(minimum=-30, maximum=30, value=5, step=1, label="Rotation Angle (degrees)")
                        
                        with gr.Row():
                            cc_rot_start = gr.Number(minimum=0, maximum=48, value=0, step=1, label="Start Frame", precision=0)
                            cc_rot_end = gr.Number(minimum=0, maximum=48, value=48, step=1, label="End Frame", precision=0)
                        
                        # Add rotation button in the Rotation tab
                        cc_add_rot = gr.Button("Add Camera Rotation", variant="secondary")
                        
                        # Function to add rotation motion
                        def add_rotation_motion(current_motion, rot_axis, rot_angle, rot_start, rot_end):
                            # Format: rot axis angle [start_frame end_frame]
                            frame_range = f" {int(rot_start)} {int(rot_end)}" if rot_start != 0 or rot_end != 48 else ""
                            new_motion = f"rot {rot_axis} {rot_angle}{frame_range}"
                            
                            # Append to existing motion string with semicolon separator if needed
                            if current_motion and current_motion.strip():
                                updated_motion = f"{current_motion}; {new_motion}"
                            else:
                                updated_motion = new_motion
                            
                            return updated_motion
                        
                        # Connect rotation button
                        cc_add_rot.click(
                            fn=add_rotation_motion,
                            inputs=[
                                cc_camera_motion,
                                cc_rot_axis, cc_rot_angle, cc_rot_start, cc_rot_end
                            ],
                            outputs=[cc_camera_motion]
                        )
                
                # Add a clear button to reset the motion sequence
                cc_clear_motion = gr.Button("Clear All Motions", variant="stop")
                
                def clear_camera_motion():
                    return ""
                
                cc_clear_motion.click(
                    fn=clear_camera_motion,
                    inputs=[],
                    outputs=[cc_camera_motion]
                )

                cc_tracking_method = gr.Radio(
                    label="Tracking Method",
                    choices=["spatracker", "moge"],
                    value="moge"
                )
                
                # Add run button for Camera Control tab
                cc_run_btn = gr.Button("Run Camera Control", variant="primary", size="lg")
                
                # Connect to process function
                cc_run_btn.click(
                    fn=process_camera_control,
                    inputs=[
                        source, common_prompt,
                        cc_camera_motion, cc_tracking_method, cc_override_extrinsics
                    ],
                    outputs=[output_video]
                )
            
            # Object Manipulation tab
            with gr.TabItem("Object Manipulation"):
                gr.Markdown("## Object Manipulation")
                om_object_mask = gr.File(
                    label="Object Mask Image", 
                    file_types=["image"]
                )
                gr.Markdown("Upload a binary mask image, white areas indicate the object to manipulate")
                om_object_motion = gr.Dropdown(
                    label="Object Motion Type",
                    choices=["up", "down", "left", "right", "front", "back", "rot"],
                    value="up"
                )
                om_tracking_method = gr.Radio(
                    label="Tracking Method",
                    choices=["spatracker", "moge"],
                    value="moge"
                )
                
                # Add run button for Object Manipulation tab
                om_run_btn = gr.Button("Run Object Manipulation", variant="primary", size="lg")
                
                # Connect to process function
                om_run_btn.click(
                    fn=process_object_manipulation,
                    inputs=[
                        source, common_prompt,
                        om_object_motion, om_object_mask, om_tracking_method
                    ],
                    outputs=[output_video]
                )
            
            # Animating meshes to video tab
            with gr.TabItem("Animating meshes to video"):
                gr.Markdown("## Mesh Animation to Video")
                gr.Markdown("""
                    Note: Currently only supports tracking videos generated with Blender (version > 4.0).
                    Please run the script `scripts/blender.py` in your Blender project to generate tracking videos.
                """)
                ma_tracking_video = gr.File(
                    label="Tracking Video",
                    file_types=["video"]
                )
                gr.Markdown("Tracking video needs to be generated from Blender")
                
                # Simplified controls - Radio buttons for Yes/No and separate file upload
                with gr.Row():
                    ma_repaint_option = gr.Radio(
                        label="Repaint First Frame",
                        choices=["No", "Yes"],
                        value="No"
                    )
                gr.Markdown("### Note: If you want to use your own image as repainted first frame, please upload the image in below.")
                # Custom image uploader (always visible)
                ma_repaint_image = gr.File(
                    label="Custom Repaint Image", 
                    file_types=["image"]
                )
                
                # Add run button for Mesh Animation tab
                ma_run_btn = gr.Button("Run Mesh Animation", variant="primary", size="lg")
                
                # Connect to process function
                ma_run_btn.click(
                    fn=process_mesh_animation,
                    inputs=[
                        source, common_prompt,
                        ma_tracking_video, ma_repaint_option, ma_repaint_image
                    ],
                    outputs=[output_video]
                )

# Launch interface
if __name__ == "__main__":
    print(f"Using GPU: {GPU_ID}")
    print(f"Web UI will start on port {args.port}")
    if args.share:
        print("Creating public link for remote access")
    
    # Launch interface
    demo.launch(share=args.share, server_port=args.port) 