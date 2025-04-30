import os
import sys
import argparse
from PIL import Image
project_root = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(project_root, "submodules/MoGe"))
    sys.path.append(os.path.join(project_root, "submodules/vggt"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print("Warning: MoGe not found, motion transfer will not be applied")
    
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from diffusers.utils import load_image, load_video

from models.pipelines import DiffusionAsShaderPipeline, FirstFrameRepainter, CameraMotionGenerator, ObjectMotionGenerator
from submodules.MoGe.moge.model.v1 import MoGeModel
from submodules.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from submodules.vggt.vggt.models.vggt import VGGT

def load_media(media_path, max_frames=49, transform=None):
    """Load video or image frames and convert to tensor
    
    Args:
        media_path (str): Path to video or image file
        max_frames (int): Maximum number of frames to load
        transform (callable): Transform to apply to frames
        
    Returns:
        Tuple[torch.Tensor, float, bool]: Video tensor [T,C,H,W], FPS, and is_video flag
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
    
    # Determine if input is video or image based on extension
    ext = os.path.splitext(media_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']
    
    if is_video:
        # Load video file info
        video_clip = VideoFileClip(media_path)
        duration = video_clip.duration
        original_fps = video_clip.fps
        
        # Case 1: Video longer than 6 seconds, sample first 6 seconds + 1 frame
        if duration > 6.0:
            frames = load_video(media_path)
            fps = max_frames-1 / 6.0
        # Cases 2 and 3: Video shorter than 6 seconds
        else:
            # Load all frames
            frames = load_video(media_path)
            
            # Case 2: Total frames less than max_frames, need interpolation
            if len(frames) < max_frames:
                fps = len(frames) / duration  # Keep original fps
                
                # Evenly interpolate to max_frames
                indices = np.linspace(0, len(frames) - 1, max_frames)
                new_frames = []
                for i in indices:
                    idx = int(i)
                    new_frames.append(frames[idx])
                frames = new_frames
            # Case 3: Total frames more than max_frames but video less than 6 seconds
            else:
                # Evenly sample to max_frames
                indices = np.linspace(0, len(frames) - 1, max_frames)
                new_frames = []
                for i in indices:
                    idx = int(i)
                    new_frames.append(frames[idx])
                frames = new_frames
                fps = max_frames / duration  # New fps to maintain duration
    else:
        # Handle image as single frame
        image = load_image(media_path)
        frames = [image]
        fps = 8  # Default fps for images
        
        # Duplicate frame to max_frames
        while len(frames) < max_frames:
            frames.append(frames[0].copy())

    if len(frames) > max_frames:
        frames = frames[:max_frames]
    
    # Convert frames to tensor
    video_tensor = torch.stack([transform(frame) for frame in frames])
    
    return video_tensor, fps, is_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Path to input video/image')
    parser.add_argument('--prompt', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, default="EXCAI/Diffusion-As-Shader", help='Path to model checkpoint')
    parser.add_argument('--depth_path', type=str, default=None, help='Path to depth image')
    parser.add_argument('--tracking_path', type=str, default=None, help='Path to tracking video, if provided, camera motion and object manipulation will not be applied')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--repaint', type=str, default=None, 
                       help='Path to repainted image, or "true" to perform repainting, if not provided use original frame')
    parser.add_argument('--camera_motion', type=str, default=None, 
                    help='Camera motion mode: "trans <dx> <dy> <dz>" or "rot <axis> <angle>" or "spiral <radius>"')
    parser.add_argument('--override_extrinsics', type=str, default="append", choices=["override", "append"],
                help='How to apply camera motion: "override" to replace original camera, "append" to build upon it. Override is experimental and may not work as expected.')
    parser.add_argument('--object_motion', type=str, default=None, help='Object motion mode: up/down/left/right')
    parser.add_argument('--object_mask', type=str, default=None, help='Path to object mask image (binary image)')
    parser.add_argument('--tracking_method', type=str, default='spatracker', choices=['spatracker', 'moge', 'cotracker'], 
                    help='Tracking method to use (spatracker, cotracker or moge)')
    args = parser.parse_args()
    
    # Load input video/image
    video_tensor, fps, is_video = load_media(args.input_path)
    if not is_video:
        args.tracking_method = "moge"
        print("Image input detected, using MoGe for tracking video generation.")

    # Initialize pipeline
    das = DiffusionAsShaderPipeline(gpu_id=args.gpu, output_dir=args.output_dir)
    das.fps = fps
    if args.tracking_method == "moge" and args.tracking_path is None:
        moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(das.device)
    
    # Repaint first frame if requested
    repaint_img_tensor = None
    if args.repaint:
        if args.repaint.lower() == "true":
            repainter = FirstFrameRepainter(gpu_id=args.gpu, output_dir=args.output_dir)
            repaint_img_tensor = repainter.repaint(
                video_tensor[0], 
                prompt=args.prompt,
                depth_path=args.depth_path
            )
        else:
            repaint_img_tensor, _, _ = load_media(args.repaint)
            repaint_img_tensor = repaint_img_tensor[0]  # Take first frame

    # Generate tracking if not provided
    tracking_tensor = None
    pred_tracks = None
    cam_motion = CameraMotionGenerator(args.camera_motion)

    if args.tracking_path:
        tracking_tensor, _, _ = load_media(args.tracking_path)
        
    elif args.tracking_method == "moge":
        # Use the first frame from previously loaded video_tensor
        infer_result = moge.infer(video_tensor[0].to(das.device))  # [C, H, W] in range [0,1]
        H, W = infer_result["points"].shape[0:2]
        pred_tracks = infer_result["points"].unsqueeze(0).repeat(49, 1, 1, 1) #[T, H, W, 3]
        cam_motion.set_intr(infer_result["intrinsics"])

        # Apply object motion if specified
        if args.object_motion:
            if args.object_mask is None:
                raise ValueError("Object motion specified but no mask provided. Please provide a mask image with --object_mask")
                
            # Load mask image
            mask_image = Image.open(args.object_mask).convert('L')  # Convert to grayscale
            mask_image = transforms.Resize((480, 720))(mask_image)  # Resize to match video size
            # Convert to binary mask
            mask = torch.from_numpy(np.array(mask_image) > 127)  # Threshold at 127
            
            motion_generator = ObjectMotionGenerator(device=das.device)

            pred_tracks = motion_generator.apply_motion(
                pred_tracks=pred_tracks,
                mask=mask,
                motion_type=args.object_motion,
                distance=50,
                num_frames=49,
                tracking_method="moge"
            )
            print("Object motion applied")

        # Apply camera motion if specified
        if args.camera_motion:
            poses = cam_motion.get_default_motion() # shape: [49, 4, 4]
            print("Camera motion applied")
        else:
            # no poses
            poses = torch.eye(4).unsqueeze(0).repeat(49, 1, 1)
        # change pred_tracks into screen coordinate
        pred_tracks_flatten = pred_tracks.reshape(video_tensor.shape[0], H*W, 3)
        pred_tracks = cam_motion.w2s_moge(pred_tracks_flatten, poses).reshape([video_tensor.shape[0], H, W, 3]) # [T, H, W, 3]
        _, tracking_tensor = das.visualize_tracking_moge(
            pred_tracks.cpu().numpy(), 
            infer_result["mask"].cpu().numpy()
        )
        print('export tracking video via MoGe.')

    else:

        if args.tracking_method == "cotracker":
            pred_tracks, pred_visibility = das.generate_tracking_cotracker(video_tensor) # T N 3, T N
        else:
            pred_tracks, pred_visibility, T_Firsts = das.generate_tracking_spatracker(video_tensor) # T N 3, T N, B N

        # Preprocess video tensor to match VGGT requirements
        t, c, h, w = video_tensor.shape
        new_width = 518
        new_height = round(h * (new_width / w) / 14) * 14
        resize_transform = transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC)
        video_vggt = resize_transform(video_tensor)  # [T, C, H, W]
        
        if new_height > 518:
            start_y = (new_height - 518) // 2
            video_vggt = video_vggt[:, :, start_y:start_y + 518, :]

        # Get extrinsic and intrinsic matrices
        vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(das.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=das.dtype):

                video_vggt = video_vggt.unsqueeze(0)  # [1, T, C, H, W]
                aggregated_tokens_list, ps_idx = vggt_model.aggregator(video_vggt.to(das.device))
            
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extr, intr = pose_encoding_to_extri_intri(vggt_model.camera_head(aggregated_tokens_list)[-1], video_vggt.shape[-2:])
                depth_map, depth_conf = vggt_model.depth_head(aggregated_tokens_list, video_vggt, ps_idx)
        
        cam_motion.set_intr(intr)
        cam_motion.set_extr(extr)

        del vggt_model

        # Apply camera motion if specified
        if args.camera_motion:
            poses = cam_motion.get_default_motion() # shape: [49, 4, 4]
            pred_tracks_world = cam_motion.s2w_vggt(pred_tracks, extr, intr)
            pred_tracks = cam_motion.w2s_vggt(pred_tracks_world, extr, intr, poses, 
                                 override_extrinsics=(args.override_extrinsics == "override"))
            print("Camera motion applied")
        
        # Apply object motion if specified
        if args.object_motion:
            if args.object_mask is None:
                raise ValueError("Object motion specified but no mask provided. Please provide a mask image with --object_mask")
                
            # Load mask image
            mask_image = Image.open(args.object_mask).convert('L')  # Convert to grayscale
            mask_image = transforms.Resize((480, 720))(mask_image)  # Resize to match video size
            # Convert to binary mask
            mask = torch.from_numpy(np.array(mask_image) > 127)  # Threshold at 127
            
            motion_generator = ObjectMotionGenerator(device=das.device)
            
            pred_tracks = motion_generator.apply_motion(
                pred_tracks=pred_tracks,
                mask=mask,
                motion_type=args.object_motion,
                distance=50,
                num_frames=49,
                tracking_method="spatracker"
            ).unsqueeze(0)
            print(f"Object motion '{args.object_motion}' applied using mask from {args.object_mask}")
    
        if args.tracking_method == "cotracker":
            _, tracking_tensor = das.visualize_tracking_cotracker(pred_tracks, pred_visibility)
        else:
            _, tracking_tensor = das.visualize_tracking_spatracker(video_tensor, pred_tracks, pred_visibility, T_Firsts)
    
    das.apply_tracking(
        video_tensor=video_tensor,
        fps=fps,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=repaint_img_tensor,
        prompt=args.prompt,
        checkpoint_path=args.checkpoint_path,
        num_inference_steps=args.num_inference_steps
    )
