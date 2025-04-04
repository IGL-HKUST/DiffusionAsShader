import os
import sys
import math
from tqdm import tqdm
from PIL import Image, ImageDraw
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from diffusers import FluxControlPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking

from submodules.MoGe.moge.model.v1 import MoGeModel

from image_gen_aux import DepthPreprocessor
from moviepy.editor import ImageSequenceClip

class DiffusionAsShaderPipeline:
    def __init__(self, gpu_id=0, output_dir='outputs'):
        """Initialize MotionTransfer class
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        # video parameters
        self.max_depth = 65.0
        self.fps = 8

        # camera parameters
        self.camera_motion=None
        self.fov=55

        # device
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        self.dtype = torch.bfloat16

        # files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])

    @torch.no_grad()
    def _infer(
        self, 
        prompt: str,
        model_path: str,
        tracking_tensor: torch.Tensor = None,
        image_tensor: torch.Tensor = None,  # [C,H,W] in range [0,1]
        output_path: str = "./output.mp4",
        num_inference_steps: int = 25,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        fps: int = 24,
        seed: int = 42,
    ):
        """
        Generates a video based on the given prompt and saves it to the specified path.

        Parameters:
        - prompt (str): The description of the video to be generated.
        - model_path (str): The path of the pre-trained model to be used.
        - tracking_tensor (torch.Tensor): Tracking video tensor [T, C, H, W] in range [0,1]
        - image_tensor (torch.Tensor): Input image tensor [C, H, W] in range [0,1]
        - output_path (str): The path where the generated video will be saved.
        - num_inference_steps (int): Number of steps for the inference process.
        - guidance_scale (float): The scale for classifier-free guidance.
        - num_videos_per_prompt (int): Number of videos to generate per prompt.
        - dtype (torch.dtype): The data type for computation.
        - seed (int): The seed for reproducibility.
        """
        from transformers import T5EncoderModel, T5Tokenizer
        from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
        from models.cogvideox_tracking import CogVideoXTransformer3DModelTracking
        
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
        transformer = CogVideoXTransformer3DModelTracking.from_pretrained(model_path, subfolder="transformer")
        scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        pipe = CogVideoXImageToVideoPipelineTracking(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler
        )
        
        # Convert tensor to PIL Image
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        height, width = image.height, image.width

        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        self.dtype = dtype

        # Process tracking tensor
        tracking_maps = tracking_tensor.float() # [T, C, H, W]
        tracking_maps = tracking_maps.to(device=self.device, dtype=dtype)
        tracking_first_frame = tracking_maps[0:1]  # Get first frame as [1, C, H, W]
        height, width = tracking_first_frame.shape[2], tracking_first_frame.shape[3]

        # 2. Set Scheduler.
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        pipe.to(self.device, dtype=dtype)
        # pipe.enable_sequential_cpu_offload()

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.vae.eval()

        pipe.transformer.gradient_checkpointing = False
        
        print("Encoding tracking maps")
        tracking_maps = tracking_maps.unsqueeze(0) # [B, T, C, H, W]
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
        tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
        tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        # 4. Generate the video frames based on the prompt.
        video_generate = pipe(
            prompt=prompt,
            negative_prompt="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            tracking_maps=tracking_maps,
            tracking_image=tracking_first_frame,
            height=height,
            width=width,
        ).frames[0]
        
        # 5. Export the generated frames to a video file. fps must be 8 for original video.
        output_path = output_path if output_path else f"result.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        export_to_video(video_generate, output_path, fps=fps)
        
    #========== camera parameters ==========#

    def _set_camera_motion(self, camera_motion):
        self.camera_motion = camera_motion
    
    ##============= SpatialTracker =============##
    
    def generate_tracking_spatracker(self, video_tensor, density=70):
        """Generate tracking video
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            
        Returns:
            str: Path to tracking video
        """
        print("Loading tracking models...")
        # Load tracking model
        tracker = SpaTrackerPredictor(
            checkpoint=os.path.join(project_root, 'checkpoints/spaT_final.pth'),
            interp_shape=(384, 576),
            seq_length=12
        ).to(self.device)
        
        # Load depth model
        self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti").to(self.device)
        
        try:
            video = video_tensor.unsqueeze(0).to(self.device)
            
            video_depths = []
            for i in range(video_tensor.shape[0]):
                frame = (video_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                depth = self.depth_preprocessor(Image.fromarray(frame))[0]
                depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
                video_depths.append(depth_tensor)
            video_depth = torch.stack(video_depths, dim=0).to(self.device)
            # print("Video depth shape:", video_depth.shape)
            
            segm_mask = np.ones((480, 720), dtype=np.uint8)
            
            pred_tracks, pred_visibility, T_Firsts = tracker(
                video * 255, 
                video_depth=video_depth,
                grid_size=density,
                backward_tracking=False,
                depth_predictor=None,
                grid_query_frame=0,
                segm_mask=torch.from_numpy(segm_mask)[None, None].to(self.device),
                wind_length=12,
                progressive_tracking=False
            )

            return pred_tracks.squeeze(0), pred_visibility.squeeze(0), T_Firsts
            
        finally:
            # Clean up GPU memory
            del tracker, self.depth_preprocessor
            torch.cuda.empty_cache()

    def visualize_tracking_spatracker(self, video, pred_tracks, pred_visibility, T_Firsts, save_tracking=True):
        video = video.unsqueeze(0).to(self.device)
        pred_tracks = pred_tracks.unsqueeze(0).detach().cpu()
        pred_visibility = pred_visibility.unsqueeze(0).detach().cpu()
        vis = Visualizer(save_dir=self.output_dir, grayscale=False, fps=24, pad_value=0)
        
        T_Firsts = T_Firsts.detach().cpu() if isinstance(T_Firsts, torch.Tensor) else T_Firsts
        msk_query = (T_Firsts == 0)
        
        pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
        pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
        
        tracking_video = vis.visualize(video=video, tracks=pred_tracks,
                        visibility=pred_visibility, save_video=False,
                        filename="temp")
        
        tracking_video = tracking_video.squeeze(0) # [T, C, H, W]
        wide_list = list(tracking_video.unbind(0))
        wide_list = [wide.permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        clip = ImageSequenceClip(wide_list, fps=self.fps)

        tracking_path = None
        if save_tracking:
            try:
                tracking_path = os.path.join(self.output_dir, "tracking_video.mp4")
                clip.write_videofile(tracking_path, codec="libx264", fps=self.fps, logger=None)
                print(f"Video saved to {tracking_path}")
            except Exception as e:
                print(f"Warning: Failed to save tracking video: {e}")
                tracking_path = None
        
        # Convert tracking_video back to tensor in range [0,1]
        tracking_frames = np.array(list(clip.iter_frames())) / 255.0
        tracking_video = torch.from_numpy(tracking_frames).permute(0, 3, 1, 2).float()
        
        return tracking_path, tracking_video
    
    ##============= MoGe =============##

    def valid_mask(self, pixels, W, H):
        """Check if pixels are within valid image bounds
        
        Args:
            pixels (numpy.ndarray): Pixel coordinates of shape [N, 2]
            W (int): Image width
            H (int): Image height
            
        Returns:
            numpy.ndarray: Boolean mask of valid pixels
        """
        return ((pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] > 0) & \
                 (pixels[:, 1] < H))

    def sort_points_by_depth(self, points, depths):
        """Sort points by depth values
        
        Args:
            points (numpy.ndarray): Points array of shape [N, 2]
            depths (numpy.ndarray): Depth values of shape [N]
            
        Returns:
            tuple: (sorted_points, sorted_depths, sort_index)
        """
        # Combine points and depths into a single array for sorting
        combined = np.hstack((points, depths[:, None]))  # Nx3 (points + depth)
        # Sort by depth (last column) in descending order
        sort_index = combined[:, -1].argsort()[::-1]
        sorted_combined = combined[sort_index]
        # Split back into points and depths
        sorted_points = sorted_combined[:, :-1]
        sorted_depths = sorted_combined[:, -1]
        return sorted_points, sorted_depths, sort_index

    def draw_rectangle(self, rgb, coord, side_length, color=(255, 0, 0)):
        """Draw a rectangle on the image
        
        Args:
            rgb (PIL.Image): Image to draw on
            coord (tuple): Center coordinates (x, y)
            side_length (int): Length of rectangle sides
            color (tuple): RGB color tuple
        """
        draw = ImageDraw.Draw(rgb)
        # Calculate the bounding box of the rectangle
        left_up_point = (coord[0] - side_length//2, coord[1] - side_length//2)  
        right_down_point = (coord[0] + side_length//2, coord[1] + side_length//2)
        color = tuple(list(color))

        draw.rectangle(
            [left_up_point, right_down_point],
            fill=tuple(color),
            outline=tuple(color),
        )
    
    def visualize_tracking_moge(self, points, mask, save_tracking=True):
        """Visualize tracking results from MoGe model
        
        Args:
            points (numpy.ndarray): Points array of shape [T, H, W, 3]
            mask (numpy.ndarray): Binary mask of shape [H, W]
            save_tracking (bool): Whether to save tracking video
            
        Returns:
            tuple: (tracking_path, tracking_video)
                - tracking_path (str): Path to saved tracking video, None if save_tracking is False
                - tracking_video (torch.Tensor): Tracking visualization tensor of shape [T, C, H, W] in range [0,1]
        """
        # Create color array
        T, H, W, _ = points.shape
        colors = np.zeros((H, W, 3), dtype=np.uint8)

        # Set R channel - based on x coordinates (smaller on the left)
        colors[:, :, 0] = np.tile(np.linspace(0, 255, W), (H, 1))

        # Set G channel - based on y coordinates (smaller on the top)
        colors[:, :, 1] = np.tile(np.linspace(0, 255, H), (W, 1)).T

        # Set B channel - based on depth
        z_values = points[0, :, :, 2]  # get z values
        inv_z = 1 / z_values  # calculate 1/z
        # Calculate 2% and 98% percentiles
        p2 = np.percentile(inv_z, 2)
        p98 = np.percentile(inv_z, 98)
        # Normalize to [0,1] range
        normalized_z = np.clip((inv_z - p2) / (p98 - p2), 0, 1)
        colors[:, :, 2] = (normalized_z * 255).astype(np.uint8)
        colors = colors.astype(np.uint8)
        
        points = points.reshape(T, -1, 3)
        colors = colors.reshape(-1, 3)
        
        # Initialize list to store frames
        frames = []
        
        for i, pts_i in enumerate(tqdm(points, desc="rendering frames")):
            pixels, depths = pts_i[..., :2], pts_i[..., 2]
            pixels[..., 0] = pixels[..., 0] * W
            pixels[..., 1] = pixels[..., 1] * H
            pixels = pixels.astype(int)
            
            valid = self.valid_mask(pixels, W, H)
            frame_rgb = colors[valid]
            pixels = pixels[valid]
            depths = depths[valid]
            
            img = Image.fromarray(np.uint8(np.zeros([H, W, 3])), mode="RGB")
            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            step = 1
            sorted_pixels = sorted_pixels[::step]
            sorted_rgb = frame_rgb[sort_index][::step]
            
            for j in range(sorted_pixels.shape[0]):
                self.draw_rectangle(
                    img,
                    coord=(sorted_pixels[j, 0], sorted_pixels[j, 1]),
                    side_length=2,
                    color=sorted_rgb[j],
                )
            frames.append(np.array(img))

        # Convert frames to video tensor in range [0,1]
        tracking_video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

        tracking_path = None
        if save_tracking:
            try:
                tracking_path = os.path.join(self.output_dir, "tracking_video_moge.mp4")
                # Convert back to uint8 for saving
                uint8_frames = [frame.astype(np.uint8) for frame in frames]
                clip = ImageSequenceClip(uint8_frames, fps=self.fps)
                clip.write_videofile(tracking_path, codec="libx264", fps=self.fps, logger=None)
                print(f"Video saved to {tracking_path}")
            except Exception as e:
                print(f"Warning: Failed to save tracking video: {e}")
                tracking_path = None

        return tracking_path, tracking_video


    ##============= CoTracker =============##

    def generate_tracking_cotracker(self, video_tensor, density=70):
        """Generate tracking video
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            
        Returns:
            tuple: (pred_tracks, pred_visibility)
                - pred_tracks (torch.Tensor): Tracking points with depth [T, N, 3]
                - pred_visibility (torch.Tensor): Visibility mask [T, N, 1]
        """
        # Generate tracking points
        if not hasattr(self, 'cotracker') or self.cotracker is None:
            self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
        
        # Load depth model
        if not hasattr(self, 'depth_preprocessor') or self.depth_preprocessor is None:
            self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti").to(self.device)
        
        try:
            video = video_tensor.unsqueeze(0).to(self.device)
            
            # Process all frames to get depth maps
            video_depths = []
            for i in tqdm(range(video_tensor.shape[0]), desc="estimating depth"):
                frame = (video_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                depth = self.depth_preprocessor(Image.fromarray(frame))[0]
                depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
                video_depths.append(depth_tensor)
            
            video_depth = torch.stack(video_depths, dim=0).to(self.device)  # [T, 1, H, W]
            
            # Get tracking points and visibility
            print("tracking...")
            pred_tracks, pred_visibility = self.cotracker(video, grid_size=density)  # B T N 2,  B T N 1
            
            # Extract dimensions
            B, T, N, _ = pred_tracks.shape
            H, W = video_depth.shape[2], video_depth.shape[3]
            
            # Create output tensor with depth
            pred_tracks_with_depth = torch.zeros((B, T, N, 3), device=self.device)
            pred_tracks_with_depth[:, :, :, :2] = pred_tracks  # Copy x,y coordinates
            
            # Vectorized approach to get depths for all points
            # Reshape pred_tracks to process all batches and frames at once
            flat_tracks = pred_tracks.reshape(B*T, N, 2)
            
            # Clamp coordinates to valid image bounds
            x_coords = flat_tracks[:, :, 0].clamp(0, W-1).long()  # [B*T, N]
            y_coords = flat_tracks[:, :, 1].clamp(0, H-1).long()  # [B*T, N]
            
            # Get depths for all points at once
            # For each point in the flattened batch, get its depth from the corresponding frame
            depths = torch.zeros((B*T, N), device=self.device)
            for bt in range(B*T):
                t = bt % T  # Time index
                depths[bt] = video_depth[t, 0, y_coords[bt], x_coords[bt]]
            
            # Reshape depths back to [B, T, N] and assign to output tensor
            pred_tracks_with_depth[:, :, :, 2] = depths.reshape(B, T, N)

            return pred_tracks_with_depth.squeeze(0), pred_visibility.squeeze(0)
            
        finally:
            del self.cotracker
            del self.depth_preprocessor
            torch.cuda.empty_cache()

    def visualize_tracking_cotracker(self, points, vis_mask=None, save_tracking=True, point_wise=4, video_size=(480, 720)):
        """Visualize tracking results from CoTracker
        
        Args:
            points (torch.Tensor): Points array of shape [T, N, 3]
            vis_mask (torch.Tensor): Visibility mask of shape [T, N, 1]
            save_tracking (bool): Whether to save tracking video
            point_wise (int): Size of points in visualization
            video_size (tuple): Render size (height, width)
            
        Returns:
            tuple: (tracking_path, tracking_video)
        """
        # Move tensors to CPU and convert to numpy
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if vis_mask is not None and isinstance(vis_mask, torch.Tensor):
            vis_mask = vis_mask.detach().cpu().numpy()
            # Reshape if needed
            if vis_mask.ndim == 3 and vis_mask.shape[2] == 1:
                vis_mask = vis_mask.squeeze(-1)
        
        T, N, _ = points.shape
        H, W = video_size
        
        if vis_mask is None:
            vis_mask = np.ones((T, N), dtype=bool)
        
        colors = np.zeros((N, 3), dtype=np.uint8)
        
        first_frame_pts = points[0]
        
        u_min, u_max = 0, W
        u_normalized = np.clip((first_frame_pts[:, 0] - u_min) / (u_max - u_min), 0, 1)
        colors[:, 0] = (u_normalized * 255).astype(np.uint8)
        
        v_min, v_max = 0, H
        v_normalized = np.clip((first_frame_pts[:, 1] - v_min) / (v_max - v_min), 0, 1)
        colors[:, 1] = (v_normalized * 255).astype(np.uint8)
        
        z_values = first_frame_pts[:, 2]
        if np.all(z_values == 0):
            colors[:, 2] = np.random.randint(0, 256, N, dtype=np.uint8)
        else:
            inv_z = 1 / (z_values + 1e-10)
            p2 = np.percentile(inv_z, 2)
            p98 = np.percentile(inv_z, 98)
            normalized_z = np.clip((inv_z - p2) / (p98 - p2 + 1e-10), 0, 1)
            colors[:, 2] = (normalized_z * 255).astype(np.uint8)
        
        frames = []
        
        for i in tqdm(range(T), desc="rendering frames"):
            pts_i = points[i]
            
            visibility = vis_mask[i]
            
            pixels, depths = pts_i[visibility, :2], pts_i[visibility, 2]
            pixels = pixels.astype(int)
            
            in_frame = self.valid_mask(pixels, W, H) 
            pixels = pixels[in_frame]
            depths = depths[in_frame]
            frame_rgb = colors[visibility][in_frame]
            
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), mode="RGB")
            
            sorted_pixels, _, sort_index = self.sort_points_by_depth(pixels, depths)
            sorted_rgb = frame_rgb[sort_index]
            
            for j in range(sorted_pixels.shape[0]):
                self.draw_rectangle(
                    img,
                    coord=(sorted_pixels[j, 0], sorted_pixels[j, 1]),
                    side_length=point_wise,
                    color=sorted_rgb[j],
                )
            
            frames.append(np.array(img))
        
        # Convert frames to video tensor in range [0,1]
        tracking_video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0

        tracking_path = None
        if save_tracking:
            try:
                tracking_path = os.path.join(self.output_dir, "tracking_video_cotracker.mp4")
                # Convert back to uint8 for saving
                uint8_frames = [frame.astype(np.uint8) for frame in frames]
                clip = ImageSequenceClip(uint8_frames, fps=self.fps)
                clip.write_videofile(tracking_path, codec="libx264", fps=self.fps, logger=None)
                print(f"Video saved to {tracking_path}")
            except Exception as e:
                print(f"Warning: Failed to save tracking video: {e}")
                tracking_path = None

        return tracking_path, tracking_video

    
    def apply_tracking(self, video_tensor, fps=8, tracking_tensor=None, img_cond_tensor=None, prompt=None, checkpoint_path=None, num_inference_steps=50):
        """Generate final video with motion transfer
        
        Args:
            video_tensor (torch.Tensor): Input video tensor [T,C,H,W]
            fps (float): Input video FPS
            tracking_tensor (torch.Tensor): Tracking video tensor [T,C,H,W]
            image_tensor (torch.Tensor): First frame tensor [C,H,W] to use for generation
            prompt (str): Generation prompt
            checkpoint_path (str): Path to model checkpoint
        """
        self.fps = fps

        # Use first frame if no image provided
        if img_cond_tensor is None:
            img_cond_tensor = video_tensor[0]
        
        # Generate final video
        final_output = os.path.join(os.path.abspath(self.output_dir), "result.mp4")
        self._infer(
            prompt=prompt,
            model_path=checkpoint_path,
            tracking_tensor=tracking_tensor,
            image_tensor=img_cond_tensor,
            output_path=final_output,
            num_inference_steps=50,
            guidance_scale=6.0,
            dtype=torch.bfloat16,
            fps=self.fps
        )
        print(f"Final video generated successfully at: {final_output}")

    def _set_object_motion(self, motion_type):
        """Set object motion type
        
        Args:
            motion_type (str): Motion direction ('up', 'down', 'left', 'right')
        """
        self.object_motion = motion_type

class FirstFrameRepainter:
    def __init__(self, gpu_id=0, output_dir='outputs'):
        """Initialize FirstFrameRepainter
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        self.device = f"cuda:{gpu_id}"
        self.output_dir = output_dir
        self.max_depth = 65.0
        os.makedirs(output_dir, exist_ok=True)
        
    def repaint(self, image_tensor, prompt, depth_path=None, method="dav"):
        """Repaint first frame using Flux
        
        Args:
            image_tensor (torch.Tensor): Input image tensor [C,H,W]
            prompt (str): Repaint prompt
            depth_path (str): Path to depth image
            method (str): depth estimator, "moge" or "dav" or "zoedepth"
            
        Returns:
            torch.Tensor: Repainted image tensor [C,H,W]
        """
        print("Loading Flux model...")
        # Load Flux model
        flux_pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev", 
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Get depth map
        if depth_path is None:
            if method == "moge":
                self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
                depth_map = self.moge_model.infer(image_tensor.to(self.device))["depth"]
                depth_map = torch.clamp(depth_map, max=self.max_depth)
                depth_normalized = 1.0 - (depth_map / self.max_depth)
                depth_rgb = (depth_normalized * 255).cpu().numpy().astype(np.uint8)
                control_image = Image.fromarray(depth_rgb).convert("RGB")
            elif method == "zoedepth":
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
                self.depth_preprocessor.to(self.device)
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(image_np))[0].convert("RGB")
                control_image = control_image.point(lambda x: 255 - x) # the zoedepth depth is inverted
            else:
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                self.depth_preprocessor.to(self.device)
                image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(image_np))[0].convert("RGB")
        else:
            control_image = Image.open(depth_path).convert("RGB")

        try:
            repainted_image = flux_pipe(
                prompt=prompt,
                control_image=control_image,
                height=480,
                width=720,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # Save repainted image
            repainted_image.save(os.path.join(self.output_dir, "temp_repainted.png"))
            
            # Convert PIL Image to tensor
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            repainted_tensor = transform(repainted_image)
            
            return repainted_tensor
            
        finally:
            # Clean up GPU memory
            del flux_pipe
            if method == "moge":
                del self.moge_model
            else:
                del self.depth_preprocessor
            torch.cuda.empty_cache()

class CameraMotionGenerator:
    def __init__(self, motion_type, frame_num=49, H=480, W=720, fx=None, fy=None, fov=55, device='cuda'):
        self.motion_type = motion_type
        self.frame_num = frame_num
        self.fov = fov
        self.device = device
        self.W = W
        self.H = H
        self.intr = torch.tensor([
            [0, 0, W / 2],
            [0, 0, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        # if fx, fy not provided
        if not fx or not fy:
            fov_rad = math.radians(fov)
            fx = fy = (W / 2) / math.tan(fov_rad / 2)
 
        self.intr[0, 0] = fx
        self.intr[1, 1] = fy   

        self.extr = torch.eye(4, device=device)

    def s2w_vggt(self, points, extrinsics, intrinsics):
        """
        Transform points from pixel coordinates to world coordinates
        
        Args:
            points: Point cloud data of shape [T, N, 3] in uvz format
            extrinsics: Camera extrinsic matrices [B, T, 3, 4] or [T, 3, 4]
            intrinsics: Camera intrinsic matrices [B, T, 3, 3] or [T, 3, 3]
            
        Returns:
            world_points: Point cloud in world coordinates [T, N, 3]
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.detach().cpu().numpy()
            # Handle batch dimension
            if extrinsics.ndim == 4:  # [B, T, 3, 4]
                extrinsics = extrinsics[0]  # Take first batch
            
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
            # Handle batch dimension
            if intrinsics.ndim == 4:  # [B, T, 3, 3]
                intrinsics = intrinsics[0]  # Take first batch
            
        T, N, _ = points.shape
        world_points = np.zeros_like(points)
        
        # Extract uvz coordinates
        uvz = points
        valid_mask = uvz[..., 2] > 0
        
        # Create homogeneous coordinates [u, v, 1]
        uv_homogeneous = np.concatenate([uvz[..., :2], np.ones((T, N, 1))], axis=-1)
        
        # Transform from pixel to camera coordinates
        for i in range(T):
            K = intrinsics[i]
            K_inv = np.linalg.inv(K)
            
            R = extrinsics[i, :, :3]
            t = extrinsics[i, :, 3]
            
            R_inv = np.linalg.inv(R)
            
            valid_indices = np.where(valid_mask[i])[0]
            
            if len(valid_indices) > 0:
                valid_uv = uv_homogeneous[i, valid_indices]
                valid_z = uvz[i, valid_indices, 2]
                
                valid_xyz_camera = valid_uv @ K_inv.T
                valid_xyz_camera = valid_xyz_camera * valid_z[:, np.newaxis]
                
                # Transform from camera to world coordinates: X_world = R^-1 * (X_camera - t)
                valid_world_points = (valid_xyz_camera - t) @ R_inv.T
                
                world_points[i, valid_indices] = valid_world_points
        
        return world_points

    def w2s_vggt(self, world_points, extrinsics, intrinsics, poses=None, override_extrinsics=True):
        """
        Project points from world coordinates to camera view
        
        Args:
            world_points: Point cloud in world coordinates [T, N, 3]
            extrinsics: Original camera extrinsic matrices [B, T, 3, 4] or [T, 3, 4]
            intrinsics: Camera intrinsic matrices [B, T, 3, 3] or [T, 3, 3]
            poses: Camera pose matrices [T, 4, 4], if None use first frame extrinsics
            override_extrinsics: If True, replace extrinsics with poses; if False, apply poses on top of extrinsics
            
        Returns:
            camera_points: Point cloud in camera coordinates [T, N, 3] in uvz format
        """
        if isinstance(world_points, torch.Tensor):
            world_points = world_points.detach().cpu().numpy()
            
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.detach().cpu().numpy()
            if extrinsics.ndim == 4:
                extrinsics = extrinsics[0]
            
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
            if intrinsics.ndim == 4:
                intrinsics = intrinsics[0]
            
        T, N, _ = world_points.shape
        
        # If no poses provided, use first frame extrinsics
        if poses is None:
            pose1 = np.eye(4)
            pose1[:3, :3] = extrinsics[0, :, :3]
            pose1[:3, 3] = extrinsics[0, :, 3]
            
            camera_poses = np.tile(pose1[np.newaxis, :, :], (T, 1, 1))
        else:
            if isinstance(poses, torch.Tensor):
                camera_poses = poses.cpu().numpy()
            else:
                camera_poses = poses
            
            # Scale translation by 1/5
            scaled_poses = camera_poses.copy()
            scaled_poses[:, :3, 3] = camera_poses[:, :3, 3] / 5.0
            camera_poses = scaled_poses
            
            # If not overriding extrinsics, combine poses with original extrinsics
            if not override_extrinsics and poses is not None:
                for i in range(T):
                    # Convert extrinsics to 4x4 matrix
                    ext_mat = np.eye(4)
                    ext_mat[:3, :3] = extrinsics[i, :, :3]
                    ext_mat[:3, 3] = extrinsics[i, :, 3]
                    
                    # Combine pose with extrinsics: pose * extrinsics
                    combined = np.matmul(camera_poses[i], ext_mat)
                    
                    # Update camera_poses
                    camera_poses[i] = combined
        
        # Add homogeneous coordinates
        ones = np.ones([T, N, 1])
        world_points_hom = np.concatenate([world_points, ones], axis=-1)
        
        # Transform points using batch matrix multiplication
        pts_cam_hom = np.matmul(world_points_hom, np.transpose(camera_poses, (0, 2, 1)))
        pts_cam = pts_cam_hom[..., :3]
        
        # Extract depth information
        depths = pts_cam[..., 2:3]
        valid_mask = depths[..., 0] > 0
        
        # Normalize coordinates
        normalized_pts = pts_cam / (depths + 1e-10)
        
        # Apply intrinsic matrix for projection
        pts_pixel = np.matmul(normalized_pts, np.transpose(intrinsics, (0, 2, 1)))
        
        # Extract pixel coordinates
        u = pts_pixel[..., 0:1]
        v = pts_pixel[..., 1:2]
        
        # Set invalid points to zero
        u[~valid_mask] = 0
        v[~valid_mask] = 0
        depths[~valid_mask] = 0
        
        # Return points in uvz format
        result = np.concatenate([u, v, depths], axis=-1)
        
        return torch.from_numpy(result)
    
    def w2s_moge(self, pts, poses):
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        assert poses.shape[0] == self.frame_num
        poses = poses.to(torch.float32).to(self.device)
        T, N, _ = pts.shape  # (T, N, 3)
        intr = self.intr.unsqueeze(0).repeat(self.frame_num, 1, 1)
        ones = torch.ones((T, N, 1), device=self.device, dtype=pts.dtype)
        points_world_h = torch.cat([pts, ones], dim=-1)
        points_camera_h = torch.bmm(poses, points_world_h.permute(0, 2, 1))
        points_camera = points_camera_h[:, :3, :].permute(0, 2, 1)

        points_image_h = torch.bmm(points_camera, intr.permute(0, 2, 1))

        uv = points_image_h[:, :, :2] / points_image_h[:, :, 2:3]
        depth = points_camera[:, :, 2:3]  # (T, N, 1)
        uvd = torch.cat([uv, depth], dim=-1)  # (T, N, 3)

        return uvd
    
    def set_intr(self, K):
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.intr = K.to(self.device)

    def set_extr(self, extr):
        if isinstance(extr, np.ndarray):    
            extr = torch.from_numpy(extr)
        self.extr = extr.to(self.device)

    def rot_poses(self, angle, axis='y'):
        """Generate a single rotation matrix
        
        Args:
            angle (float): Rotation angle in degrees
            axis (str): Rotation axis ('x', 'y', or 'z')
            
        Returns:
            torch.Tensor: Single rotation matrix [4, 4]
        """
        angle_rad = math.radians(angle)
        cos_theta = torch.cos(torch.tensor(angle_rad))
        sin_theta = torch.sin(torch.tensor(angle_rad))
        
        if axis == 'x':
            rot_mat = torch.tensor([
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        elif axis == 'y':
            rot_mat = torch.tensor([
                [cos_theta, 0, sin_theta, 0],
                [0, 1, 0, 0],
                [-sin_theta, 0, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        elif axis == 'z':
            rot_mat = torch.tensor([
                [cos_theta, -sin_theta, 0, 0],
                [sin_theta, cos_theta, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        else:
            raise ValueError("Invalid axis value. Choose 'x', 'y', or 'z'.")
            
        return rot_mat.to(self.device)

    def trans_poses(self, dx, dy, dz):
        """
        params:
        - dx: float, displacement along x axis。
        - dy: float, displacement along y axis。
        - dz: float, displacement along z axis。

        ret:
        - matrices: torch.Tensor
        """
        trans_mats = torch.eye(4).unsqueeze(0).repeat(self.frame_num, 1, 1)  # (n, 4, 4)

        delta_x = dx / (self.frame_num - 1)
        delta_y = dy / (self.frame_num - 1)
        delta_z = dz / (self.frame_num - 1)

        for i in range(self.frame_num):
            trans_mats[i, 0, 3] = i * delta_x
            trans_mats[i, 1, 3] = i * delta_y
            trans_mats[i, 2, 3] = i * delta_z

        return trans_mats.to(self.device)
    

    def _look_at(self, camera_position, target_position):
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        # calculate rotation matrix
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction])
        rotation_matrix = np.linalg.inv(rotation_matrix)
        return rotation_matrix

    def spiral_poses(self, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
        """Generate spiral camera poses
        
        Args:
            radius (float): Base radius of the spiral
            forward_ratio (float): Scale factor for forward motion
            backward_ratio (float): Scale factor for backward motion
            rotation_times (float): Number of rotations to complete
            look_at_times (float): Scale factor for look-at point distance
            
        Returns:
            torch.Tensor: Camera poses of shape [num_frames, 4, 4]
        """
        # Generate spiral trajectory
        t = np.linspace(0, 1, self.frame_num)
        r = np.sin(np.pi * t) * radius * rotation_times
        theta = 2 * np.pi * t
        
        # Calculate camera positions
        # Limit y motion for better floor/sky view
        y = r * np.cos(theta) * 0.3  
        x = r * np.sin(theta)
        z = -r
        z[z < 0] *= forward_ratio
        z[z > 0] *= backward_ratio
        
        # Set look-at target
        target_pos = np.array([0, 0, radius * look_at_times])
        cam_pos = np.vstack([x, y, z]).T
        cam_poses = []
        
        for pos in cam_pos:
            rot_mat = self._look_at(pos, target_pos)
            trans_mat = np.eye(4)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = pos
            cam_poses.append(trans_mat[None])
            
        camera_poses = np.concatenate(cam_poses, axis=0)
        return torch.from_numpy(camera_poses).to(self.device)

    def get_default_motion(self):
        """Parse motion parameters and generate corresponding motion matrices
        
        Supported formats:
        - trans <dx> <dy> <dz> [start_frame] [end_frame]: Translation motion
        - rot <axis> <angle> [start_frame] [end_frame]: Rotation motion
        - spiral <radius> [start_frame] [end_frame]: Spiral motion
        
        Multiple transformations can be combined using semicolon (;) as separator:
        e.g., "trans 0 0 0.5 0 30; rot x 25 0 30; trans 0.1 0 0 30 48"
        
        Note:
            - start_frame and end_frame are optional
            - frame range: 0-49 (will be clamped to this range)
            - if not specified, defaults to 0-49
            - frames after end_frame will maintain the final transformation
            - for combined transformations, they are applied in sequence
            - moving left, up and zoom out is positive in video
        
        Returns:
            torch.Tensor: Motion matrices [num_frames, 4, 4]
        """
        if not isinstance(self.motion_type, str):
            raise ValueError(f'camera_motion must be a string, but got {type(self.motion_type)}')
        
        # Split combined transformations
        transform_sequences = [s.strip() for s in self.motion_type.split(';')]
        
        # Initialize the final motion matrices
        final_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
        
        # Process each transformation in sequence
        for transform in transform_sequences:
            params = transform.lower().split()
            if not params:
                continue
                
            motion_type = params[0]
            
            # Default frame range
            start_frame = 0
            end_frame = 48  # 49 frames in total (0-48)
            
            if motion_type == 'trans':
                # Parse translation parameters
                if len(params) not in [4, 6]:
                    raise ValueError(f"trans motion requires 3 or 5 parameters: 'trans <dx> <dy> <dz>' or 'trans <dx> <dy> <dz> <start_frame> <end_frame>', got: {transform}")
                
                dx, dy, dz = map(float, params[1:4])
                
                if len(params) == 6:
                    start_frame = max(0, min(48, int(params[4])))
                    end_frame = max(0, min(48, int(params[5])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                # Generate current transformation
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
                for frame_idx in range(49):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        current_motion[frame_idx, :3, 3] = torch.tensor([dx, dy, dz], device=self.device) * t
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'rot':
                # Parse rotation parameters
                if len(params) not in [3, 5]:
                    raise ValueError(f"rot motion requires 2 or 4 parameters: 'rot <axis> <angle>' or 'rot <axis> <angle> <start_frame> <end_frame>', got: {transform}")
                
                axis = params[1]
                if axis not in ['x', 'y', 'z']:
                    raise ValueError(f"Invalid rotation axis '{axis}', must be 'x', 'y' or 'z'")
                angle = float(params[2])
                
                if len(params) == 5:
                    start_frame = max(0, min(48, int(params[3])))
                    end_frame = max(0, min(48, int(params[4])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
                for frame_idx in range(49):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        current_angle = angle * t
                        current_motion[frame_idx] = self.rot_poses(current_angle, axis)
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'spiral':
                # Parse spiral motion parameters
                if len(params) not in [2, 4]:
                    raise ValueError(f"spiral motion requires 1 or 3 parameters: 'spiral <radius>' or 'spiral <radius> <start_frame> <end_frame>', got: {transform}")
                
                radius = float(params[1])
                
                if len(params) == 4:
                    start_frame = max(0, min(48, int(params[2])))
                    end_frame = max(0, min(48, int(params[3])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
                spiral_motion = self.spiral_poses(radius)
                for frame_idx in range(49):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        idx = int(t * (len(spiral_motion) - 1))
                        current_motion[frame_idx] = spiral_motion[idx]
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            else:
                raise ValueError(f'camera_motion type must be in [trans, spiral, rot], but got {motion_type}')
        
        return final_motion

class ObjectMotionGenerator:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.num_frames = 49
        
    def _get_points_in_mask(self, pred_tracks, mask):
        """Get points that lie within the mask
        
        Args:
            pred_tracks (torch.Tensor): Point trajectories [num_frames, num_points, 3] 
            mask (torch.Tensor): Binary mask [H, W]
            
        Returns:
            torch.Tensor: Boolean mask for selected points [num_points]
        """
        first_frame_points = pred_tracks[0]  # [num_points, 3]
        xy_points = first_frame_points[:, :2]  # [num_points, 2]
        
        xy_pixels = xy_points.round().long()
        xy_pixels[:, 0].clamp_(0, mask.shape[1] - 1)
        xy_pixels[:, 1].clamp_(0, mask.shape[0] - 1)
        
        points_in_mask = mask[xy_pixels[:, 1], xy_pixels[:, 0]]
        
        return points_in_mask

    def apply_motion(self, pred_tracks, mask, motion_type, distance, num_frames=49, tracking_method="spatracker"):

        self.num_frames = num_frames
        pred_tracks = pred_tracks.to(self.device).float()
        mask = mask.to(self.device)

        template = {
            'up': ('trans', torch.tensor([0, -1, 0])),
            'down': ('trans', torch.tensor([0, 1, 0])), 
            'left': ('trans', torch.tensor([-1, 0, 0])),
            'right': ('trans', torch.tensor([1, 0, 0])),
            'front': ('trans', torch.tensor([0, 0, 1])),
            'back': ('trans', torch.tensor([0, 0, -1])),
            'rot': ('rot', None) # rotate around y axis
        }
        
        if motion_type not in template:
            raise ValueError(f"unknown motion type: {motion_type}")
            
        motion_type, base_vec = template[motion_type]
        if base_vec is not None:
            base_vec = base_vec.to(self.device) * distance

        if tracking_method == "moge":
            T, H, W, _ = pred_tracks.shape
            valid_selected = ~torch.any(torch.isnan(pred_tracks[0]), dim=2) & mask
            points = pred_tracks[0][valid_selected].reshape(-1, 3)
        else:
            points_in_mask = self._get_points_in_mask(pred_tracks, mask)
            points = pred_tracks[0, points_in_mask]
            
        center = points.mean(dim=0)
        
        motions = []
        for frame_idx in range(num_frames):
            t = frame_idx / (num_frames - 1)
            current_motion = torch.eye(4, device=self.device)
            current_motion[:3, 3] = -center
            motion_mat = torch.eye(4, device=self.device)
            if motion_type == 'trans':
                motion_mat[:3, 3] = base_vec * t
            else:  # 'rot'
                angle_rad = torch.deg2rad(torch.tensor(distance * t, device=self.device))
                cos_t = torch.cos(angle_rad)
                sin_t = torch.sin(angle_rad)
                motion_mat[0, 0] = cos_t
                motion_mat[0, 2] = sin_t
                motion_mat[2, 0] = -sin_t
                motion_mat[2, 2] = cos_t
            
            current_motion = motion_mat @ current_motion
            current_motion[:3, 3] += center
            motions.append(current_motion)
            
        motions = torch.stack(motions)  # [num_frames, 4, 4]

        if tracking_method == "moge":
            modified_tracks = pred_tracks.clone().reshape(T, -1, 3)
            valid_selected = valid_selected.reshape([-1])

            for frame_idx in range(self.num_frames):
                motion_mat = motions[frame_idx]
                if W > 1: 
                    motion_mat = motion_mat.clone()
                    motion_mat[0, 3] /= W
                    motion_mat[1, 3] /= H
                points = modified_tracks[frame_idx, valid_selected]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, valid_selected] = transformed_points[:, :3]
            
            return modified_tracks.reshape(T, H, W, 3)
            
        else:
            points_in_mask = self._get_points_in_mask(pred_tracks, mask)
            modified_tracks = pred_tracks.clone()
            
            for frame_idx in range(pred_tracks.shape[0]):
                motion_mat = motions[frame_idx]
                points = modified_tracks[frame_idx, points_in_mask]
                points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
                transformed_points = torch.matmul(points_homo, motion_mat.T)
                modified_tracks[frame_idx, points_in_mask] = transformed_points[:, :3]
            
            return modified_tracks