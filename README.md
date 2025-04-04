# Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2501.03847'><img src='https://img.shields.io/badge/arXiv-2501.03847-b31b1b.svg'></a> &nbsp;
 <a href='https://igl-hkust.github.io/das/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/EXCAI/Diffusion-As-Shader)&nbsp;
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/EXCAI/Diffusion-As-Shader)

![teaser](assets/teaser.gif)

## Quickstart

### Create environment
1. Clone the repository and create conda environment: 

    ```
    git clone git@github.com:IGL-HKUST/DiffusionAsShader.git
    conda create -n das python=3.10
    conda activate das
    ```

2. Install pytorch, we recommend `Pytorch 2.5.1` with `CUDA 11.8`:

    ```
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```


<!-- 3. Install `MoGe`:
```
pip install git+https://github.com/asomoza/image_gen_aux.git
``` -->

3. Make sure the submodule and requirements are installed:
    ```
    mkdir -p submodules
    git submodule update --init --recursive
    pip install -r requirements.txt
    ```
    If the submodules are not installed, you need to manually download them and move them to `submodules/`. Run the following commands to install the submodules:
    ```
    # MoGe
    git clone https://github.com/microsoft/MoGe.git submodules/MoGe
    # VGGT
    git clone https://github.com/facebookresearch/vggt.git submodules/vggt
    ```

4. Manually download these checkpoints:
   - SpatialTracker checkpoint: [Google Drive](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ) and move it to `checkpoints/`.
   - Our *Diffusion as Shader* checkpoint: https://huggingface.co/EXCAI/Diffusion-As-Shader

<!-- 5. Manually download the ZoeDepth checkpoints (dpt_beit_large_384.pt, ZoeD_M12_K.pt, ZoeD_M12_NK.pt) to `models/monoD/zoeDepth/ckpts/`. For more information, refer to [this issue](https://github.com/henry123-boy/SpaTracker/issues/20). -->

<!-- Then download a dataset:

```bash
# install `huggingface_hub`
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney
``` -->

### Inference

The inference code was tested on

- Ubuntu 20.04
- Python 3.10
- PyTorch 2.5.1
- 1 NVIDIA H800 with CUDA version 11.8. (32GB GPU memory is sufficient for generating videos with our code.)

We provide a inference script for our tasks. You can run the `demo.py` script directly as follows.
**We also provide a validation dataset in [Google Drive](https://drive.google.com/file/d/1pVB_2AEoz1v4vXWe6-pdDAEQdmlGEIci/view?usp=sharing) for our 4 tasks. You can run the `scripts/evaluate_DaS.sh` to evaluate the performance of our model.**

We release the gradio interface for our tasks. You can run the `webui.py` script directly as follows.
```
python webui.py --gpu <gpu_id>
```

Or you can run these tasks one by one as follows.

#### 1. Motion Transfer 
```python
python demo.py \
    --prompt <"prompt text"> \ # prompt text
    --checkpoint_path <model_path> \ # checkpoint path
    --output_dir <output_dir> \ # output directory
    --input_path <input_path> \ # the reference video path
    --repaint < True/repaint_path > \ # the repaint first frame image path of input source video or use FLUX to repaint the first frame
    --gpu <gpu_id> \ # the gpu id

```

#### 2. Camera Control

<table border="1">
<tr>
  <th>Arc Right + Zoom out</th>
  <th>Arc Left + Zoom out</th>
  <th>Arc Up + Zoom out</th>
</tr>
<tr>
  <td><img src="assets/videos/panright+out.gif" alt="Pans Right + Zoom out"></td>
  <td><img src="assets/videos/panleft+out.gif" alt="Pans Left + Zoom out"></td>
  <td><img src="assets/videos/panup+out.gif" alt="Pans Up + Zoom out"></td>
</tr>
<tr>
  <th>Pans Left + Yaw Left</th>
  <th>Static</th>
  <th>Zoom out</th>
</tr>
<tr>
  <td><img src="assets/videos/car_panright.gif" alt="Pans Right"></td>
  <td><img src="assets/videos/car_static.gif" alt="Static"></td>
  <td><img src="assets/videos/car_zoomout.gif" alt="Zoom out"></td>
</tr>
</table>

We provide several template camera motion types, you can choose one of them. In practice, we find that providing a description of the camera motion in prompt will get better results.
```python
python demo.py \
    --prompt <"prompt text"> \ # prompt text
    --checkpoint_path <model_path> \ # checkpoint path
    --output_dir <output_dir> \ # output directory
    --input_path <input_path> \ # the reference image or video path
    --camera_motion <camera_motion> \ # the camera motion type, see examples below
    --tracking_method <tracking_method> \ # the tracking method (moge, spatracker, cotracker). For image input, 'moge' is necessary.
    --override_extrinsics <override/append> \ # how to apply camera motion: "override" to replace original camera, "append" to build upon it
    --gpu <gpu_id> \ # the gpu id
```

Here are some tips for camera motion:
- trans: translation motion, the camera will move in the direction of the vector (dx, dy, dz) with range [-1, 1]
  - Positive X: Move left, Negative X: Move right
  - Positive Y: Move down, Negative Y: Move up
  - Positive Z: Zoom in, Negative Z: Zoom out
  - e.g., 'trans -0.1 -0.1 -0.1' moving right, down and zoom in
  - e.g., 'trans -0.1 0.0 0.0 5 45' moving right 0.1 from frame 5 to 45
- rot: rotation motion, the camera will rotate around the axis (x, y, z) by the angle
  - X-axis rotation: positive X: pitch down, negative X: pitch up
  - Y-axis rotation: positive Y: yaw left, negative Y: yaw right
  - Z-axis rotation: positive Z: roll counter-clockwise, negative Z: roll clockwise
  - e.g., 'rot y 25' rotating 25 degrees around y-axis (yaw left)
  - e.g., 'rot x -30 10 40' rotating -30 degrees around x-axis (pitch up) from frame 10 to 40
- spiral: spiral motion, the camera will move in a spiral path with the given radius
  - e.g., 'spiral 2' spiral motion with radius 2
  - e.g., 'spiral 2 15 35' spiral motion with radius 2 from frame 15 to 35

Multiple transformations can be combined using semicolon (;) as separator:
- e.g., "trans 0 0 -0.5 0 30; rot x -25 0 30; trans -0.1 0 0 30 48"
  This will:
  1. Zoom in (z-0.5) from frame 0 to 30
  2. Pitch up (rotate -25 degrees around x-axis) from frame 0 to 30
  3. Move right (x-0.1) from frame 30 to 48

Notes:
- Frame range is 0-48 (49 frames in total)
- If start_frame and end_frame are not specified, the motion will be applied to all frames (0-48)
- Frames after end_frame will maintain the final transformation
- For combined transformations, they are applied in sequence

#### 3. Object Manipulation

<table border="1">
<tr>
  <th>Move Left</th>
  <th>Move Up</th>
  <th>Rotate</th>
</tr>
<tr>
  <td><img src="assets/videos/move left.gif" alt="Move Left"></td>
  <td><img src="assets/videos/move up.gif" alt="Move Right"></td>
  <td><img src="assets/videos/rotate.gif" alt="Rotate"></td>
</tr>
</table>

We provide several template object manipulation types, you can choose one of them. In practice, we find that providing a description of the object motion in prompt will get better results.
```python
python demo.py \
    --prompt <"prompt text"> \ # prompt text
    --checkpoint_path <model_path> \ # checkpoint path
    --output_dir <output_dir> \ # output directory
    --input_path <input_path> \ # the reference image path
    --object_motion <object_motion> \ # the object motion type (up, down, left, right)
    --object_mask <object_mask_path> \ # the object mask path
    --tracking_method <tracking_method> \ # the tracking method (moge, spatracker). For image input, 'moge' is nesserary.
    --gpu <gpu_id> \ # the gpu id

```
Or you can create your own object motion and camera motion as follows and replace related codes in `demo.py`:

1. object motion
    ```
    dict: Motion dictionary containing:
      - mask (torch.Tensor): Binary mask for selected object
      - motions (torch.Tensor): Per-frame motion vectors [49, 4, 4] (49 frames, 4x4 homogenous objects motion matrix)
    ``` 
2. camera motion
    ```
    list: CameraMotion list containing:
      - camera_motion (list): Per-frame camera poses matrix [49, 4, 4] (49 frames, 4x4 homogenous camera poses matrix)
    ``` 
It should be noted that depending on the tracker you choose, you may need to modify the scale of translation.

#### 4. Animating meshes to video

<table border="1">
<tr>
  <td><img src="assets/videos/m2v1.gif" alt="video 1"></td>
  <td><img src="assets/videos/m2v2.gif" alt="video 2"></td>
</tr>
</table>

We only support using Blender (version > 4.0) to generate the tracking video now. Before running the following command, you need to install Blender and run the script `scripts/blender.py` in your blender project and generate the tracking video for your blender project. Then you need to provide the tracking video path to the `tracking_path` argument:

```python
python demo.py \
    --prompt <"prompt text"> \ # prompt text
    --checkpoint_path <model_path> \ # checkpoint path
    --output_dir <output_dir> \ # output directory
    --input_path <input_path> \ # the reference video path
    --tracking_path <tracking_path> \ # the tracking video path (need to be generated by Blender)
    --repaint < True/repaint_path > \ # the rendered first frame image path of input mesh video or use FLUX to repaint the first frame
    --gpu <gpu_id> \ # the gpu id

```

## Finetune Diffusion as Shader

### Prepare Dataset

Before starting the training, please check whether the dataset has been prepared according to the [dataset specifications](assets/dataset.md). 

In short, your dataset structure should look like this. Running the `tree` command, you should see:

```
dataset
├── prompt.txt
├── videos.txt
├── trackings.txt

├── tracking
    ├── tracking/00000_tracking.mp4
    ├── tracking/00001_tracking.mp4
    ├── ...

├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...

```

### Training

Training can be started using the `scripts/train_image_to_video_sft.sh` scripts.

- Configure environment variables as per your choice:

  ```bash
  export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export WANDB_MODE="offline"
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ENABLE_MONITORING=0
  ```

- Configure which GPUs to use for training: `GPU_IDS="0,1"`

- Choose hyperparameters for training. Let's try to do a sweep on learning rate and optimizer type as an example:

  ```bash
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("2000")
  ```

- Select which Accelerate configuration you would like to train with: `ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"`. We provide some default configurations in the `accelerate_configs/` directory - single GPU uncompiled/compiled, 2x GPU DDP, DeepSpeed, etc. You can create your own config files with custom settings using `accelerate config --config_file my_config.yaml`.

- Specify the absolute paths and columns/files for captions and videos.

  ```bash
  DATA_ROOT="../datasets/cogshader"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  TRACKING_COLUMN="trackings.txt"
  ```

- Launch experiments sweeping different hyperparameters:
  ```
  # training dataset parameters
  DATA_ROOT="../datasets/cogshader"
  MODEL_PATH="../ckpts/CogVideoX-5b-I2V"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  TRACKING_COLUMN="trackings.txt"

  # validation parameters
  TRACKING_MAP_PATH="../eval/3d/tracking/dance_tracking.mp4"
  VALIDATION_PROMPT="text"
  VALIDATION_IMAGES="../000000046_0.png"

  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="/aifs4su/mmcode/lipeng/cogvideo/ckpts/cogshader_inv-avatar-physics_steps_${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES --main_process_port $PORT training/cogvideox_image_to_video_sft.py \
            --pretrained_model_name_or_path $MODEL_PATH \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --tracking_column $TRACKING_COLUMN \
            --tracking_map_path $TRACKING_MAP_PATH \
            --num_tracking_blocks 18 \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 8 \
            --pin_memory \
            --validation_prompt $VALIDATION_PROMPT \
            --validation_images $VALIDATION_IMAGES \
            --validation_prompt_separator ::: \
            --num_validation_videos 1 \
            --validation_epochs 1 \
            --seed 42 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size $TRAIN_BATCH_SIZE \
            --max_train_steps $steps \
            --checkpointing_steps $CHECKPOINT_STEPS \
            --gradient_accumulation_steps 4 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps $WARMUP_STEPS \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --noised_image_dropout 0.05 \
            --max_grad_norm 1.0 \
            --allow_tf32 \
            --report_to wandb \
            --resume_from_checkpoint \"latest\" \
            --nccl_timeout 1800"
          
          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

  To understand what the different parameters mean, you could either take a look at the [args](./training/args.py) file or run the training script with `--help`.

## Acknowledgements

This project builds upon several excellent open source projects:

* [CogVideo](https://github.com/THUDM/CogVideo) - A large-scale video generation model developed by Tsinghua University that provides the foundational architecture for this project.

* [finetrainers](https://github.com/a-r-r-o-w/finetrainers) - Offering efficient video model training scripts that helped optimize our training pipeline.

* [SpaTracker](https://github.com/henry123-boy/SpaTracker) - Providing excellent 2D pixel to 3D space tracking capabilities that enable our motion control features.

* [MoGe](https://github.com/microsoft/MoGe) - Microsoft's monocular geometry estimation model that helps achieve more accurate 3D reconstruction.

* [vggt](https://github.com/facebookresearch/vggt) - Facebook's video generation model that provides the foundational architecture for this project.
We thank the authors and contributors of these projects for their valuable contributions to the open source community!

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{gu2025das,
    title={Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control}, 
    author={Zekai Gu and Rui Yan and Jiahao Lu and Peng Li and Zhiyang Dou and Chenyang Si and Zhen Dong and Qifeng Liu and Cheng Lin and Ziwei Liu and Wenping Wang and Yuan Liu},
    year={2025},
    journal={arXiv preprint arXiv:2501.03847}
}
```



