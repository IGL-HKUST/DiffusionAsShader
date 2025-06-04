import bpy
import os
import random
import bpy_extras
from mathutils import Vector, Matrix
import numpy as np
import sys

argv = sys.argv

try:
    index = argv.index('--base_path')
    if index + 1 < len(argv):
        base_path = argv[index + 1]
    else:
        raise ValueError
except ValueError:
    default_path = r"~/blender"
    print(f"base_path parameter not provided, using default path: {default_path}")
    base_path = default_path

if not os.path.exists(base_path):
    print(f"Error: Specified path does not exist - {base_path}")
    if base_path != default_path:
        print(f"Attempting to use default path: {default_path}")
        if os.path.exists(default_path):
            base_path = default_path
        else:
            print(f"Error: Default path also does not exist")
            sys.exit(1)
    else:
        print(f"Error: Default path does not exist")
        sys.exit(1)

raw_path = os.path.join(base_path, "raw")
if not os.path.exists(raw_path):
    print(f"Error: raw folder does not exist - {raw_path}")
    sys.exit(1)

original_materials = set(bpy.data.materials.keys())
original_images = set(bpy.data.images.keys())

initial_camera_matrix = None
cam = bpy.context.scene.camera
if cam:
    initial_camera_matrix = cam.matrix_world.copy()
else:
    print("Warning: No camera found in the scene")

skybox_path = os.path.join(base_path, "raw/skybox")
if not os.path.exists(skybox_path):
    print(f"Warning: skybox folder does not exist - {skybox_path}")

initial_env_positions = {}

def clean_scene(original_materials, original_images):
    env_collection = bpy.context.scene.collection.children.get('env')
    env_objects = set()
    if env_collection:
        # Restore the positions of env objects
        for obj in env_collection.objects:
            if obj.type == 'EMPTY':
                if obj.name in initial_env_positions:
                    obj.location = initial_env_positions[obj.name].copy()
        env_objects = {obj.name for obj in env_collection.objects}
    
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.name not in env_objects:
            try:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.data.objects.remove(obj, do_unlink=True)
            except ReferenceError:
                continue

    for action in bpy.data.actions:
        try:
            bpy.data.actions.remove(action, do_unlink=True)
        except ReferenceError:
            continue

    current_materials = set(bpy.data.materials.keys())
    imported_materials = current_materials - original_materials
    for material_name in imported_materials:
        try:
            material = bpy.data.materials.get(material_name)
            if material:
                bpy.data.materials.remove(material, do_unlink=True)
        except ReferenceError:
            continue

    current_images = set(bpy.data.images.keys())
    imported_images = current_images - original_images
    for image_name in imported_images:
        try:
            image = bpy.data.images.get(image_name)
            if image:
                bpy.data.images.remove(image, do_unlink=True)
        except ReferenceError:
            continue
            
    import gc
    gc.collect()

def setup_random_background():
    world = bpy.data.worlds['World']
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    nodes.clear()
    
    background = nodes.new('ShaderNodeBackground')
    output = nodes.new('ShaderNodeOutputWorld')
    links.new(background.outputs[0], output.inputs[0])
    
    if random.choice([True, False]):
        color = (random.random(), random.random(), random.random(), 1)
        background.inputs[0].default_value = color
        return "blank"
    else:
        skybox_files = [f for f in os.listdir(skybox_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if skybox_files:
            tex_coord = nodes.new('ShaderNodeTexCoord')
            mapping = nodes.new('ShaderNodeMapping')
            texture = nodes.new('ShaderNodeTexEnvironment')
            
            random_rotation = random.uniform(0, 2 * 3.14159)
            mapping.inputs['Rotation'].default_value[2] = random_rotation
            
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], texture.inputs['Vector'])
            
            random_image = random.choice(skybox_files)
            img_path = os.path.join(skybox_path, random_image)
            texture.image = bpy.data.images.load(img_path)
            
            links.new(texture.outputs[0], background.inputs[0])
            
            return os.path.splitext(random_image)[0]
        return "blank"

def setup_render_settings(is_animation=True):
    scene = bpy.context.scene
    
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    prefs.addons['cycles'].preferences.compute_device_type = 'CUDA'  # Or 'OPTIX' for RTX cards
    
    for device in prefs.addons['cycles'].preferences.devices:
        device.use = True
    
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    
    scene.eevee.taa_render_samples = 16
    
    scene.render.resolution_x = 720
    scene.render.resolution_y = 480
    scene.render.fps = 24
    
    if is_animation:
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
    else:
        scene.render.image_settings.file_format = 'PNG'

def import_fbx(file_path, **options):
    selected_before = set(obj.name for obj in bpy.context.selected_objects)
    active_before = bpy.context.active_object
    
    bpy.ops.import_scene.fbx(filepath=file_path, **options)
    
    new_selected = set(obj.name for obj in bpy.context.selected_objects) - selected_before
    armature = None
    
    for obj_name in new_selected:
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.type == 'ARMATURE':
            armature = obj
            for collection in obj.users_collection:
                collection.objects.unlink(obj)
            avatar_collection = bpy.context.scene.collection.children.get('avatar')
            if avatar_collection:
                avatar_collection.objects.link(obj)
            break
            
    bpy.ops.object.select_all(action='DESELECT')
    for obj_name in selected_before:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            obj.select_set(True)
    if active_before:
        bpy.context.view_layer.objects.active = active_before
        
    return armature

def get_scene_inverse_depth_range(scene, cam):
    """Calculate the inverse depth values of all vertices in the scene and return the 2% and 98% percentiles."""
    inverse_depths = []
    
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
            
        world_matrix = obj.matrix_world
        mesh = obj.data
        
        for vertex in mesh.vertices:
            world_coord = world_matrix @ vertex.co
            distance = (world_coord - cam.matrix_world.translation).length
            inverse_depth = 1.0 / distance if distance > 0 else 0
            inverse_depths.append(inverse_depth)
    
    if not inverse_depths:
        return 0, 1  # Default values
        
    inverse_depths = np.array(inverse_depths)
    min_inverse = np.percentile(inverse_depths, 2)
    max_inverse = np.percentile(inverse_depths, 98)
    
    return min_inverse, max_inverse

def set_vertex_colors_for_object(obj, scene, cam, inverse_depth_range):
    """Set vertex colors and material for a single object."""
    if obj.type != 'MESH':
        return
    
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    
    mesh = obj.data
    if obj.data.shape_keys or obj.modifiers:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_eval = object_eval.data
    else:
        mesh_eval = mesh
    
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="Col")
    
    color_layer = mesh.vertex_colors.active
    
    world_matrix = obj.matrix_world
    
    cam_location = cam.matrix_world.translation
    cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    
    min_inverse, max_inverse = inverse_depth_range
    inverse_range = max_inverse - min_inverse
    
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            
            world_coord = world_matrix @ mesh_eval.vertices[vertex_index].co
            
            to_vertex = (world_coord - cam_location).normalized()
            
            is_backface = to_vertex.dot(cam_direction) < 0
            
            screen_coord = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_coord)
            
            distance = (world_coord - cam_location).length
            inverse_depth = 1.0 / distance if distance > 0 else 0
            
            # Normalize based on the percentile range
            b = (inverse_depth - min_inverse) / inverse_range if inverse_range > 0 else 0
            b = max(0, min(1, b))  # Ensure the value is within the 0-1 range
            
            r = screen_coord.x
            g = 1 - screen_coord.y
            
            # If coordinates are outside screen space or on a backface, set to black
            if r < 0 or r > 1 or g < 0 or g > 1 or is_backface:
                r, g, b = 0, 0, 0
            
            color_layer.data[loop_index].color = (r, g, b, 1.0)
    
    mat_name = f"ScreenSpaceColor_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    nodes.clear()
    
    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    vertex_color.layer_name = "Col"
    
    emission = nodes.new(type='ShaderNodeEmission')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    links = mat.node_tree.links
    links.new(vertex_color.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def process_model(model_name):
    print(f"Starting to process model: {model_name}")
    
    output_path = os.path.join(base_path, "trackingavatars")
    animation_folder = os.path.join(raw_path, model_name, "motion")
    model_file_path = os.path.join(raw_path, model_name, f"{model_name}.fbx")
    
    if not os.path.exists(animation_folder):
        print(f"Warning: Animation folder does not exist - {animation_folder}")
        return
    
    if not os.path.exists(model_file_path):
        print(f"Warning: Model file does not exist - {model_file_path}")
        return
        
    videos_path = os.path.join(output_path, "videos")
    tracking_path = os.path.join(output_path, "tracking")
    images_path = os.path.join(output_path, "images")
    
    for path in [videos_path, tracking_path, images_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        
    prompt_file = os.path.join(output_path, "prompt.txt")
    videos_txt = os.path.join(output_path, "videos.txt")
    trackings_txt = os.path.join(output_path, "trackings.txt")
    images_txt = os.path.join(output_path, "images.txt")
    
    for txt_file in [prompt_file, videos_txt, trackings_txt, images_txt]:
        if not os.path.exists(txt_file):
            open(txt_file, "a").close()  # Create file but do not clear existing content
            
    def get_max_file_number():
        max_num = 0
        if os.path.exists(videos_path):
            video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
            if video_files:
                max_num = max(int(os.path.splitext(f)[0]) for f in video_files)
        return max_num
    
    start_number = get_max_file_number() + 1
    
    animation_files = [f for f in os.listdir(animation_folder) if f.endswith('.fbx')]
    animation_files.sort()
    
    # Record the initial positions of env objects on the first run
    env_collection = bpy.context.scene.collection.children.get('env')
    if env_collection:
        for obj in env_collection.objects:
            if obj.type == 'EMPTY' and obj.name not in initial_env_positions:
                initial_env_positions[obj.name] = obj.location.copy()
    
    for i, animation_file in enumerate(animation_files, start_number):
        print(f"Processing file {i}: {animation_file}")
        animation_name = os.path.splitext(animation_file)[0]
        
        try:
            clean_scene(original_materials, original_images)
            
            # Randomly move the camera and env objects
            cam = bpy.context.scene.camera
            if cam and initial_camera_matrix is not None:
                cam.matrix_world = initial_camera_matrix.copy()
                random_offset = random.uniform(0.5, 2.5)
                cam.location.y += random_offset
                print(f"Camera Y-axis offset: {random_offset} meters")
                
                # Get objects in the env collection and move them
                env_collection = bpy.context.scene.collection.children.get('env')
                if env_collection:
                    # Calculate env object offset (0.1 times camera offset)
                    env_offset = random_offset * 0.1
                    for obj in env_collection.objects:
                        if obj.type == 'EMPTY':
                            obj.location.z += env_offset
                    print(f"env empty object z-axis offset: {env_offset} meters")
            
            avatar_collection = bpy.context.scene.collection.children.get('avatar')
            if not avatar_collection:
                avatar_collection = bpy.data.collections.new('avatar')
                bpy.context.scene.collection.children.link(avatar_collection)
            
            # Import animation and model
            animation_armature = import_fbx(
                os.path.join(animation_folder, animation_file),
                use_manual_orientation=True,
                axis_forward='-Z',
                axis_up='Y',
                use_anim=True,
                bake_space_transform=True
            )
            
            model_armature = import_fbx(
                model_file_path,
                use_manual_orientation=True,
                axis_forward='-Z',
                axis_up='Y'
            )
            
            if not animation_armature or not model_armature:
                raise Exception("Import failed")
            
            # Disable physics simulation
            if bpy.context.scene.rigidbody_world:
                bpy.ops.rigidbody.world_remove()
            
            for obj in bpy.data.objects:
                if obj.rigid_body:
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.rigidbody.object_remove()
                if obj.rigid_body_constraint:
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.rigidbody.constraint_remove()
            
            # Update armature modifiers
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    for modifier in obj.modifiers:
                        if modifier.type == 'ARMATURE' and modifier.object == model_armature:
                            modifier.object = animation_armature
            
            # Hide model armature
            model_armature.hide_viewport = True
            model_armature.hide_render = True
            
            # Set random background and render settings
            background_type = setup_random_background()
            setup_render_settings()
            
            # Optimize scene settings
            scene = bpy.context.scene
            
            # Generate new prompt
            if background_type == "blank":
                new_prompt = f"cartoon character {model_name} {animation_name} in a blank room."
            else:
                background_name = background_type.replace('_', ' ')
                new_prompt = f"cartoon character {model_name} {animation_name} in {background_name}."
            
            # Read existing prompts
            with open(prompt_file, "r") as f:
                prompts = f.readlines()
            
            # Ensure prompts list is long enough
            while len(prompts) < i:
                prompts.append("\n")
            
            # Replace the prompt for the corresponding line
            if i <= len(prompts):
                prompts[i-1] = new_prompt + "\n"
            else:
                prompts.append(new_prompt + "\n")
            
            # Write back to file
            with open(prompt_file, "w") as f:
                f.writelines(prompts)
            
            # Render animation
            output_file = f"{i}.mp4"
            bpy.context.scene.render.filepath = os.path.join(videos_path, output_file)
            bpy.ops.render.render(animation=True)
            
            # Update videos.txt
            with open(videos_txt, "a") as f:
                f.write(f"videos/{output_file}\n")
            
            # Render first frame (image)
            setup_render_settings(is_animation=False)
            image_file = f"{i}.png"
            bpy.context.scene.frame_set(bpy.context.scene.frame_start)
            bpy.context.scene.render.filepath = os.path.join(images_path, image_file)
            bpy.ops.render.render(write_still=True)
            
            # Update images.txt
            with open(images_txt, "a") as f:
                f.write(f"images/{image_file}\n")
            
            # Render tracking video
            scene = bpy.context.scene
            cam = scene.camera
            
            if cam:
                # Get objects in the env collection
                env_collection = bpy.context.scene.collection.children.get('env')
                env_objects = set()
                if env_collection:
                    env_objects = {obj.name for obj in env_collection.objects}
                
                # Find all meshes to process (excluding objects in the env collection)
                character_meshes = []
                for obj in bpy.data.objects:
                    if obj.type == 'MESH' and obj.name not in env_objects:
                        character_meshes.append(obj)
                
                # Find plane
                plane_mesh = None
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH' and obj.name == 'plane':
                        plane_mesh = obj
                        break
                
                if character_meshes:
                    inverse_depth_range = get_scene_inverse_depth_range(scene, cam)
                    
                    # Save original materials
                    original_materials_dict = {}
                    for mesh_obj in character_meshes:
                        if mesh_obj.data.materials:
                            original_materials_dict[mesh_obj] = mesh_obj.data.materials[0]
                    
                    # Save plane's original material
                    plane_original_mat = None
                    if plane_mesh and plane_mesh.data.materials:
                        plane_original_mat = plane_mesh.data.materials[0]
                    
                    # Set tracking material
                    for mesh_obj in character_meshes:
                        set_vertex_colors_for_object(mesh_obj, scene, cam, inverse_depth_range)
                    
                    # Set plane's tracking material
                    if plane_mesh:
                        if not bpy.data.materials.get(f"ScreenSpaceColor_{plane_mesh.name}"):
                            set_vertex_colors_for_object(plane_mesh, scene, cam, inverse_depth_range)
                        else:
                            plane_mesh.data.materials[0] = bpy.data.materials.get(f"ScreenSpaceColor_{plane_mesh.name}")
                    
                    # Set black background and render tracking video
                    world = bpy.data.worlds['World']
                    world.use_nodes = True
                    nodes = world.node_tree.nodes
                    nodes.clear()
                    background = nodes.new('ShaderNodeBackground')
                    output = nodes.new('ShaderNodeOutputWorld')
                    world.node_tree.links.new(background.outputs[0], output.inputs[0])
                    background.inputs[0].default_value = (0, 0, 0, 1)
                    
                    # Set back to animation rendering mode
                    setup_render_settings(is_animation=True)
                    
                    tracking_file = f"{i}_tracking.mp4"
                    bpy.context.scene.render.filepath = os.path.join(tracking_path, tracking_file)
                    bpy.ops.render.render(animation=True)
                    
                    # Update trackings.txt
                    with open(trackings_txt, "a") as f:
                        f.write(f"tracking/{tracking_file}\n")
                    
                    # Restore original materials
                    for mesh_obj, original_mat in original_materials_dict.items():
                        mesh_obj.data.materials[0] = original_mat
                    
                    # Restore plane's original material
                    if plane_mesh and plane_original_mat:
                        plane_mesh.data.materials[0] = plane_original_mat
                    
                    # Clean tracking materials
                    for mesh_obj in [*character_meshes, plane_mesh] if plane_mesh else character_meshes:
                        mat_name = f"ScreenSpaceColor_{mesh_obj.name}"
                        tracking_material = bpy.data.materials.get(mat_name)
                        if tracking_material:
                            bpy.data.materials.remove(tracking_material, do_unlink=True)
                else:
                    print("Warning: Character meshes not found, cannot generate tracking video")
            else:
                print("Warning: No camera in scene, cannot generate tracking video")
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            continue
            
        finally:
            clean_scene(original_materials, original_images)

def main():
    model_folders = []
    for item in os.listdir(raw_path):
        if os.path.isdir(os.path.join(raw_path, item)) and item != "skybox":
            model_folders.append(item)
    
    if not model_folders:
        print("Error: No model folders found")
        return
    
    print(f"Found the following model folders: {', '.join(model_folders)}")
    
    for model_name in model_folders:
        try:
            process_model(model_name)
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue

main() 