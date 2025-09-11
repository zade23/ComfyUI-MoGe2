import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import logging
from .moge.model import import_model_class_by_version
import cv2
import utils3d
from .moge.utils.vis import colorize_depth, colorize_normal
import torch
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from typing import Dict, Tuple
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger(__name__)

class RunMoGe2Process:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["v1","v2"], {"default": "v2"}),
                "image": ("IMAGE",),
                "max_size": ("INT", {"default": 800, "min": 100, "max": 1000, "step": 100}),
                "resolution_level": (["Low", "Medium", "High", "Ultra"], {"default": "High"}),
                "remove_edge": ("BOOLEAN", {"default": True}),
                "apply_mask": ("BOOLEAN", {"default": True}),
                "output_glb": ("BOOLEAN", {"default": True}),  # 新增的开关按钮
                "filename_prefix": ("STRING", {"default": "3D/MoGe"}),
            }
        }
        
    RETURN_TYPES = ("IMAGE", "IMAGE","STRING")
    RETURN_NAMES = ("depth", "normal", "glb_path")
    FUNCTION = "process"
    CATEGORY = "MoGe2"
    OUTPUT_NODE = True
    DESCRIPTION = "Runs the MoGe2 model on the input image. \n v1: Ruicheng/moge-vitl \n v2: Ruicheng/moge-2-vitl-normal"
    
    def process(self, model: str, image, max_size: int, resolution_level: str, remove_edge: bool, apply_mask: bool, output_glb: bool, filename_prefix: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        
        model_version = model
        
        if model_version == "v1":
            pretrained_model_name_or_path = "Ruicheng/moge-vitl"
        elif model_version == "v2":
            pretrained_model_name_or_path = "Ruicheng/moge-2-vitl-normal"
        
        model_instance = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).cuda().eval()
        
        # Convert ComfyUI tensor to numpy array if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        # Remove batch dimension if present
        if len(image.shape) == 4:
            image = image[0]
        
        # Ensure image is in the range [0, 255] and convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        larger_size = max(image.shape[:2])
        if larger_size > max_size:
            scale = max_size / larger_size
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        height, width = image.shape[:2]
        resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)
        
        # Convert image to tensor and format it properly for the model
        use_fp16 = True
        image_tensor = torch.tensor(image, dtype=torch.float32 if not use_fp16 else torch.float16, device=torch.device('cuda')).permute(2, 0, 1) / 255
        
        output = model_instance.infer(image_tensor, apply_mask=apply_mask, resolution_level=resolution_level_int, use_fp16=use_fp16)
        output = {k: v.cpu().numpy() for k, v in output.items()}
        
        points = output['points']
        depth = output['depth']
        mask = output['mask']
        normal = output.get('normal', None)
        
        # mask
        if remove_edge:
            mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=0.04)
        else:
            mask_cleaned = mask
        
        # normal visualization
        if normal is not None:
            normal_vis = colorize_normal(normal)
        else:
            normal_vis = np.zeros_like(image)
        
        # depth visualization
        depth_for_vis = depth.copy()
        
        masked_depth = depth_for_vis[mask]
        
        if masked_depth.size == 0:
            # If nothing is detected, create a black image
            depth_normalized = np.zeros_like(depth_for_vis)
        else:
            # Normalize the depth values in the masked region to the [0, 1] range
            min_val = masked_depth.min()
            max_val = masked_depth.max()
            
            # Avoid division by zero if depth is constant (e.g., a flat plane)
            if max_val > min_val:
                depth_normalized = (depth_for_vis - min_val) / (max_val - min_val)
            else:
                depth_normalized = np.ones_like(depth_for_vis) * 0.5 # Mid-gray for flat depth

        # Invert the depth map: closer objects become brighter
        depth_inverted = 1.0 - depth_normalized
        depth_inverted[~mask] = 0
        depth_gray_uint8 = (depth_inverted * 255).astype(np.uint8)

        # Convert the single-channel grayscale image to a 3-channel RGB image for ComfyUI compatibility
        depth_gray_rgb = cv2.cvtColor(depth_gray_uint8, cv2.COLOR_GRAY2RGB)

        # Convert numpy array to ComfyUI tensor
        def numpy_to_tensor(img_np):
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            img_np = img_np.astype(np.float32) / 255.0
            if len(img_np.shape) == 3:
                img_np = img_np[None, ...] # Add batch dimension
            return torch.from_numpy(img_np)

        # Convert final visualization to tensor
        depth_tensor = numpy_to_tensor(depth_gray_rgb)
        normal_vis_tensor = numpy_to_tensor(normal_vis)
        
        # mesh
        if normal is None:
            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                points,
                image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=width, height=height),
                mask=mask_cleaned,
                tri=True
            )
            vertex_normals = None
        else:
            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                points,
                image.astype(np.float32) / 255,
                utils3d.numpy.image_uv(width=width, height=height),
                normal,
                mask=mask_cleaned,
                tri=True
            )
        vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
        vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
        if vertex_normals is not None:
            vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        relative_path = "" # Initialize to empty string
        
        if output_glb:
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=vertex_normals,
                visual = trimesh.visual.texture.TextureVisuals(
                    uv=vertex_uvs,
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.fromarray(image),
                        metallicFactor=0.5,
                        roughnessFactor=1.0
                    )
                ),
                process=False
            )

            output_glb_path = Path(full_output_folder) / f'{filename}_{counter:05}_.glb'
            output_glb_path.parent.mkdir(exist_ok=True, parents=True)
            mesh.export(output_glb_path)
            relative_path = str(Path(subfolder) / f'{filename}_{counter:05}_.glb')
        else:
            relative_path = "GLB export disabled"

        return (depth_tensor, normal_vis_tensor, relative_path)

NODE_CLASS_MAPPINGS = {
    "RunMoGe2Process": RunMoGe2Process,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunMoGe2Process": "MoGe2 Process",
}
