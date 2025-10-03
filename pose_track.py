import os
import argparse
import logging
import json
import numpy as np
import trimesh
import imageio.v2 as imageio
import cv2
import glob

# Ensure these imports are in your PYTHONPATH
from estimater import *
from datareader import *

# It's assumed that dr (nvdiffrast) is available if you use FoundationPose
try:
    import nvdiffrast.torch as dr
except ImportError:
    print("Warning: nvdiffrast not found. FoundationPose may not work.")
    dr = None

class CustomDataReader:
    """
    A custom data reader to handle the specific directory structure you provided.
    """
    def __init__(self, data_dir, object_id_str):
        self.data_dir = data_dir
        self.object_id_str = object_id_str
        
        self.rgb_files = sorted(glob.glob(os.path.join(data_dir, 'images0', '*.jpg')), key=lambda f: int(os.path.basename(f).split('.')[0]))
        self.depth_files = sorted(glob.glob(os.path.join(data_dir, 'depth', '*.npy')), key=lambda f: int(os.path.basename(f).split('.')[0]))
        
        # Load camera intrinsics
        self.K = np.load(os.path.join(data_dir, 'intrinsics.npy'))
        
        logging.info(f"Found {len(self.rgb_files)} frames for object '{self.object_id_str}' in {self.data_dir}")

    def __len__(self):
        return len(self.rgb_files)

    def get_color(self, i):
        return imageio.imread(self.rgb_files[i])

    def get_depth(self, i):
        return np.load(self.depth_files[i])

    def get_mask(self, i):
        # Frame numbers in JSON are 1-based (frame_00001.json)
        mask_json_path = os.path.join(self.data_dir, 'bounding_boxes_and_masks', self.object_id_str, f'frame_{i+1:05d}.json')
        with open(mask_json_path, 'r') as f:
            data = json.load(f)

        # Load mask and ensure it is a 2D array by squeezing singleton dimensions
        mask_array = np.array(data['mask']).astype(bool)
        if mask_array.ndim > 2:
            mask_array = np.squeeze(mask_array)

        return mask_array

def main(args):
    set_logging_format()
    set_seed(0)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output will be saved to: {args.output_dir}")

    # Load shared data: intrinsics and object scales
    transformations_path = os.path.join(args.input_dir, 'masks', 'transformations.json')
    with open(transformations_path, 'r') as f:
        transformations = json.load(f)

    # Discover objects to process
    object_dirs = sorted(glob.glob(os.path.join(args.input_dir, 'bounding_boxes_and_masks', 'object_*')))
    if not object_dirs:
        logging.error("No 'object_*' directories found in 'bounding_boxes_and_masks'. Exiting.")
        return
        
    logging.info(f"Discovered objects: {[os.path.basename(d) for d in object_dirs]}")

    # Initialize the CUDA context for the renderer
    glctx = dr.RasterizeCudaContext() if dr else None
    
    # --- Main Loop: Iterate over each object ---
    for object_dir_path in [d for d in object_dirs if 'vis' not in d]:
        object_id_str = os.path.basename(object_dir_path) # e.g., "object_0"
        object_id = int(object_id_str.split('_')[-1])   # e.g., 0

        logging.info(f"--- Processing {object_id_str} ---")

        # 1. Load mesh and apply scale
        mesh_key = f"image_{object_id:04d}_segmented"
        mesh_file = os.path.join(args.asset_dir, f"{mesh_key}.glb")
        if not os.path.exists(mesh_file):
            logging.warning(f"Mesh file not found for {object_id_str}: {mesh_file}. Skipping.")
            continue
            

        scale = transformations[mesh_key]['scale']
        # Load the file, which might be a Scene or a single Trimesh
        loaded_object = trimesh.load(mesh_file, force='mesh')

        # If the loaded object is a scene, concatenate all its geometries into a single mesh
        if isinstance(loaded_object, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded_object.geometry.values()))
        else:
            # It was already a single mesh
            mesh = loaded_object

        # Now that we are sure we have a single mesh, apply the scale
        mesh.apply_scale(scale)
        
        logging.info(f"Loaded mesh {mesh_file} and applied scale {scale:.4f}")

        # For visualization
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        
        # 2. Initialize the pose estimator for the current object
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        est = FoundationPose(
            model_pts=mesh.vertices, 
            model_normals=mesh.vertex_normals, 
            mesh=mesh, 
            scorer=scorer, 
            refiner=refiner, 
            debug=args.debug,
            debug_dir=args.output_dir, # Use a common debug dir
            glctx=glctx
        )
        logging.info(f"FoundationPose estimator initialized for {object_id_str}")

        # 3. Create a data reader for this object
        reader = CustomDataReader(data_dir=args.input_dir, object_id_str=object_id_str)

        # Prepare per-object output directories
        pose_output_dir = os.path.join(args.output_dir, 'poses', object_id_str)
        vis_output_dir = os.path.join(args.output_dir, 'vis', object_id_str)
        os.makedirs(pose_output_dir, exist_ok=True)
        if args.debug >= 1:
            os.makedirs(vis_output_dir, exist_ok=True)

        # --- Frame Loop: Iterate over each frame for the current object ---
        for i in range(len(reader)):
            logging.info(f"Processing {object_id_str}, frame {i}/{len(reader)-1}")
            
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            K = reader.K

            if i == 0:
                # Initial registration using the mask
                mask = reader.get_mask(i)
                pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            else:
                # Track from the previous frame's pose
                pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

            # Save the estimated pose
            frame_id_str = f"{i:06d}"
            np.savetxt(os.path.join(pose_output_dir, f'{frame_id_str}.txt'), pose.reshape(4,4))

            # Generate visualizations if debug level is high enough
            if args.debug >= 1:
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                
                if args.debug >= 2:
                    imageio.imwrite(os.path.join(vis_output_dir, f'{frame_id_str}.png'), vis)

        logging.info(f"--- Finished processing {object_id_str} ---")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input data folder.')
    parser.add_argument('--asset_dir', type=str, required=True, help='Path to the asset folder containing .glb files.')
    parser.add_argument('--output_dir', type=str, default=f'{code_dir}/output', help='Directory to save poses and visualizations.')
    
    parser.add_argument('--est_refine_iter', type=int, default=5, help='Number of iterations for initial pose estimation.')
    parser.add_argument('--track_refine_iter', type=int, default=2, help='Number of iterations for tracking refinement.')
    parser.add_argument('--debug', type=int, default=2, help='Debug level. 1 for basic vis, 2 to save vis images.')
    
    args = parser.parse_args()
    main(args)