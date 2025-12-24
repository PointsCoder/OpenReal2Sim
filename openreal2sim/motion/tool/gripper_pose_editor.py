#!/usr/bin/env python3
"""
Interactive Gripper Pose Editor Tool

A Flask web application for loading scene GLB files and positioning a Franka gripper.
Supports two modes:
1. Transform both scene and gripper together
2. Transform only gripper (scene stays fixed)

Usage:
    python gripper_pose_editor.py --key <scene_key> [--port 5000]
"""

import json
import argparse
from pathlib import Path
import numpy as np
import trimesh
import trimesh.transformations as tra
import open3d as o3d
from flask import Flask, render_template, request, jsonify, send_from_directory
import sys

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(base_dir / 'third_party' / 'GraspGen'))
sys.path.append(str(base_dir / 'openreal2sim' / 'motion' / 'utils'))
sys.path.append(str(base_dir / 'openreal2sim' / 'motion' / 'modules'))
from grasp_utils import save_grasps_npz, load_grasps_npz

GRIPPER_NAME = 'franka_panda'

app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / 'static'),
            static_folder=str(Path(__file__).parent / 'static'))


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(app.static_folder, filename)

# Global state
scene_data = {}
gripper_mesh = None
gripper_pose = np.eye(4, dtype=np.float32)
scene_transform = np.eye(4, dtype=np.float32)


def load_scene_glb(scene_key: str):
    """Load scene GLB from simulation folder."""
    base = Path.cwd()
    scene_json_path = base / 'outputs' / scene_key / 'simulation' / 'scene.json'
    
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")
    
    with open(scene_json_path, 'r') as f:
        scene_config = json.load(f)
    
    # Load background mesh
    bg_data = scene_config.get('background', {})
    bg_path = bg_data.get('registered') or bg_data.get('original')
    
    if bg_path and bg_path.startswith('/app/'):
        bg_path = bg_path.replace('/app/', str(base) + '/')
    elif not Path(bg_path).is_absolute():
        bg_path = scene_json_path.parent / bg_path
    
    bg_path = Path(bg_path)
    if not bg_path.exists():
        raise FileNotFoundError(f"Background GLB not found: {bg_path}")
    
    scene_mesh = trimesh.load(str(bg_path), force='mesh')
    if isinstance(scene_mesh, trimesh.Scene):
        scene_mesh = scene_mesh.dump(concatenate=True)
    
    # Load objects
    objects = {}
    for obj_id, obj_data in scene_config.get('objects', {}).items():
        obj_mesh_path = obj_data.get('optimized') or obj_data.get('registered')
        if obj_mesh_path:
            if obj_mesh_path.startswith('/app/'):
                obj_mesh_path = obj_mesh_path.replace('/app/', str(base) + '/')
            elif not Path(obj_mesh_path).is_absolute():
                obj_mesh_path = scene_json_path.parent / obj_mesh_path
            
            obj_mesh_path = Path(obj_mesh_path)
            if obj_mesh_path.exists():
                obj_mesh = trimesh.load(str(obj_mesh_path), force='mesh')
                if isinstance(obj_mesh, trimesh.Scene):
                    obj_mesh = obj_mesh.dump(concatenate=True)
                objects[obj_id] = {
                    'mesh': obj_mesh,
                    'name': obj_data.get('name', obj_id),
                    'oid': obj_data.get('oid', obj_id),
                }
    
    return {
        'background': scene_mesh,
        'objects': objects,
        'scene_json': scene_config,
        'bg_path': bg_path,
    }


def load_franka_gripper():
    """Load Franka Panda gripper mesh directly from GraspGen assets folder."""
    base_dir = Path(__file__).parent.parent.parent.parent
    gripper_assets_dir = base_dir / 'third_party' / 'GraspGen' / 'assets' / 'panda_gripper'
    
    if not gripper_assets_dir.exists():
        raise FileNotFoundError(f"Gripper assets directory not found: {gripper_assets_dir}")
    
    # Load STL files
    hand_stl = gripper_assets_dir / 'hand.stl'
    finger_stl = gripper_assets_dir / 'finger.stl'
    
    if not hand_stl.exists() or not finger_stl.exists():
        raise FileNotFoundError(f"Gripper STL files not found in {gripper_assets_dir}")
    
    # Load meshes
    base_mesh = trimesh.load(str(hand_stl))
    finger_mesh = trimesh.load(str(finger_stl))
    
    # Transform fingers relative to base (same as in franka_panda.py)
    finger_l = finger_mesh.copy()
    finger_r = finger_mesh.copy()
    
    # Apply transforms
    finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
    finger_l.apply_translation([0.04, 0, 0.0584])
    finger_r.apply_translation([-0.04, 0, 0.0584])
    
    # Combine meshes
    fingers = trimesh.util.concatenate([finger_l, finger_r])
    gripper_mesh = trimesh.util.concatenate([fingers, base_mesh])
    
    return gripper_mesh


def append_gripper_to_viz(npz_path, gripper_mesh, grasp_pose, samples=800, color=(0.6, 0.0, 0.6)):
    """Append a sampled, transformed gripper pointcloud (purple) to an existing PLY viz.
    
    If the viz PLY doesn't exist, create a new PLY that contains only the gripper marker.
    
    Args:
        npz_path: Path to the NPZ file (used to determine viz PLY path)
        gripper_mesh: Trimesh object of the gripper
        grasp_pose: 4x4 transformation matrix for the gripper pose
        samples: Number of points to sample from gripper mesh
        color: RGB color tuple for the gripper marker (default: purple)
    
    Returns:
        (success: bool, viz_path: str) - Whether the operation succeeded and the viz file path
    """
    # Determine viz path in same folder as npz_path
    viz_path = npz_path.parent / (npz_path.stem + "_viz.ply")

    # Sample gripper surface points
    try:
        pts, face_idx = trimesh.sample.sample_surface(gripper_mesh, samples)
    except Exception:
        # Fallback to using vertices if sampling fails
        pts = np.asarray(gripper_mesh.vertices, dtype=np.float32)

    # Apply transform (4x4)
    gp = np.array(grasp_pose, dtype=np.float32)
    if gp.shape == (4, 4):
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts_t = (gp @ pts_h.T).T[:, :3]
    else:
        pts_t = pts

    # Create Open3D point cloud for gripper marker
    pcd_gr = o3d.geometry.PointCloud()
    pcd_gr.points = o3d.utility.Vector3dVector(pts_t.astype(np.float32))
    pcd_gr.colors = o3d.utility.Vector3dVector(np.tile(np.array(color, dtype=np.float32), (len(pts_t), 1)))

    if viz_path.exists():
        try:
            existing = o3d.io.read_point_cloud(str(viz_path))
            merged = existing + pcd_gr
            o3d.io.write_point_cloud(str(viz_path), merged)
            return True, str(viz_path)
        except Exception:
            # Fall back to trimesh concatenation if open3d fails
            try:
                existing_tm = trimesh.load(str(viz_path), force='mesh')
                if isinstance(existing_tm, trimesh.Scene):
                    existing_tm = existing_tm.dump(concatenate=True)
                # Convert sampled gripper points into a pointcloud Trimesh
                gm = trimesh.Trimesh(vertices=pts_t, faces=None)
                combined = trimesh.util.concatenate([existing_tm, gm])
                combined.export(str(viz_path))
                return True, str(viz_path)
            except Exception:
                return False, str(viz_path)
    else:
        # Create new PLY containing only the gripper marker
        o3d.io.write_point_cloud(str(viz_path), pcd_gr)
        return True, str(viz_path)


@app.route('/')
def index():
    """Main page."""
    return render_template('gripper_editor.html')


@app.route('/api/load_scene', methods=['POST'])
def api_load_scene():
    """Load scene GLB."""
    global scene_data, gripper_mesh, gripper_pose, scene_transform
    
    try:
        data = request.json
        scene_key = data.get('key')
        
        if not scene_key:
            return jsonify({'error': 'Scene key is required'}), 400
        
        scene_data = load_scene_glb(scene_key)
        gripper_mesh = load_franka_gripper()
        
        # Reset transforms
        gripper_pose = np.eye(4, dtype=np.float32)
        scene_transform = np.eye(4, dtype=np.float32)
        
        # Export meshes to GLB for web display
        output_dir = Path(__file__).parent / 'static' / 'scenes'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save background
        bg_glb_path = output_dir / 'background.glb'
        scene_data['background'].export(str(bg_glb_path))
        
        # Save objects and collect object info
        obj_paths = {}
        obj_info_dict = {}
        for obj_id, obj_info in scene_data['objects'].items():
            obj_glb_path = output_dir / f'object_{obj_id}.glb'
            obj_info['mesh'].export(str(obj_glb_path))
            obj_paths[obj_id] = f'/static/scenes/object_{obj_id}.glb'
            obj_info_dict[obj_id] = {
                'name': obj_info.get('name', obj_id),
                'oid': obj_info.get('oid', obj_id),
            }
        
        # Get manipulated object ID from scene.json
        manipulated_oid = scene_data.get('scene_json', {}).get('manipulated_oid')
        
        # Save gripper
        gripper_glb_path = output_dir / 'gripper.glb'
        gripper_mesh.export(str(gripper_glb_path))
        
        return jsonify({
            'success': True,
            'background': '/static/scenes/background.glb',
            'objects': obj_paths,
            'object_info': obj_info_dict,  # Object names and IDs
            'manipulated_oid': manipulated_oid,  # Manipulated object ID
            'gripper': '/static/scenes/gripper.glb',
            'gripper_pose': gripper_pose.tolist(),
            'scene_transform': scene_transform.tolist(),
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error loading scene: {error_msg}")
        print(traceback_str)
        return jsonify({'error': error_msg, 'traceback': traceback_str}), 500


@app.route('/api/update_pose', methods=['POST'])
def api_update_pose():
    """Update gripper or scene pose."""
    global gripper_pose, scene_transform
    
    try:
        data = request.json
        mode = data.get('mode', 'gripper_only')  # 'both' or 'gripper_only'
        transform = np.array(data.get('transform', np.eye(4).tolist()), dtype=np.float32)
        
        if mode == 'both':
            # Update both scene and gripper
            scene_transform = transform
            gripper_pose = transform
        else:
            # Update only gripper (relative to scene)
            gripper_pose = transform
        
        return jsonify({
            'success': True,
            'gripper_pose': gripper_pose.tolist(),
            'scene_transform': scene_transform.tolist(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_pose', methods=['POST'])
def api_save_pose():
    """Save gripper pose to npz file (similar to grasp_generation.py)."""
    global scene_data, gripper_mesh
    
    try:
        data = request.json
        scene_key = data.get('key')
        manipulated_oid = data.get('manipulated_oid')  # Object ID to save grasp for
        transform = data.get('transform')  # 4x4 matrix from frontend
        
        if not scene_key:
            return jsonify({'error': 'Scene key is required'}), 400
        
        if not manipulated_oid:
            return jsonify({'error': 'Manipulated object ID is required'}), 400
        
        if not transform:
            return jsonify({'error': 'Transform matrix is required'}), 400
        
        base = Path.cwd()
        
        # Find object info from scene_data
        obj_info = None
        obj_name = None
       
        for obj_id, obj_data in scene_data.get('objects', {}).items():
            print(f"obj_id: {obj_id}, manipulated_oid: {manipulated_oid}")
            if int(obj_id) == int(manipulated_oid) or int(obj_data.get('oid')) == int(manipulated_oid):
                obj_info = obj_data
                obj_name = obj_data.get('name', obj_id)
                break
        
        if not obj_info:
            return jsonify({'error': f'Object {manipulated_oid} not found in scene'}), 404
        
        # Determine grasp file path (check both proposals and grasps directories)
        proposals_dir = base / 'outputs' / scene_key / 'proposals'
        grasps_dir = base / 'outputs' / scene_key / 'grasps'
        
        # Try proposals first, then grasps
        npz_path = None
        if proposals_dir.exists():
            npz_path = proposals_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        elif grasps_dir.exists():
            npz_path = grasps_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        else:
            # Create grasps directory if neither exists
            grasps_dir.mkdir(parents=True, exist_ok=True)
            npz_path = grasps_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        
        # Convert transform from frontend (list of 4 lists) to 4x4 numpy array
        # Frontend sends: [[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]]
        grasp_pose = np.array(transform, dtype=np.float32)  # Shape: (4, 4)
        new_grasp = grasp_pose[np.newaxis, :, :]  # Shape: (1, 4, 4)
        
        # Set score to 10.0
        new_score = np.array([10.0], dtype=np.float32)
        

        # Set bite_distance to zeros (2 values per grasp)
        new_bite_distance = np.zeros((1), dtype=np.float32)
        
        # Check if user wants to create new file or append to existing
        create_new = data.get('create_new', False)  # Default: append to existing
        
        if create_new:
            # Create a new npz file with timestamp
            import time
            timestamp = int(time.time() * 1000)  # milliseconds
            npz_filename = f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp_{timestamp}.npz'
            npz_path = (proposals_dir if proposals_dir.exists() else grasps_dir) / npz_filename
            
            # Save new grasp to new NPZ file
            save_grasps_npz(new_grasp, new_score, npz_path, bite_distance=new_bite_distance)
            total_grasps = 1
            
            # Optionally update visualization PLY by appending a purple gripper marker
            save_vis_flag = True
            if save_vis_flag:
                try:
                    if gripper_mesh is None:
                        gripper_mesh = load_franka_gripper()

                    appended, viz_file = append_gripper_to_viz(npz_path, gripper_mesh, grasp_pose)
                except Exception as e_vis:
                    print(f"[WARN] Failed to save viz PLY: {e_vis}")
        else:
            # Append to existing file or create if doesn't exist
            if npz_path.exists():
                existing_grasps, existing_scores, existing_bite_distance = load_grasps_npz(npz_path)
                
                # Append new grasp to existing ones
                grasps = np.vstack([existing_grasps, new_grasp])
                scores = np.concatenate([existing_scores, new_score])
             
                bite_distance = np.vstack([existing_bite_distance, new_bite_distance])
                total_grasps = len(grasps)
            else:
                # First grasp for this object
                grasps = new_grasp
                scores = new_score
                bite_distance = new_bite_distance
                total_grasps = 1
            
            # Save to NPZ (with bite_points and bite_distances)
            save_grasps_npz(grasps, scores, npz_path, bite_distance=bite_distance)

            save_vis_flag = True
            if save_vis_flag:
                try:
                    if gripper_mesh is None:
                        gripper_mesh = load_franka_gripper()

                    appended, viz_file = append_gripper_to_viz(npz_path, gripper_mesh, grasp_pose)
                except Exception as e_vis:
                    print(f"[WARN] Failed to save viz PLY: {e_vis}")
        
        return jsonify({
            'success': True,
            'npz_path': str(npz_path),
            'object_name': obj_name,
            'object_id': manipulated_oid,
            'total_grasps': total_grasps,
            'created_new': create_new,
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error saving pose: {error_msg}")
        print(traceback_str)
        return jsonify({'error': error_msg, 'traceback': traceback_str}), 500


@app.route('/api/get_pose', methods=['GET'])
def api_get_pose():
    """Get current pose."""
    return jsonify({
        'gripper_pose': gripper_pose.tolist(),
        'scene_transform': scene_transform.tolist(),
    })


@app.route('/api/load_grasps', methods=['POST'])
def api_load_grasps():
    """Load saved grasps from npz file."""
    global scene_data
    
    try:
        data = request.json
        scene_key = data.get('key')
        manipulated_oid = data.get('manipulated_oid')
        
        if not scene_key:
            return jsonify({'error': 'Scene key is required'}), 400
        
        if not manipulated_oid:
            return jsonify({'error': 'Manipulated object ID is required'}), 400
        
        base = Path.cwd()
        
        # Find object info
        obj_info = None
        obj_name = None
        for obj_id, obj_data in scene_data.get('objects', {}).items():
            if obj_id == str(manipulated_oid) or obj_data.get('oid') == manipulated_oid:
                obj_info = obj_data
                obj_name = obj_data.get('name', obj_id)
                break
        
        if not obj_info:
            return jsonify({'error': f'Object {manipulated_oid} not found'}), 404
        
        # Find npz file
        proposals_dir = base / 'outputs' / scene_key / 'proposals'
        grasps_dir = base / 'outputs' / scene_key / 'grasps'
        
        npz_path = None
        if (proposals_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz').exists():
            npz_path = proposals_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        elif (grasps_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz').exists():
            npz_path = grasps_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        
        if not npz_path or not npz_path.exists():
            return jsonify({
                'success': True,
                'grasps': [],
                'message': 'No saved grasps found'
            })
        
        # Load grasps
        grasps, scores, bite_distance = load_grasps_npz(npz_path)
        
        # Convert to list format for JSON
        grasps_list = []
        for i in range(len(grasps)):
            grasps_list.append({
                'index': i,
                'pose': grasps[i].tolist(),
                'score': float(scores[i]),
                'bite_distance': bite_distance[i].tolist() if bite_distance is not None else None,
            })
        
        return jsonify({
            'success': True,
            'grasps': grasps_list,
            'total': len(grasps_list),
            'npz_path': str(npz_path),
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error loading grasps: {error_msg}")
        print(traceback_str)
        return jsonify({'error': error_msg, 'traceback': traceback_str}), 500


@app.route('/api/delete_grasp', methods=['POST'])
def api_delete_grasp():
    """Delete a grasp from npz file."""
    global scene_data
    
    try:
        data = request.json
        scene_key = data.get('key')
        manipulated_oid = data.get('manipulated_oid')
        grasp_index = data.get('index')
        
        if not scene_key:
            return jsonify({'error': 'Scene key is required'}), 400
        
        if not manipulated_oid:
            return jsonify({'error': 'Manipulated object ID is required'}), 400
        
        if grasp_index is None:
            return jsonify({'error': 'Grasp index is required'}), 400
        
        base = Path.cwd()
        
        # Find object info
        obj_info = None
        obj_name = None
        for obj_id, obj_data in scene_data.get('objects', {}).items():
            if obj_id == str(manipulated_oid) or obj_data.get('oid') == manipulated_oid:
                obj_info = obj_data
                obj_name = obj_data.get('name', obj_id)
                break
        
        if not obj_info:
            return jsonify({'error': f'Object {manipulated_oid} not found'}), 404
        
        # Find npz file
        proposals_dir = base / 'outputs' / scene_key / 'proposals'
        grasps_dir = base / 'outputs' / scene_key / 'grasps'
        
        npz_path = None
        if (proposals_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz').exists():
            npz_path = proposals_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        elif (grasps_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz').exists():
            npz_path = grasps_dir / f'{obj_info.get("oid", manipulated_oid)}_{obj_name}_grasp.npz'
        
        if not npz_path or not npz_path.exists():
            return jsonify({'error': 'Grasp file not found'}), 404
        
        # Load grasps
        grasps, scores,bite_distance = load_grasps_npz(npz_path)
        
        if grasp_index < 0 or grasp_index >= len(grasps):
            return jsonify({'error': f'Invalid grasp index: {grasp_index}'}), 400
        
        # Remove the grasp at the specified index
        grasps = np.delete(grasps, grasp_index, axis=0)
        scores = np.delete(scores, grasp_index)
        if bite_distance is not None:
            bite_distance = np.delete(bite_distance, grasp_index, axis=0)
        
        # Save updated grasps
        save_grasps_npz(grasps, scores, npz_path, bite_distance=bite_distance)
        
        return jsonify({
            'success': True,
            'total_grasps': len(grasps),
            'message': f'Grasp #{grasp_index + 1} deleted successfully'
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error deleting grasp: {error_msg}")
        print(traceback_str)
        return jsonify({'error': error_msg, 'traceback': traceback_str}), 500


def main():
    parser = argparse.ArgumentParser(description='Interactive Gripper Pose Editor')
    parser.add_argument('--key', type=str, help='Scene key to load initially')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    if args.key:
        try:
            global scene_data, gripper_mesh
            scene_data = load_scene_glb(args.key)
            gripper_mesh = load_franka_gripper()
            print(f"Pre-loaded scene: {args.key}")
        except Exception as e:
            print(f"Warning: Could not pre-load scene: {e}")
    
    print(f"Starting Gripper Pose Editor on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

