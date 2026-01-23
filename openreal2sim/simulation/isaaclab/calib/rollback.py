#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rollback script to restore original optimized meshes from backup.
This script restores the 'original' mesh files back to 'optimized' files
and recalculates object bounds from the restored mesh.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[WARN] trimesh not available. Will not recalculate bounds.")


def rollback_optimized_meshes(key: str, dry_run: bool = False) -> None:
    """
    Rollback optimized meshes to their original versions.
    
    Args:
        key: Scene key (e.g., "demo_video")
        dry_run: If True, only print what would be done without actually doing it
    """
    # Load scene.json
    scene_json_path = Path.cwd() / "outputs" / key / "simulation" / "scene.json"
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")
    
    with open(scene_json_path, "r", encoding="utf-8") as f:
        scene_dict = json.load(f)
    
    if "objects" not in scene_dict:
        print("[WARN] No objects found in scene.json")
        return
    
    restored_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"[INFO] {'[DRY RUN] ' if dry_run else ''}Rolling back optimized meshes for key: {key}")
    print("-" * 60)
    
    for oid_str, obj_info in scene_dict["objects"].items():
        # Check if original backup exists
        if "original" not in obj_info:
            print(f"[SKIP] Object {oid_str}: No 'original' backup found")
            skipped_count += 1
            continue
        
        if "optimized" not in obj_info:
            print(f"[SKIP] Object {oid_str}: No 'optimized' path found")
            skipped_count += 1
            continue
        
        original_path_str = obj_info["original"]
        optimized_path_str = obj_info["optimized"]
        
        # Handle absolute paths (remove /app prefix if present)
        if original_path_str.startswith("/app/"):
            original_path = Path.cwd() / original_path_str[5:]
        elif Path(original_path_str).is_absolute():
            original_path = Path(original_path_str)
        else:
            original_path = Path.cwd() / original_path_str
        
        if optimized_path_str.startswith("/app/"):
            optimized_path = Path.cwd() / optimized_path_str[5:]
        elif Path(optimized_path_str).is_absolute():
            optimized_path = Path(optimized_path_str)
        else:
            optimized_path = Path.cwd() / optimized_path_str
        
        # Check if original file exists
        if not original_path.exists():
            print(f"[ERROR] Object {oid_str}: Original backup not found: {original_path}")
            error_count += 1
            continue
        
        # Perform rollback
        try:
            if dry_run:
                print(f"[DRY RUN] Would restore: {original_path} -> {optimized_path}")
                if TRIMESH_AVAILABLE:
                    print(f"[DRY RUN] Would recalculate bounds from restored mesh")
            else:
                # Step 1: Copy original back to optimized
                shutil.copy2(original_path, optimized_path)
                
                # Step 2: Recalculate bounds from the restored mesh
                if TRIMESH_AVAILABLE:
                    try:
                        # Load the restored mesh
                        mesh = trimesh.load(str(optimized_path))
                        
                        # Calculate bounds from mesh
                        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                        object_min = bounds[0].tolist()  # [min_x, min_y, min_z]
                        object_max = bounds[1].tolist()  # [max_x, max_y, max_z]
                        object_center = ((bounds[0] + bounds[1]) / 2.0).tolist()  # [center_x, center_y, center_z]
                        
                        # Update scene.json with recalculated bounds
                        scene_dict["objects"][oid_str]["object_min"] = object_min
                        scene_dict["objects"][oid_str]["object_max"] = object_max
                        scene_dict["objects"][oid_str]["object_center"] = object_center
                        
                        print(f"[OK] Object {oid_str}: Restored {optimized_path.name}")
                        print(f"      Recalculated bounds: min={object_min}, max={object_max}, center={object_center}")
                    except Exception as e:
                        print(f"[WARN] Object {oid_str}: Failed to recalculate bounds: {e}")
                        # Fallback: try to restore from backup parameters if available
                        if "original_object_min" in obj_info:
                            scene_dict["objects"][oid_str]["object_min"] = obj_info["original_object_min"]
                        if "original_object_max" in obj_info:
                            scene_dict["objects"][oid_str]["object_max"] = obj_info["original_object_max"]
                        if "original_object_center" in obj_info:
                            scene_dict["objects"][oid_str]["object_center"] = obj_info["original_object_center"]
                        print(f"      Restored bounds from backup parameters")
                else:
                    # Fallback: restore from backup parameters if trimesh not available
                    if "original_object_min" in obj_info:
                        scene_dict["objects"][oid_str]["object_min"] = obj_info["original_object_min"]
                    if "original_object_max" in obj_info:
                        scene_dict["objects"][oid_str]["object_max"] = obj_info["original_object_max"]
                    if "original_object_center" in obj_info:
                        scene_dict["objects"][oid_str]["object_center"] = obj_info["original_object_center"]
                    print(f"[OK] Object {oid_str}: Restored {optimized_path.name} (bounds from backup)")
            
            restored_count += 1
        except Exception as e:
            print(f"[ERROR] Object {oid_str}: Failed to restore: {e}")
            error_count += 1
    
    # Save updated scene.json if not dry run
    if not dry_run and restored_count > 0:
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, indent=2)
        print(f"[INFO] Updated scene.json with restored parameters")
    
    print("-" * 60)
    print(f"[INFO] Rollback complete:")
    print(f"  - Restored: {restored_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Errors: {error_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rollback optimized meshes to original versions")
    parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Scene key (outputs/<key>/simulation/scene.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: show what would be done without actually doing it",
    )
    
    args = parser.parse_args()
    
    try:
        rollback_optimized_meshes(args.key, dry_run=args.dry_run)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

