import numpy as np
import cv2

def load_ply(ply_path):
    """Load PLY file and extract 3D points."""
    with open(ply_path, 'rb') as f:
        # Read header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError("Not a valid PLY file")
        
        format_type = None
        vertex_count = 0
        properties = []
        in_header = True
        
        while in_header:
            line = f.readline().decode('ascii').strip()
            
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('property'):
                prop_type = line.split()[1]
                prop_name = line.split()[2]
                properties.append((prop_name, prop_type))
            elif line == 'end_header':
                in_header = False
        
        # Find x, y, z indices
        prop_names = [p[0] for p in properties]
        if 'x' not in prop_names or 'y' not in prop_names or 'z' not in prop_names:
            raise ValueError("PLY file must contain x, y, z properties")
        
        x_idx = prop_names.index('x')
        y_idx = prop_names.index('y')
        z_idx = prop_names.index('z')
        
        # Read vertex data
        points = []
        
        if format_type == 'ascii':
            # ASCII format
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip().split()
                x = float(line[x_idx])
                y = float(line[y_idx])
                z = float(line[z_idx])
                points.append([x, y, z])
        
        elif format_type == 'binary_little_endian':
            # Binary format
            dtype_map = {
                'float': 'f4', 'double': 'f8',
                'uchar': 'u1', 'uint8': 'u1',
                'char': 'i1', 'int8': 'i1',
                'ushort': 'u2', 'uint16': 'u2',
                'short': 'i2', 'int16': 'i2',
                'uint': 'u4', 'uint32': 'u4',
                'int': 'i4', 'int32': 'i4'
            }
            
            # Create numpy dtype for binary reading
            dtype_list = [(p[0], dtype_map.get(p[1], 'f4')) for p in properties]
            vertex_data = np.frombuffer(f.read(vertex_count * np.dtype(dtype_list).itemsize), 
                                       dtype=dtype_list, count=vertex_count)
            
            points = np.column_stack([vertex_data['x'], vertex_data['y'], vertex_data['z']])
            return points
        
        else:
            raise ValueError(f"Unsupported PLY format: {format_type}")
    
    return np.array(points)

def project_points_to_image(points, fx, fy, cx, cy, width, height):
    """
    Project 3D points to 2D image plane using camera intrinsics.
    
    Args:
        points: Nx3 array of 3D points in camera coordinates
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates
        width, height: Image dimensions
    
    Returns:
        depth_image: 2D array containing depth values
    """
    # Filter points behind the camera
    valid_mask = points[:, 2] > 0
    points = points[valid_mask]
    
    if len(points) == 0:
        print("Warning: No points in front of camera")
        return np.zeros((height, width), dtype=np.float32)
    
    # Project 3D points to 2D using pinhole camera model
    # u = fx * (X/Z) + cx
    # v = fy * (Y/Z) + cy
    u = (fx * points[:, 0] / points[:, 2] + cx).astype(int)
    v = (fy * points[:, 1] / points[:, 2] + cy).astype(int)
    depth = points[:, 2]
    
    # Filter points within image bounds
    valid_idx = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_idx]
    v = v[valid_idx]
    depth = depth[valid_idx]
    
    # Create depth image
    depth_image = np.zeros((height, width), dtype=np.float32)
    
    # Handle multiple points projecting to same pixel (keep closest)
    for i in range(len(u)):
        if depth_image[v[i], u[i]] == 0 or depth[i] < depth_image[v[i], u[i]]:
            depth_image[v[i], u[i]] = depth[i]
    
    return depth_image

def save_depth_image(depth_image, output_path, normalize=True):
    """Save depth image as TIFF, PNG, or NPY file."""
    if output_path.endswith('.npy'):
        # Save raw depth values
        np.save(output_path, depth_image)
        print(f"Saved raw depth to {output_path}")
    elif output_path.endswith(('.tiff', '.tif')):
        # Save as 32-bit float TIFF with raw depth values
        cv2.imwrite(output_path, depth_image.astype(np.float32))
        print(f"Saved depth as TIFF to {output_path}")
    else:
        # Normalize and save as grayscale image (PNG/JPG)
        if normalize:
            depth_vis = depth_image.copy()
            mask = depth_vis > 0
            if mask.any():
                depth_vis[mask] = (depth_vis[mask] - depth_vis[mask].min()) / (depth_vis[mask].max() - depth_vis[mask].min())
                depth_vis = (depth_vis * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_image, dtype=np.uint8)
        else:
            depth_vis = depth_image.astype(np.uint8)
        
        cv2.imwrite(output_path, depth_vis)
        print(f"Saved depth visualization to {output_path}")

def main():
    # ===== CONFIGURATION - EDIT THESE VALUES =====
    INPUT_PLY = '/app/outputs/demo_video/reconstruction/background_points.ply'          # Path to input PLY file
    OUTPUT_IMAGE = '/app/outputs/demo_video/reconstruction/image_background_depth.png'       # Output path (.tiff, .tif, .png, .jpg, or .npy)

    # Camera intrinsic parameters
    FX = 664.4203491210938      # Focal length x in pixels
    FY = 663.5459594726562     # Focal length y in pixels
    CX = 295.5      # Principal point x
    CY = 166.0      # Principal point y
    WIDTH = 584     # Image width
    HEIGHT = 328    # Image height
    # ============================================
    
    # Load PLY file
    print(f"Loading PLY file: {INPUT_PLY}")
    points = load_ply(INPUT_PLY)
    print(f"Loaded {len(points)} points")
    
    # Project to depth image
    print("Projecting points to image plane...")
    depth_image = project_points_to_image(
        points, 
        FX, FY, CX, CY,
        WIDTH, HEIGHT
    )
    
    # Count valid pixels
    valid_pixels = np.sum(depth_image > 0)
    print(f"Valid depth pixels: {valid_pixels}/{WIDTH * HEIGHT}")
    
    # Save depth image
    save_depth_image(depth_image, OUTPUT_IMAGE)
    
    # Print depth statistics
    if valid_pixels > 0:
        print(f"\nDepth statistics:")
        print(f"  Min depth: {depth_image[depth_image > 0].min():.3f}")
        print(f"  Max depth: {depth_image.max():.3f}")
        print(f"  Mean depth: {depth_image[depth_image > 0].mean():.3f}")

if __name__ == '__main__':
    main()

