#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace the generated object texture with the photograph wherever the object's
surface is visible from the frame-0 camera.

The mesh generator paints the whole object, including the faces it never saw,
so the parts that *are* in the photo come out blurred and the printing on them
is unreadable. The geometry is already registered to the frame-0 camera by
scenario_construction, so every texel of a placed mesh can be projected into
that photo and, when it is in front and inside the object mask, filled with the
pixel it came from. The rest of the texture is kept but gain corrected to the
photo, so the observed and the generated halves of the object match in tone.

Inputs:
    - outputs/{key}/scene/scene.pkl
Outputs:
    - outputs/{key}/reconstruction/objects/{oid}_{name}_optimized.glb (re-textured)
    - outputs/{key}/reconstruction/objects/{oid}_{name}_optimized_sam.glb
      (the mesh as the generator textured it, kept so the two can be diffed)
    - outputs/{key}/reconstruction/scenario/scene_optimized.glb (rebuilt)
    - outputs/{key}/scene/scene.pkl (updated)
Note:
    - Only the texture image changes: same single geometry, same material, same
      vertex order, same faces and same UVs, written back to the same path, so
      "optimized" keeps pointing where it did and scene.json does not change.
    - Set `photo_texture: false` in the reconstruction config to skip the stage.
"""

import os
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml
from PIL import Image

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Rec. 601 luma, used to split the texture into a dark and a light cluster.
LUMA = np.array([0.299, 0.587, 0.114])

# How wide the ramp from "left to the generated texture" to "fully taken from
# the photo" is, in cosine of the angle between the surface normal and the view
# ray. A hard threshold would print a visible step across a curved surface.
GRAZING_SOFTNESS = 0.1

# Gains past these are a sign that the cluster split caught two unrelated
# things rather than a tone difference, so the correction is held back.
MIN_GAIN, MAX_GAIN = 0.25, 4.0


# ──────────────────────────────── loading ──────────────────────────────── #

def load_single(mesh_path):
    """
    The one geometry of a glb, placed, with its material and UVs kept.

    The default processing merges vertices, which would renumber them, and
    that is not acceptable here: the texture is written back onto this very
    mesh. The node transform has to be applied by hand in exchange, because
    the background mesh keeps its gravity alignment in the scene graph rather
    than in its vertices, and reading the vertices alone puts it elsewhere.
    """
    loaded = trimesh.load(mesh_path, process=False)
    if not isinstance(loaded, trimesh.Scene):
        return loaded
    if not loaded.geometry:
        raise ValueError(f"{mesh_path} holds no geometry")
    name = next(iter(loaded.geometry))
    mesh = loaded.geometry[name]
    for node in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node]
        if geometry_name != name or np.allclose(transform, np.eye(4)):
            continue
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        break
    return mesh


def frame0_photo(scene_dict):
    """The frame-0 colour image, the one the objects were registered against."""
    photo = scene_dict.get("recon", {}).get("foreground")
    if photo is None:
        photo = np.asarray(scene_dict["images"])[0]
    return np.asarray(photo, dtype=np.float32)[..., :3]


def frame0_mask(obj_meta, shape, erode_pixels):
    """
    The object's frame-0 mask, eroded by a couple of pixels.

    Masks come out of the annotator as [H, W] and out of the propagation as
    [N, H, W]; frame 0 is the one the photo belongs to either way. The erosion
    keeps the silhouette's halo pixels, which are part background, out of the
    texture.
    """
    mask = np.asarray(obj_meta["mask"])
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != tuple(shape):
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if erode_pixels > 0:
        kernel = np.ones((2 * int(erode_pixels) + 1,) * 2, np.uint8)
        mask = cv2.erode(mask, kernel)
    return mask > 0


def frame0_camera(scene_dict):
    """Intrinsics, OpenCV camera-to-world pose and size of the frame-0 camera."""
    cam = scene_dict["info"]["camera"]
    # abs: one scene has a negative stored focal length, see OPE-193
    intrinsics = np.array([[abs(float(cam["fx"])), 0.0, float(cam["cx"])],
                           [0.0, abs(float(cam["fy"])), float(cam["cy"])],
                           [0.0, 0.0, 1.0]])
    pose = np.asarray(cam["camera_opencv_to_world"], dtype=np.float64).reshape(4, 4)
    return intrinsics, pose, int(float(cam["width"])), int(float(cam["height"]))


# ─────────────────────────────── rendering ─────────────────────────────── #

def scene_depth(mesh_paths, intrinsics, cam_to_world, width, height):
    """
    Depth of the whole scene in the frame-0 camera.

    The test a texel has to pass is against *everything*, not against its own
    object: a face can be hidden behind another object or behind the background
    mesh, and its texel must then keep the generated colour instead of picking
    up whatever stands in front of it.
    """
    from openreal2sim.steps.reconstruction.steps.pose_candidate_search import ObjectRenderer

    world_to_cam = np.linalg.inv(cam_to_world)
    vertices, faces, offset = [], [], 0
    for mesh_path in mesh_paths:
        mesh = load_single(mesh_path)
        in_cam = trimesh.transform_points(
            np.asarray(mesh.vertices, dtype=np.float64), world_to_cam)
        vertices.append(in_cam)
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(in_cam)
    combined = trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces),
                               process=False)
    renderer = ObjectRenderer(intrinsics, width, height)
    try:
        _, depth = renderer.render(combined)
    finally:
        renderer.close()
    return np.asarray(depth, dtype=np.float32)


def texel_positions(mesh, tex_height, tex_width):
    """
    World position and face index of every texel, by rasterising the UV layout.

    This is the inverse of the usual render: instead of asking what colour a
    pixel has, it asks where in the scene a texel sits, which is what lets the
    photo be *gathered* into the texture at full texture resolution.
    """
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    # glTF texture row 0 is v = 1
    corners = np.stack([uv[:, 0] * (tex_width - 1),
                        (1.0 - uv[:, 1]) * (tex_height - 1)], axis=1)

    position = np.full((tex_height, tex_width, 3), np.nan, dtype=np.float32)
    face_id = np.full((tex_height, tex_width), -1, dtype=np.int32)
    for index, face in enumerate(np.asarray(mesh.faces)):
        tri = corners[face]
        x0 = max(int(np.floor(tri[:, 0].min())), 0)
        x1 = min(int(np.ceil(tri[:, 0].max())), tex_width - 1)
        y0 = max(int(np.floor(tri[:, 1].min())), 0)
        y1 = min(int(np.ceil(tri[:, 1].max())), tex_height - 1)
        if x1 < x0 or y1 < y0:
            continue
        first, second = tri[1] - tri[0], tri[2] - tri[0]
        determinant = first[0] * second[1] - first[1] * second[0]
        if abs(determinant) < 1e-12:  # a UV-degenerate triangle covers nothing
            continue
        grid_x, grid_y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        delta_x, delta_y = grid_x - tri[0, 0], grid_y - tri[0, 1]
        beta = (delta_x * second[1] - delta_y * second[0]) / determinant
        gamma = (delta_y * first[0] - delta_x * first[1]) / determinant
        alpha = 1.0 - beta - gamma
        inside = (alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0)
        if not inside.any():
            continue
        bary = np.stack([alpha[inside], beta[inside], gamma[inside]], axis=1)
        position[grid_y[inside], grid_x[inside]] = bary @ vertices[face]
        face_id[grid_y[inside], grid_x[inside]] = index
    return position, face_id


# ─────────────────────────────── visibility ────────────────────────────── #

def texel_weights(position, face_id, face_normals, intrinsics, cam_to_world,
                  depth, mask, tolerance, max_grazing_deg):
    """
    Per texel: where it lands in the photo, and how much of the photo to take.

    A texel counts as observed when it projects inside the image and inside the
    object mask, when the scene depth at that pixel is its own depth rather
    than something closer, and when its face is not seen edge on. Texels past
    the grazing limit get weight 0: one photo pixel there covers so much
    surface that copying it smears the texture.
    """
    height, width = depth.shape
    flat = position.reshape(-1, 3).astype(np.float64)
    faces_flat = face_id.reshape(-1)
    covered = faces_flat >= 0

    rotation, centre = cam_to_world[:3, :3], cam_to_world[:3, 3]
    in_cam = (flat - centre) @ rotation  # world -> camera is the transpose
    forward = in_cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = intrinsics[0, 0] * in_cam[:, 0] / forward + intrinsics[0, 2]
        v = intrinsics[1, 1] * in_cam[:, 1] / forward + intrinsics[1, 2]
    finite = covered & np.isfinite(u) & np.isfinite(v) & (forward > 0)
    in_image = finite & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)

    column = np.clip(np.nan_to_num(u), 0, width - 1).round().astype(np.int64)
    row = np.clip(np.nan_to_num(v), 0, height - 1).round().astype(np.int64)
    # the dilation gives the depth test the one pixel of slack a rasterised
    # silhouette needs, so a texel right at the edge is not called occluded
    front = cv2.dilate(depth, np.ones((3, 3), np.uint8))[row, column]
    observed = in_image & (front > 0) & (forward <= front + tolerance) & mask[row, column]

    view = centre - flat
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
    normals = face_normals[np.clip(faces_flat, 0, None)]
    cosine = np.einsum("ij,ij->i", view, normals)
    # a generator can hand back inward normals; the depth test already rejects
    # back faces, so the sign is read off the texels that passed it
    flipped = observed.any() and cosine[observed].mean() < 0.0
    if flipped:
        cosine = -cosine

    limit = np.cos(np.radians(float(max_grazing_deg)))
    weight = np.where(observed,
                      np.clip((cosine - limit) / GRAZING_SOFTNESS, 0.0, 1.0), 0.0)
    shape = face_id.shape
    return (weight.reshape(shape), u.reshape(shape), v.reshape(shape),
            {"normals_flipped": bool(flipped)})


# ────────────────────────────── colour matching ────────────────────────── #

def colour_clusters(samples, count, iterations=12):
    """
    Cluster centres of the generated colours, started from luminance quantiles.

    Lloyd's algorithm from a deterministic seed, so the same texture always
    gives the same clusters. The seed is the luminance split the agent used;
    the iterations then let a cluster follow a hue, which is what a fused
    multi-coloured object needs.
    """
    luma = samples @ LUMA
    edges = np.quantile(luma, np.linspace(0.0, 1.0, count + 1))[1:-1]
    labels = np.clip(np.searchsorted(edges, luma), 0, count - 1)
    centres = np.stack([samples[labels == i].mean(axis=0) if np.any(labels == i)
                        else samples.mean(axis=0) for i in range(count)])
    for _ in range(iterations):
        labels = np.argmin(((samples[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        moved = np.stack([samples[labels == i].mean(axis=0) if np.any(labels == i)
                          else centres[i] for i in range(count)])
        if np.allclose(moved, centres, atol=1e-3):
            break
        centres = moved
    return centres, labels


def gain_correct(texture, photo_colour, strong, clusters):
    """
    Scale the generated texture so its colour matches the photo.

    The gain is fitted on the texels that *are* observed and applied to the
    whole texture, which carries the correction onto the faces the camera never
    saw. One gain over a two-tone object (dark suede, light fleece) splits the
    difference and matches neither, so the fit is done per colour cluster and
    every texel takes the gain of the cluster its generated colour is nearest
    to. Clustering on colour rather than on luminance alone matters for the
    objects the segmentation fused: a black fob and a silver key share a
    luminance band with the red and yellow ribbon next to them, and one gain
    over that band tints both of them.
    """
    if int(strong.sum()) < 500:
        return texture, []
    generated, observed = texture[strong], photo_colour[strong]
    count = max(int(clusters), 1)
    if count == 1:
        gain = np.clip((observed.mean(axis=0) + 1.0) / (generated.mean(axis=0) + 1.0),
                       MIN_GAIN, MAX_GAIN)
        return np.clip(texture * gain, 0.0, 255.0), [
            {"texels": int(strong.sum()), "gain_rgb": gain.round(3).tolist()}]

    centres, labels = colour_clusters(generated, count)
    distance = np.stack([((texture - centre) ** 2).sum(-1) for centre in centres], axis=-1)
    nearest = np.argmin(distance, axis=-1)
    corrected, report = texture.copy(), []
    for index in range(count):
        fitted = labels == index
        if int(fitted.sum()) < 100:
            continue
        gain = (observed[fitted].mean(axis=0) + 1.0) / (generated[fitted].mean(axis=0) + 1.0)
        gain = np.clip(gain, MIN_GAIN, MAX_GAIN)
        corrected[nearest == index] = np.clip(texture[nearest == index] * gain, 0.0, 255.0)
        report.append({"texels": int(fitted.sum()),
                       "centre_rgb": centres[index].round(1).tolist(),
                       "gain_rgb": gain.round(3).tolist()})
    return corrected, report


def fill_gutters(texture, covered):
    """
    Push the nearest covered colour into the texels no triangle covers.

    Those gutters are never sampled at the centre of a texel, but bilinear
    filtering and mip-mapping reach into them at chart edges, and whatever was
    left there bleeds into the surface as a seam.
    """
    empty = (~covered).astype(np.uint8)
    if not empty.any() or not covered.any():
        return texture
    _, labels = cv2.distanceTransformWithLabels(
        empty, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    rows, columns = np.nonzero(covered)
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lookup[labels[rows, columns]] = rows * texture.shape[1] + columns
    nearest = texture.reshape(-1, 3)[lookup[labels].reshape(-1)].reshape(texture.shape)
    return np.where(empty[..., None] > 0, nearest, texture)


# ──────────────────────────── super resolution ─────────────────────────── #

def super_resolve(photo, model_id, scale=4):
    """
    Optional 4x upscale of the photo before it is projected.

    No checkpoint ships in the image, so this is off by default and a failure
    to reach one is a warning rather than an error: the stage then projects the
    photo at its own resolution, which is what the scale factor returned says.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
    except Exception as error:
        print(f"[Warn] No super-resolution backend ({error}); projecting at 1x")
        return photo, 1
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_id).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        image = Image.fromarray(photo.astype(np.uint8))
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model(**inputs).reconstruction
        upscaled = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
    except Exception as error:
        print(f"[Warn] Super resolution with {model_id} failed ({error}); projecting at 1x")
        return photo, 1
    height, width = photo.shape[:2]
    upscaled = upscaled[:height * scale, :width * scale]
    if upscaled.shape[:2] != (height * scale, width * scale):
        print("[Warn] Super resolution returned an unexpected size; projecting at 1x")
        return photo, 1
    print(f"[Info] Photo super resolved to {upscaled.shape[1]}x{upscaled.shape[0]}")
    return upscaled.astype(np.float32), scale


# ───────────────────────────────── one object ──────────────────────────── #

def project_photo_onto_object(mesh_path, backup_path, photo, photo_scale, mask,
                              intrinsics, cam_to_world, depth, cfg):
    """
    Re-texture one placed object from the photo. Returns a report, or None when
    the mesh carries no texture to write into.
    """
    # a re-run must read the generated texture, not the one this stage wrote
    if not Path(backup_path).exists():
        shutil.copy(mesh_path, backup_path)
    mesh = load_single(backup_path)

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if uv is None or image is None:
        print(f"[Warn] {Path(mesh_path).name} has no UV texture, left as it is")
        return None

    texture = np.asarray(image.convert("RGB"), dtype=np.float32)
    tex_height, tex_width = texture.shape[:2]
    position, face_id = texel_positions(mesh, tex_height, tex_width)
    covered = face_id >= 0

    weight, u, v, flags = texel_weights(
        position, face_id, np.asarray(mesh.face_normals, dtype=np.float64),
        intrinsics, cam_to_world, depth, mask,
        float(cfg["photo_texture_depth_tolerance"]),
        float(cfg["photo_texture_max_grazing_deg"]))

    # bilinear sample, in the upscaled photo's pixel grid when there is one
    offset = (photo_scale - 1) / 2.0
    map_x = np.nan_to_num(u * photo_scale + offset).astype(np.float32)
    map_y = np.nan_to_num(v * photo_scale + offset).astype(np.float32)
    photo_colour = cv2.remap(photo, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    corrected, gains = gain_correct(
        texture, photo_colour, weight > 0.7, int(cfg["photo_texture_gain_clusters"]))

    # feather the seam between the photo and the generated texture, but only
    # inwards, so the blur never pulls photo colour onto an unobserved texel
    sigma = float(cfg["photo_texture_blend_sigma"])
    feathered = np.minimum(cv2.GaussianBlur(weight, (0, 0), sigma), weight) if sigma > 0 else weight
    blended = corrected * (1.0 - feathered[..., None]) + photo_colour * feathered[..., None]
    blended = fill_gutters(blended, covered)

    material.baseColorTexture = Image.fromarray(
        np.clip(blended, 0, 255).astype(np.uint8))
    mesh.export(mesh_path)

    report = {
        "texture_size": [tex_width, tex_height],
        "covered_fraction": float(covered.mean()),
        "photo_fraction_of_covered": float((weight > 0.3).sum() / max(int(covered.sum()), 1)),
        "photo_scale": int(photo_scale),
        "gain_clusters": gains,
    }
    report.update(flags)
    return report


# ──────────────────────────────── main function ────────────────────────── #

def object_photo_texture(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        cfg = key_cfgs[key]
        if not cfg.get("photo_texture", True):
            print(f"[Info] [{key}] photo_texture is off, keeping the generated textures")
            continue

        objects = scene_dict["info"].get("objects", {})
        if not objects:
            print(f"[Warning] No objects for key {key}, skipping")
            continue

        photo = frame0_photo(scene_dict)
        intrinsics, cam_to_world, width, height = frame0_camera(scene_dict)
        photo_scale = 1
        if cfg.get("photo_texture_super_resolution", False):
            photo, photo_scale = super_resolve(
                photo, cfg["photo_texture_super_resolution_model"])

        background_path = scene_dict["info"]["background"]["registered"]
        mesh_paths = [background_path] + [obj["optimized"] for obj in objects.values()]
        print(f"[Info] [{key}] Rendering the scene depth of {len(mesh_paths)} meshes...")
        depth = scene_depth(mesh_paths, intrinsics, cam_to_world, width, height)

        for oid, obj in objects.items():
            name = obj["name"]
            mesh_path = Path(obj["optimized"])
            backup_path = mesh_path.with_name(f"{mesh_path.stem}_sam.glb")
            mask = frame0_mask(scene_dict["objects"][oid],
                               (height, width),
                               int(cfg["photo_texture_mask_erode_pixels"]))
            print(f"[Info] [{key}] Projecting the photo onto {oid}_{name}...")
            report = project_photo_onto_object(
                str(mesh_path), str(backup_path), photo, photo_scale, mask,
                intrinsics, cam_to_world, depth, cfg)
            if report is None:
                continue
            obj["optimized_sam"] = str(backup_path)
            obj["photo_texture"] = report
            scene_dict["info"]["objects"][oid] = obj
            print(f"[Info] [{key}] {oid}_{name}: {report['photo_fraction_of_covered']:.1%} "
                  f"of the surface came from the photo, generated texture kept elsewhere")

        # the merged scene is the same meshes, so it has to be written again
        scene = trimesh.Scene()
        scene.add_geometry(load_single(background_path), "background")
        for oid, obj in objects.items():
            scene.add_geometry(load_single(obj["optimized"]), f"obj{oid}_{obj['name']}")
        merged = Path(scene_dict["info"]["scene_mesh"]["optimized"])
        merged.parent.mkdir(parents=True, exist_ok=True)
        scene.export(str(merged))
        print(f"[Info] [{key}] Re-textured scene saved to {merged}")

        key_scene_dicts[key] = scene_dict
        with open(base_dir / f"outputs/{key}/scene/scene.pkl", "wb") as handle:
            pickle.dump(scene_dict, handle)

    return key_scene_dicts


# ──────────────────────────────────── main ─────────────────────────────── #

if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg = yaml.safe_load((base_dir / "config" / "config.yaml").open("r"))
    keys = cfg["keys"]
    from openreal2sim.utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg, "reconstruction") for key in keys}
    key_scene_dicts = {}
    for key in keys:
        with open(base_dir / f"outputs/{key}/scene/scene.pkl", "rb") as handle:
            key_scene_dicts[key] = pickle.load(handle)
    object_photo_texture(keys, key_scene_dicts, key_cfgs)
