"""
Pick the object orientation by render-and-compare, refine it, and rest it on the table.

SAM-3D predicts a full layout, so the placement it gives is a good seed, but it
is free to get a symmetry wrong: a remote comes out turned 180 degrees in the
table plane and a bowl comes out tilted. Both are invisible to a point cloud
fit, which is why there is no ICP here. Instead:

  1. the seed is the SAM-3D placement, in the frame-0 camera frame;
  2. a small set of candidates is built from the seed by the rotations that a
     symmetric object is confused by: yaw 180 about the table normal, 180 about
     each horizontal axis of its oriented bounding box, and the 90 degree yaws
     when the footprint is nearly square;
  3. each candidate is scored by rendering the textured mesh with our own
     camera over several frames of the clip and comparing silhouette, colour
     and depth against the real frames, the propagated object masks and the
     mega-sam depth;
  4. the winner is refined locally, either with the FoundationPose refiner or
     with a pattern search on the same photometric cost, capped so the
     refinement cannot turn the object into a different candidate;
  5. the object is rested on the table: the smallest rotation that puts the
     support face of its oriented bounding box flat on the plane, then a
     translation along the plane normal down to contact.

Every transform here is a proper rigid motion, so the determinant of
object_to_world keeps its sign.
"""

import os
from pathlib import Path

import numpy as np
import trimesh

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


# ─────────────────────────────── rendering ───────────────────────────────

def _pinhole_camera_class():
    """
    A pyrender camera that takes the full intrinsics matrix.

    The pyrender in the image is 0.1.17, which has no IntrinsicsCamera, and
    PerspectiveCamera drops the principal point. Scoring a silhouette against a
    mask needs the principal point, so the projection matrix is built here.
    """
    import pyrender

    class PinholeCamera(pyrender.camera.Camera):
        def __init__(self, fx, fy, cx, cy, znear=0.01, zfar=100.0):
            super().__init__(znear=znear, zfar=zfar)
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        def get_projection_matrix(self, width=None, height=None):
            width, height = float(width), float(height)
            P = np.zeros((4, 4))
            P[0, 0] = 2.0 * self.fx / width
            P[1, 1] = 2.0 * self.fy / height
            P[0, 2] = 1.0 - 2.0 * self.cx / width
            P[1, 2] = 2.0 * self.cy / height - 1.0
            P[2, 2] = (self.zfar + self.znear) / (self.znear - self.zfar)
            P[2, 3] = 2.0 * self.zfar * self.znear / (self.znear - self.zfar)
            P[3, 2] = -1.0
            return P

    return PinholeCamera


class ObjectRenderer:
    """
    Offscreen renderer for one object against one camera.

    The mesh is rendered with ambient light only, so what comes out is the
    texture albedo rather than a shaded image. That is what can be compared
    against a photograph without modelling the lighting of the room.
    """

    def __init__(self, intrinsics, width, height):
        import pyrender

        self._pyrender = pyrender
        self.width, self.height = int(width), int(height)
        K = np.asarray(intrinsics, dtype=np.float64)
        camera_class = _pinhole_camera_class()
        # abs: one scene has a negative stored focal length, see OPE-193
        self.camera = camera_class(abs(K[0, 0]), abs(K[1, 1]), K[0, 2], K[1, 2])
        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def render(self, mesh, transform=None):
        """Albedo and depth of the mesh, which must already be in the camera frame."""
        pyrender = self._pyrender
        scene = pyrender.Scene(ambient_light=np.ones(3), bg_color=[0.0, 0.0, 0.0, 0.0])
        node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(node, pose=np.eye(4) if transform is None else transform)
        scene.add(self.camera, pose=OPENCV_TO_OPENGL)
        color, depth = self.renderer.render(scene)
        return color[..., :3], depth


# ─────────────────────────────── candidates ───────────────────────────────

def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(3)
    return trimesh.transformations.rotation_matrix(
        np.radians(degrees), axis / norm)[:3, :3]


def rotation_about(R, center):
    """4x4 transform applying the rotation R about the given centre."""
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = center - R @ center
    return transform


def footprint_axes(mesh, normal):
    """
    The two horizontal axes of the oriented bounding box, with the in-plane
    extents they span, and whether the footprint is close to square.
    """
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    except Exception:
        return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])], np.ones(2), False
    # rows of the rotation map world to box axes, so the columns are the axes
    axes = np.asarray(to_origin[:3, :3], dtype=np.float64).T
    vertical = int(np.argmax(np.abs(axes.T @ normal)))
    horizontal = [i for i in range(3) if i != vertical]
    spans = np.array([extents[i] for i in horizontal], dtype=np.float64)
    # project onto the plane so the twins are rotations about the table normal
    picked = []
    for i in horizontal:
        axis = axes[:, i] - np.dot(axes[:, i], normal) * normal
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        picked.append(axis / np.linalg.norm(axis))
    ratio = float(spans.min() / max(spans.max(), 1e-9))
    return picked, spans, ratio


def candidate_rotations(mesh, normal, square_ratio=0.8, maximum=8):
    """
    The seed and its symmetric twins, as rotations in the camera frame.

    Order matters only for the printed table: the seed is first, so a tie goes
    to what SAM-3D predicted.
    """
    axes, _spans, ratio = footprint_axes(mesh, normal)
    near_square = ratio >= square_ratio
    yaw180 = _rotation(normal, 180.0)
    flip_a = _rotation(axes[0], 180.0)
    candidates = [
        ("seed", np.eye(3)),
        ("yaw180", yaw180),
        ("flip_a", flip_a),
        ("flip_b", flip_a @ yaw180),
    ]
    if near_square:
        yaw90 = _rotation(normal, 90.0)
        yaw270 = _rotation(normal, -90.0)
        candidates += [
            ("yaw90", yaw90),
            ("yaw270", yaw270),
            ("flip_a_yaw90", flip_a @ yaw90),
            ("flip_a_yaw270", flip_a @ yaw270),
        ]
    return candidates[:maximum], near_square, ratio


# ─────────────────────────────── scoring ───────────────────────────────

def scoring_frames(scene_dict, oid, count=5, min_iou=0.8):
    """
    Frames to score against, spread over the part of the clip where the object
    has not been moved yet.

    These are manipulation videos: the propagated mask follows the object once
    a hand picks it up, so a late frame would disagree with every candidate
    equally and only add noise. The usable span is the run of frames whose mask
    still overlaps the frame-0 mask, which is the pose the reconstruction is
    built from.
    """
    masks = scene_dict["mask"]
    assert 0 in masks and oid in masks[0], (
        f"object {oid} has no propagated mask on frame 0, so no frame can be scored against it")
    reference = np.asarray(masks[0][oid]["mask"], dtype=bool)
    last = 0
    for frame in range(1, int(scene_dict["n_frames"])):
        entry = masks.get(frame, {}).get(oid)
        if entry is None:
            break
        mask = np.asarray(entry["mask"], dtype=bool)
        union = np.logical_or(reference, mask).sum()
        iou = 0.0 if union == 0 else float(np.logical_and(reference, mask).sum() / union)
        if iou < min_iou:
            break
        last = frame
    if count <= 1 or last == 0:
        return [0], last
    step = last / float(count - 1)
    frames = sorted({int(round(index * step)) for index in range(count)})
    return frames, last


def _normalised(values):
    values = values.astype(np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def score_render(color, depth, frame_rgb, frame_depth, object_mask, depth_scale, weights):
    """
    Cost of one rendered view against one real frame, lower is better.

    Three terms, each in [0, 1] so the weights mean what they say:
      silhouette: one minus the intersection over union against the mask;
      colour:     one minus the normalised cross correlation of the albedo
                  against the photograph, which cancels the gain and offset of
                  the room lighting;
      depth:      the median absolute depth error against the mega-sam depth,
                  over depth_scale, which is a fraction of the object size.
                  Without that fraction the term is a rounding error next to
                  the other two and a refinement is free to slide the object
                  along the camera ray.
    """
    render_mask = depth > 1e-6
    union = np.logical_or(render_mask, object_mask).sum()
    intersection = np.logical_and(render_mask, object_mask)
    iou = 0.0 if union == 0 else float(intersection.sum() / union)
    terms = {"silhouette": 1.0 - iou}

    overlap = int(intersection.sum())
    if overlap >= 25:
        rendered = color[intersection].reshape(-1, 3)
        observed = frame_rgb[intersection].reshape(-1, 3)
        correlation = float(np.mean(_normalised(rendered) * _normalised(observed)))
        terms["colour"] = float(np.clip(0.5 * (1.0 - correlation), 0.0, 1.0))
        error = np.abs(depth[intersection] - frame_depth[intersection])
        terms["depth"] = float(np.clip(np.median(error) / max(depth_scale, 1e-6), 0.0, 1.0))
    else:
        # nothing to compare: charge the worst colour and depth so a candidate
        # that misses the mask altogether cannot win on the silhouette term
        terms["colour"] = 1.0
        terms["depth"] = 1.0
    cost = sum(weights[name] * value for name, value in terms.items())
    return cost, terms, iou


class CandidateScorer:
    """Scores a camera-frame mesh against a fixed set of frames of one clip."""

    def __init__(self, key, scene_dict, oid, cfg):
        self.key, self.oid = key, oid
        self.weights = {
            "silhouette": float(cfg["pose_score_weight_silhouette"]),
            "colour": float(cfg["pose_score_weight_colour"]),
            "depth": float(cfg["pose_score_weight_depth"]),
        }
        self.depth_scale_fraction = float(cfg["pose_score_depth_scale_fraction"])
        self.frames, self.static_until = scoring_frames(
            scene_dict, oid,
            count=int(cfg["pose_score_frames"]),
            min_iou=float(cfg["pose_score_mask_iou_min"]),
        )
        width, height = int(scene_dict["width"]), int(scene_dict["height"])
        self.renderer = ObjectRenderer(scene_dict["intrinsics"], width, height)
        extrinsics = np.asarray(scene_dict["extrinsics"], dtype=np.float64).reshape(-1, 4, 4)
        reference = extrinsics[0]
        self.views = []
        for frame in self.frames:
            self.views.append({
                "frame": frame,
                "cam_from_reference": np.linalg.inv(extrinsics[frame]) @ reference,
                "rgb": np.asarray(scene_dict["images"][frame], dtype=np.float64),
                "depth": np.asarray(scene_dict["depths"][frame], dtype=np.float64),
                "mask": np.asarray(scene_dict["mask"][frame][oid]["mask"], dtype=bool),
            })
        print(f"[Info] [{key}] object {oid} scored on frames {self.frames} "
              f"(the mask stays put until frame {self.static_until} of "
              f"{int(scene_dict['n_frames']) - 1})")

    def close(self):
        self.renderer.close()

    def score(self, mesh, transform=None):
        """Total cost of a camera-frame mesh over every scoring frame."""
        depth_scale = self.depth_scale_fraction * float(
            np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        total, per_frame, ious = 0.0, [], []
        sums = {"silhouette": 0.0, "colour": 0.0, "depth": 0.0}
        for view in self.views:
            pose = view["cam_from_reference"]
            if transform is not None:
                pose = pose @ transform
            color, depth = self.renderer.render(mesh, pose)
            cost, terms, iou = score_render(
                color, depth, view["rgb"], view["depth"], view["mask"], depth_scale, self.weights)
            total += cost
            ious.append(iou)
            per_frame.append((view["frame"], cost, terms))
            for name, value in terms.items():
                sums[name] += value
        count = max(len(self.views), 1)
        summary = {name: value / count for name, value in sums.items()}
        summary["iou"] = float(np.mean(ious)) if ious else 0.0
        return total / count, summary, per_frame


def choose_candidate(key, scene_dict, obj_info, mesh, cfg, normal):
    """
    Score the seed and its symmetric twins, and return the winner.

    Returns the winning 4x4 transform in the camera frame, the table of scores
    and the scorer, which the refinement reuses.
    """
    oid = obj_info["oid"]
    scorer = CandidateScorer(key, scene_dict, oid, cfg)
    centre = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    rotations, near_square, ratio = candidate_rotations(
        mesh, normal,
        square_ratio=float(cfg["pose_candidate_square_ratio"]),
        maximum=int(cfg["pose_candidate_max"]),
    )
    print(f"[Info] [{key}] object {oid} footprint ratio {ratio:.3f}, "
          f"{'near square, 90 degree yaws included' if near_square else 'not square'}; "
          f"{len(rotations)} candidates")
    table = []
    for name, R in rotations:
        transform = rotation_about(R, centre)
        cost, summary, _ = scorer.score(mesh, transform)
        table.append({
            "name": name,
            "cost": float(cost),
            "transform": transform.tolist(),
            "silhouette_iou": summary["iou"],
            "colour": summary["colour"],
            "depth": summary["depth"],
        })
        print(f"[Info] [{key}] object {oid} candidate {name:<14} cost {cost:.4f} "
              f"(IoU {summary['iou']:.3f}, colour {summary['colour']:.4f}, "
              f"depth {summary['depth']:.4f})")
    best = min(range(len(table)), key=lambda index: table[index]["cost"])
    runner_up = sorted(entry["cost"] for entry in table)[1] if len(table) > 1 else float("nan")
    print(f"[Info] [{key}] object {oid} best candidate {table[best]['name']} "
          f"cost {table[best]['cost']:.4f}, next best {runner_up:.4f}")
    return np.asarray(table[best]["transform"], dtype=np.float64), table, best, scorer


# ─────────────────────────────── refinement ───────────────────────────────

def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))


def within_refinement_limits(transform, diagonal, max_deg, max_fraction):
    """Whether a refinement stayed close enough to the candidate to be trusted."""
    angle = rotation_angle_deg(transform)
    shift = float(np.linalg.norm(transform[:3, 3]))
    return angle <= max_deg and shift <= max_fraction * diagonal, angle, shift


def refine_photometric(key, oid, mesh, scorer, seed_cost, cfg):
    """
    Pattern search on the same cost the candidates were scored with.

    Gradient free and deliberately small: three rotation and three translation
    axes, a halving step, and a cap on how far the result may sit from the
    candidate. It is the fallback for the FoundationPose refiner and the
    alternative Esteban approved.
    """
    centre = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_deg = float(cfg["pose_refine_max_rotation_deg"])
    max_fraction = float(cfg["pose_refine_max_translation_fraction"])
    best = np.eye(4)
    best_cost = seed_cost
    angle_step = min(8.0, max_deg / 3.0)
    shift_step = 0.05 * diagonal
    for _level in range(int(cfg["pose_refine_levels"])):
        improved = True
        while improved:
            improved = False
            moves = []
            for axis in np.eye(3):
                for sign in (1.0, -1.0):
                    moves.append(rotation_about(_rotation(axis, sign * angle_step), centre))
                    shift = np.eye(4)
                    shift[:3, 3] = sign * shift_step * axis
                    moves.append(shift)
            for move in moves:
                trial = move @ best
                inside, _angle, _shift = within_refinement_limits(
                    trial, diagonal, max_deg, max_fraction)
                if not inside:
                    continue
                cost, _summary, _per_frame = scorer.score(mesh, trial)
                if cost < best_cost - 1e-6:
                    best, best_cost, improved = trial, cost, True
        angle_step *= 0.5
        shift_step *= 0.5
    angle = rotation_angle_deg(best)
    _cost, summary, _per_frame = scorer.score(mesh, best)
    print(f"[Info] [{key}] object {oid} photometric refinement moved {angle:.2f} deg and "
          f"{1000.0 * np.linalg.norm(best[:3, 3]):.1f} mm, cost {seed_cost:.4f} -> {best_cost:.4f} "
          f"(IoU {summary['iou']:.3f}, colour {summary['colour']:.4f}, depth {summary['depth']:.4f})")
    return best, best_cost, {"mode": "photometric", "rotation_deg": angle,
                             "translation_mm": float(1000.0 * np.linalg.norm(best[:3, 3])),
                             "cost_before": float(seed_cost), "cost_after": float(best_cost),
                             "silhouette_iou": summary["iou"], "colour": summary["colour"],
                             "depth": summary["depth"]}


_FDPOSE_PARTS = {}


def _foundationpose_parts():
    """The FoundationPose networks and raster context, built once per process."""
    import sys

    if not _FDPOSE_PARTS:
        repo = str(Path.cwd() / "third_party/FoundationPose")
        if repo not in sys.path:
            sys.path.append(repo)
        from estimater import ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr

        _FDPOSE_PARTS["scorer"] = ScorePredictor()
        _FDPOSE_PARTS["refiner"] = PoseRefinePredictor()
        _FDPOSE_PARTS["glctx"] = dr.RasterizeCudaContext()
    return _FDPOSE_PARTS


def refine_foundationpose(key, oid, mesh, scene_dict, scorer, seed_cost, cfg):
    """
    FoundationPose's RGB-D refiner, seeded with the winning candidate.

    The refiner is render-and-compare already, which is why it is preferred
    over the pattern search. register() is never called: that is the frame-0
    global rotation grid which flipped the bowl in OPE-196, and the whole point
    of the candidate search is to decide the orientation ourselves.
    """
    import torch

    parts = _foundationpose_parts()
    from estimater import FoundationPose

    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    frame = scorer.frames[0]
    K = np.asarray(scene_dict["intrinsics"], dtype=np.float64)
    rgb = np.asarray(scene_dict["images"][frame], dtype=np.uint8)
    depth = np.asarray(scene_dict["depths"][frame], dtype=np.float64)

    plain = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64),
                            faces=np.asarray(mesh.faces), process=False)
    # debug_dir has to be given: FoundationPose defaults it to the home
    # directory of its own author, which does not exist here
    debug_dir = Path.cwd() / f"outputs/{key}/reconstruction/debug/pose_search"
    debug_dir.mkdir(parents=True, exist_ok=True)
    estimator = FoundationPose(
        model_pts=plain.vertices,
        model_normals=plain.vertex_normals,
        mesh=plain,
        scorer=parts["scorer"],
        refiner=parts["refiner"],
        glctx=parts["glctx"],
        debug_dir=str(debug_dir),
        debug=0,
    )
    # the mesh is already in the camera frame, so the identity is the seed; the
    # refiner works against the centred copy it built, hence the inverse
    to_centered = estimator.get_tf_to_centered_mesh().cpu().numpy()
    estimator.pose_last = torch.as_tensor(
        np.linalg.inv(to_centered), device="cuda", dtype=torch.float)
    # track_one already returns the transform of the mesh it was handed, not of
    # the centred copy, so no further composition with to_centered is needed
    pose = estimator.track_one(rgb=rgb, depth=depth, K=K,
                               iteration=int(cfg["fdpose_est_refine_iter"]))
    delta = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    inside, angle, shift = within_refinement_limits(
        delta, diagonal, float(cfg["pose_refine_max_rotation_deg"]),
        float(cfg["pose_refine_max_translation_fraction"]))
    cost, summary, _per_frame = scorer.score(mesh, delta)
    accepted = inside and cost < seed_cost
    print(f"[Info] [{key}] object {oid} FoundationPose refinement moved {angle:.2f} deg and "
          f"{1000.0 * shift:.1f} mm, cost {seed_cost:.4f} -> {cost:.4f} "
          f"(IoU {summary['iou']:.3f}, colour {summary['colour']:.4f}, depth {summary['depth']:.4f}), "
          f"{'kept' if accepted else 'rejected'}"
          f"{'' if inside else ' (past the cap)'}")
    if not accepted:
        delta, cost = np.eye(4), seed_cost
    return delta, cost, {"mode": "fdpose", "rotation_deg": angle,
                         "translation_mm": float(1000.0 * shift),
                         "cost_before": float(seed_cost), "cost_after": float(cost),
                         "accepted": bool(accepted), "within_cap": bool(inside),
                         "silhouette_iou": summary["iou"], "colour": summary["colour"],
                         "depth": summary["depth"]}


def refine_pose(key, oid, mesh, scene_dict, scorer, seed_cost, cfg):
    """Local refinement of the winning candidate, by whichever mode is configured."""
    mode = cfg["pose_refine_mode"]
    if mode == "off":
        return np.eye(4), seed_cost, {"mode": "off"}
    if mode == "fdpose":
        try:
            return refine_foundationpose(key, oid, mesh, scene_dict, scorer, seed_cost, cfg)
        except Exception as error:  # noqa: BLE001 - any failure falls back, never kills the run
            print(f"[Warn] [{key}] object {oid} FoundationPose refinement unavailable "
                  f"({type(error).__name__}: {error}), using the pattern search")
    return refine_photometric(key, oid, mesh, scorer, seed_cost, cfg)


# ─────────────────────────────── resting ───────────────────────────────

def support_face_normal(mesh, normal):
    """
    Outward normal of the face of the oriented bounding box that faces the
    plane, which is the face the object should be standing on.
    """
    try:
        to_origin, _extents = trimesh.bounds.oriented_bounds(mesh)
        axes = np.asarray(to_origin[:3, :3], dtype=np.float64).T
    except Exception:
        axes = np.eye(3)
    candidates = [axes[:, index] * sign for index in range(3) for sign in (1.0, -1.0)]
    # the support face points away from the plane normal, so most anti-aligned
    return min(candidates, key=lambda axis: float(np.dot(axis, normal) / max(np.linalg.norm(axis), 1e-9)))


def plane_gap(mesh, point, normal):
    """How far the lowest vertex sits above the plane, and the extent along the normal."""
    along = np.asarray(mesh.vertices, dtype=np.float64) @ normal
    gap = float(along.min() - np.asarray(point, dtype=np.float64) @ normal)
    return gap, float(along.max() - along.min())


def rest_on_plane(mesh, point, normal, max_rotation_deg=45.0, max_height_fraction=0.2,
                  lift_below_plane=True):
    """
    Put the object flat on the plane and translate it to contact.

    Whether the object belongs on the plane at all is decided first, from the
    gap it starts with: past max_height_fraction of its height it is a stacked
    or hanging object and nothing is done to it. That test has to come before
    the levelling rotation, because levelling a tilted object lifts its lowest
    vertex and would otherwise disqualify the very objects this step exists for.

    Then two steps, both reported: the smallest rotation that turns the support
    face of the oriented bounding box parallel to the plane, dropped when it
    asks for more than the cap, and the translation along the normal that
    closes the remaining gap. Both are proper rigid motions, so the determinant
    of the chain keeps its sign.

    An object already under the plane is a separate case: nothing rests inside
    a table, so lift_below_plane lets that one be raised however deep it sank.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    support = support_face_normal(mesh, normal)
    support = support / np.linalg.norm(support)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(support, -normal), -1.0, 1.0))))
    gap, height = plane_gap(mesh, point, normal)
    report = {"support_angle_before_deg": angle, "rotated_deg": 0.0,
              "support_angle_after_deg": angle, "rested": False, "moved_mm": 0.0,
              "clearance_mm": 1000.0 * gap, "gap_before_mm": 1000.0 * gap,
              "height_mm": 1000.0 * height, "skipped": False}
    transform = np.eye(4)

    limit = max_height_fraction * max(height, 1e-9)
    too_high = gap > limit
    too_deep = gap < -limit and not lift_below_plane
    if too_high or too_deep:
        print(f"[Info] object starts {1000.0 * gap:.1f} mm from the plane, more than "
              f"{max_height_fraction:.0%} of its {1000.0 * height:.1f} mm height, so it is "
              f"left where it is")
        report["skipped"] = True
        return transform, report

    if angle > 1e-3 and angle <= max_rotation_deg:
        axis = np.cross(support, -normal)
        if np.linalg.norm(axis) > 1e-9:
            centre = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
            R = trimesh.transformations.rotation_matrix(
                np.radians(angle), axis / np.linalg.norm(axis))[:3, :3]
            transform = rotation_about(R, centre)
            mesh.apply_transform(transform)
            report["rotated_deg"] = angle
            report["support_angle_after_deg"] = 0.0
    elif angle > max_rotation_deg:
        print(f"[Info] the support face is {angle:.2f} deg off the plane, past the "
              f"{max_rotation_deg:.0f} deg cap, so it is not levelled")

    gap, height = plane_gap(mesh, point, normal)
    drop = np.eye(4)
    drop[:3, 3] = -gap * normal
    mesh.apply_transform(drop)
    transform = drop @ transform
    report["rested"] = report["support_angle_after_deg"] <= 1e-3
    report["clearance_mm"] = 0.0
    report["moved_mm"] = 1000.0 * abs(gap)
    print(f"[Info] object rested: support face turned {report['rotated_deg']:.2f} deg, "
          f"then moved {1000.0 * abs(gap):.1f} mm {'up' if gap < 0 else 'down'} onto the plane "
          f"(height {1000.0 * height:.1f} mm, support face now "
          f"{report['support_angle_after_deg']:.2f} deg off the plane)")
    return transform, report


# ─────────────────────────────── entry point ───────────────────────────────

def search_object_pose(key, scene_dict, obj_info, mesh, cfg):
    """
    Replace the seed orientation with the best scoring candidate, refined.

    Takes the mesh as the placement left it, in the frame-0 camera frame, and
    returns the mesh moved onto the winning pose together with the 4x4 delta
    that was applied and the record for the report.
    """
    oid = obj_info["oid"]
    normal = np.asarray(scene_dict["info"]["groundplane_in_cam"]["normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    best, table, best_index, scorer = choose_candidate(
        key, scene_dict, obj_info, mesh, cfg, normal)
    try:
        delta_refine, cost, refinement = refine_pose(
            key, oid, mesh, scene_dict, scorer, table[best_index]["cost"], cfg)
    finally:
        scorer.close()
    delta = delta_refine @ best
    determinant = float(np.linalg.det(delta[:3, :3]))
    assert determinant > 0.0, (
        f"object {oid} candidate search produced a reflection, determinant {determinant}")
    placed = mesh.copy()
    placed.apply_transform(delta)
    record = {
        "candidates": [{k: v for k, v in entry.items() if k != "transform"} for entry in table],
        "chosen": table[best_index]["name"],
        "chosen_cost": table[best_index]["cost"],
        "refined_cost": float(cost),
        "refinement": refinement,
        "frames": list(scorer.frames),
        "static_until_frame": int(scorer.static_until),
    }
    return placed, delta, record
