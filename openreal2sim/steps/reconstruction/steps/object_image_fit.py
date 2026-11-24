"""
Fit the placed object to the frame-0 image, by rendering it and comparing.

The placement upstream gives the object one uniform scale, taken either from
the SAM-3D layout or from the bounding box diagonal of the masked depth cloud,
and it keeps the aspect ratio of the generated mesh exactly. That aspect ratio
is the measured defect: against the pipeline's own depth map the objects come
out 1.45 to 4.32 times too tall for their footprint, flat things worst of all,
while their footprints match to within 15 per cent (OPE-225). A uniform scale
cannot repair that, and the silhouette pays for it: 0.44 to 0.81 intersection
over union against the mask where a per-object fit reached 0.81 to 0.97.

So after the uniform scale this step searches, per object, the five numbers a
single photograph can actually constrain:

  - a yaw about the ground normal, capped so it refines the orientation the
    candidate search chose instead of replacing it;
  - the uniform scale, as a bounded factor on the one already applied;
  - the placement in the ground plane, two numbers;
  - a thickness factor along the ground normal, at most 1 so it can only thin
    the object, and never short enough to put the object below the height its
    own depth cloud reaches. That floor is the whole guard on this axis: the
    depth term is monotonically happier the flatter the object gets, so a
    fixed factor floor would be the answer rather than the data (OPE-231).

with the lowest vertex held on the ground plane by construction, against

  (1 - IoU with the frame-0 mask) + weight * median |rendered depth - fg_depth|

read over the mask, charging a fixed miss where the render does not reach it,
so that shrinking the object onto the patch the depth map happens to agree
with is not a way to win. Everything happens in the frame-0 camera frame,
where the mask, the depth map and the intrinsics live, so `register_scene_meshes` can
apply the gravity alignment and the shift to the origin unchanged.

Two guards keep a depth map that is not measuring the object from deciding
its shape. The thickness axis is skipped when the depth residual over the
mask saturates the miss charge, which is what a transparent bottle does by
reading the table behind it, and when thinning to the floor does not lower
the residual at all. And no fit is kept that ends with less silhouette than
the placement it started from, whatever it does to the cost.

The result is baked into the vertices, like every other transform here, and it
runs before FoundationPose, which takes the registered mesh as its model: the
thinner mesh is what gets tracked and what reaches `*_optimized.glb`, and with
`fdpose_seed_from_registration` and `fdpose_placement_from_registration` the
fitted pose is also what FoundationPose starts from and what the export keeps.

The transform this step returns carries an anisotropic scale, so it is not a
rigid motion; all three factors are positive, so the determinant of
`object_to_world` stays positive like the assertions downstream expect.
"""

import os

import numpy as np
import trimesh

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from openreal2sim.steps.reconstruction.steps.pose_candidate_search import (
    OPENCV_TO_OPENGL,
    _pinhole_camera_class,
)

# Metres charged for a pixel of the mask the render does not cover, and the
# cap on any single residual. Without a charge for the misses the depth term
# would be read over the overlap alone, and then shrinking the object until
# only the best agreeing patch of it is left would lower the cost: that is
# exactly what a transparent bottle, whose depth map reads the table behind
# it, talks the fit into doing.
MISSED_DEPTH_ERROR = 0.1

# How much silhouette a fit may lose against the placement it started from
# before it is thrown away. Zero to the rounding of one pixel of a small mask:
# the fit exists to raise the overlap, so losing it is a failure, not a trade.
SILHOUETTE_TOLERANCE = 1e-4

# Weight of every bound violation. The silhouette term is at most 1, so a
# tenth outside a bound already costs more than any gain in overlap.
PENALTY = 5.0


# ─────────────────────────────── geometry ───────────────────────────────

def plane_basis(normal):
    """
    Orthonormal frame of the ground plane, as the rows (e1, e2, n) of a 3x3.

    e1 and e2 span the plane, n is its normal, which the reconstruction keeps
    pointing up out of the table. Multiplying a point by this matrix gives its
    coordinates in that frame, so the third one is the height above the plane.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.stack([e1, e2, n])


def hull_diameter(points):
    """
    Largest distance between two of the given 2D points, over their convex
    hull. Same measure the OPE-225 tables used for the footprint, so the
    thickness ratios printed here can be read against them.
    """
    points = np.asarray(points, dtype=np.float64)
    try:
        from scipy.spatial import ConvexHull

        points = points[ConvexHull(points).vertices]
    except Exception:
        pass
    if len(points) > 500:
        step = int(np.ceil(len(points) / 500.0))
        points = points[::step]
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(gaps.max()) if len(points) > 1 else 0.0


def plane_frame_measure(points, plane_offset, percentile=100.0):
    """
    Height above the plane, in-plane hull diameter, and their ratio, for points
    already given in the plane frame.

    The ratio is scale free, which is what makes a mesh comparable with a depth
    cloud that may disagree with it about absolute size. The percentile is
    there for the cloud: its top is a few outlier points, so the tables in
    OPE-225 read its height at the 99th percentile.
    """
    points = np.asarray(points, dtype=np.float64)
    height = float(np.percentile(points[:, 2], percentile) - plane_offset)
    diameter = hull_diameter(points[:, :2])
    return {
        "height_m": height,
        "diameter_m": diameter,
        "height_over_diameter": height / max(diameter, 1e-9),
    }


def shape_measure(points, basis, plane_offset, percentile=100.0):
    """The same measure for points given in the camera frame."""
    return plane_frame_measure(np.asarray(points, dtype=np.float64) @ basis.T,
                               plane_offset, percentile)


# ─────────────────────────────── rendering ───────────────────────────────

class FitRenderer:
    """
    Depth renderer for one object against the frame-0 camera.

    The mesh is handed over once and every candidate is a node pose, so the
    per-iteration cost is one rasterisation. Only depth is rendered: the
    silhouette is where it is non zero, and no texture or light has to be set
    up for that.

    The scene frame is the ground plane frame, not the camera frame, and the
    rotation between them is folded into the pose of the camera. In the plane
    frame the fit is a yaw about the third axis composed with
    diag(s, s, s * thickness), which is diagonal in the two parameters that
    are searched and makes the resting constraint one coordinate of one row.

    It also keeps the node pose renderable. pyrender stores a node pose as a
    quaternion, three scales and a translation, which reproduces a matrix only
    when its columns are orthogonal. The plane frame matrix above has
    orthogonal columns; the same transform written in the camera frame does
    not, and rendering it as a node pose disagrees with rendering the baked
    vertices. Measured with `h5py/pyrender_check2.py`, identically on the
    0.1.17 of the reconstruction image and the 0.1.45 of the eval venv:
    silhouette IoU 1.0000 in the plane frame, 0.8974 for the same fit written
    in the camera frame, 0.7315 for a general oblique matrix.
    """

    def __init__(self, intrinsics, width, height, vertices, faces, basis):
        import pyrender

        self._pyrender = pyrender
        K = np.asarray(intrinsics, dtype=np.float64)
        camera_class = _pinhole_camera_class()
        # abs: one scene has a negative stored focal length, see OPE-193.
        # The near and far planes are tight around a table top scene, which is
        # what keeps the depth buffer accurate enough to read in millimetres.
        self.camera = camera_class(abs(K[0, 0]), abs(K[1, 1]), K[0, 2], K[1, 2],
                                   znear=0.05, zfar=10.0)
        # rows of the basis are the plane axes in camera coordinates, so the
        # matrix itself takes a camera point to the plane frame
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = np.asarray(basis, dtype=np.float64)
        local = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                                faces=np.asarray(faces), process=False)
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        self.scene.add(self.camera, pose=camera_pose @ OPENCV_TO_OPENGL)
        self.node = self.scene.add(pyrender.Mesh.from_trimesh(local, smooth=False))
        self.renderer = pyrender.OffscreenRenderer(int(width), int(height))

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

    def depth(self, placement):
        """Depth map of the mesh placed by the given 4x4, in metres."""
        self.scene.set_pose(self.node, placement)
        return self.renderer.render(
            self.scene, flags=self._pyrender.RenderFlags.DEPTH_ONLY)


# ─────────────────────────────── the fit ───────────────────────────────

class ObjectImageFit:
    """
    The five parameter fit of one object against frame 0.

    The parameters are packed as [yaw, log scale, in-plane x, in-plane y] plus
    [log thickness] when the thickness axis is fitted, so the identity is the
    all-zero vector and both scales stay positive whatever the optimiser does.
    """

    def __init__(self, scene_dict, obj_info, mesh, cloud, cfg):
        self.oid = obj_info["oid"]
        self.depth_weight = float(cfg["image_fit_depth_weight"])
        self.thickness_max = float(cfg["image_fit_thickness_max"])
        self.scale_bound = float(cfg["image_fit_scale_max_factor"])
        self.yaw_bound = np.radians(float(cfg["image_fit_yaw_max_deg"]))
        self.shift_bound = (float(cfg["image_fit_shift_max_fraction"])
                            * float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])))
        self.iterations = int(cfg["image_fit_max_iterations"])

        plane = scene_dict["info"]["groundplane_in_cam"]
        self.basis = plane_basis(plane["normal"])
        self.plane_offset = float(np.asarray(plane["point"], dtype=np.float64)
                                  @ self.basis[2])

        # the frame-0 mask the object mesh was generated from, and the frame-0
        # depth map the point cloud was back projected from
        self.mask = np.asarray(scene_dict["mask"][0][self.oid]["mask"], dtype=bool)
        self.observed_depth = np.asarray(scene_dict["recon"]["fg_depth"], dtype=np.float64)
        self.depth_valid = np.isfinite(self.observed_depth) & (self.observed_depth > 1e-6)

        self.cloud = np.asarray(cloud, dtype=np.float64)
        # the mesh in the plane frame, centred on its centroid: the fit is
        # diagonal there, and so is the constraint that it rests on the plane
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.centre = vertices.mean(axis=0)
        self.local = (vertices - self.centre) @ self.basis.T
        self.centre_local = self.basis @ self.centre
        self.renderer = FitRenderer(scene_dict["intrinsics"],
                                    scene_dict["width"], scene_dict["height"],
                                    self.local, mesh.faces, self.basis)

        # how tall the object is now, and how short the depth map lets it get
        self.start_height = float(self.local[:, 2].max() - self.local[:, 2].min())
        self.height_floor = self._height_floor(
            float(cfg["image_fit_height_floor_percentile"]),
            float(cfg["image_fit_plane_noise_m"]))
        self.thickness_floor = float(np.clip(
            self.height_floor / max(self.start_height, 1e-9), 1e-3, self.thickness_max))

        # the fit may not end below the silhouette it started from, so the
        # overlap of the placement it was handed is measured first, with no
        # guard of its own
        self.iou_floor = None
        self.start_terms = self.measure(np.zeros(4))
        self.iou_floor = self.start_terms["silhouette_iou"]

        self.fit_thickness, self.thickness_note = self._thickness_decision(
            float(cfg["image_fit_depth_saturation"]))

    def close(self):
        self.renderer.close()

    # ---- parameters

    def _height_floor(self, percentile, plane_noise):
        """
        How short the object may be made, in metres, from the depth map alone.

        The depth term is minimised by a flat object: the thinner the mesh, the
        closer its visible surface to the single surface the depth map has, so
        with a fixed floor it is the floor that sets the thickness, not the
        data (OPE-231: with the floor at 0.25 the keychain lands on it, and at
        0.02 the fit takes it to 0.8 mm at a lower cost). The depth cloud of
        the object does know how tall it is: its height above the plane, read
        at a high percentile so a few outliers do not set it, and never below
        the noise of the plane fit itself, which is the thinnest thing that
        could be measured at all.
        """
        heights = (self.cloud @ self.basis[2]) - self.plane_offset
        if len(heights) == 0:
            return plane_noise
        return max(float(np.percentile(heights, percentile)), plane_noise)

    def _thickness_decision(self, saturation):
        """
        Whether the thickness axis may be fitted for this object.

        Two ways it must not be. A depth map that saturates the miss charge
        over the whole mask is not measuring the object at all, which is what
        a transparent bottle does by reading the table behind it; thinning
        then trades a silhouette the fit cannot check against a residual that
        means nothing. And a depth map that does not improve when the object
        is thinned to its floor is saying the object is not too thick, so
        there is nothing to fit.
        """
        start_error = self.start_terms["depth_error_m"]
        if start_error >= saturation * MISSED_DEPTH_ERROR:
            return False, (f"the depth map saturates over the mask "
                           f"({1000.0 * start_error:.0f} mm median residual, charge "
                           f"{1000.0 * MISSED_DEPTH_ERROR:.0f} mm), so it cannot say how "
                           f"thick the object is")
        probe = np.array([0.0, 0.0, 0.0, 0.0, float(np.log(self.thickness_floor))])
        probe_error = self.measure(probe)["depth_error_m"]
        if probe_error >= start_error:
            return False, (f"thinning to the {1000.0 * self.height_floor:.1f} mm the depth "
                           f"cloud allows does not lower the depth residual "
                           f"({1000.0 * start_error:.1f} -> {1000.0 * probe_error:.1f} mm)")
        return True, None

    def placement(self, params):
        """
        The 4x4 pose of the centred mesh in the plane frame, which is the frame
        the renderer works in.

        The linear part is diag(scale, scale, scale * thickness) composed with
        the yaw about the plane normal, which is the third axis here, so the
        two commute and the columns stay orthogonal. The height is not a free
        parameter: it is whatever puts the lowest vertex on the plane.
        """
        yaw, log_scale, shift_x, shift_y = params[:4]
        scale = float(np.exp(log_scale))
        thickness = float(np.exp(params[4])) if len(params) > 4 else 1.0
        cos, sin = np.cos(float(yaw)), np.sin(float(yaw))
        linear = np.array([[scale * cos, -scale * sin, 0.0],
                           [scale * sin, scale * cos, 0.0],
                           [0.0, 0.0, scale * thickness]])
        height = self.centre_local[2] + scale * thickness * self.local[:, 2]
        drop = self.plane_offset - float(height.min())
        placement = np.eye(4)
        placement[:3, :3] = linear
        placement[:3, 3] = self.centre_local + np.array([float(shift_x), float(shift_y), drop])
        return placement

    def transform(self, params):
        """The same fit, as a transform of the original camera frame vertices."""
        placement = self.placement(params)
        linear = self.basis.T @ placement[:3, :3] @ self.basis
        transform = np.eye(4)
        transform[:3, :3] = linear
        transform[:3, 3] = self.basis.T @ placement[:3, 3] - linear @ self.centre
        return transform

    def bounds_hit(self, params, tolerance=0.02):
        """
        Which bounds the given parameters are sitting on, for the record.

        The tolerance is two per cent because a simplex walking into a
        penalised boundary stops next to it, not on it: the keychain settles
        0.1 mm above its 8.1 mm floor, and that is a bound hit, not a fit.
        """
        hits = []
        if abs(float(params[0])) >= self.yaw_bound * (1.0 - tolerance):
            hits.append("yaw")
        if abs(float(params[1])) >= np.log(self.scale_bound) * (1.0 - tolerance):
            hits.append("scale")
        if float(np.hypot(params[2], params[3])) >= self.shift_bound * (1.0 - tolerance):
            hits.append("shift")
        if len(params) > 4:
            thickness = float(np.exp(params[4]))
            height = thickness * float(np.exp(params[1])) * self.start_height
            if height <= self.height_floor * (1.0 + tolerance):
                hits.append("height_floor")
            if thickness >= self.thickness_max * (1.0 - tolerance):
                hits.append("thickness_max")
        return hits

    def penalty(self, params):
        """
        How far the parameters sit outside their bounds, in cost units.

        The thickness is bounded by a height in metres rather than by a factor:
        the depth term always prefers a flatter object, so whatever the factor
        floor is, that is the answer it gives. The height the depth cloud of
        this object reaches is the floor that has something to do with the
        object (see `_height_floor`), and it is read against the height the
        fit actually produces, which the uniform scale also moves.
        """
        yaw, log_scale, shift_x, shift_y = params[:4]
        outside = max(0.0, abs(float(yaw)) - self.yaw_bound) / max(self.yaw_bound, 1e-9)
        outside += max(0.0, abs(float(log_scale)) - np.log(self.scale_bound)) / np.log(self.scale_bound)
        shift = float(np.hypot(shift_x, shift_y))
        outside += max(0.0, shift - self.shift_bound) / max(self.shift_bound, 1e-9)
        if len(params) > 4:
            thickness = float(np.exp(params[4]))
            outside += max(0.0, thickness - self.thickness_max)
            height = thickness * float(np.exp(log_scale)) * self.start_height
            outside += max(0.0, self.height_floor - height) / max(self.height_floor, 1e-9)
        return PENALTY * outside

    # ---- objective

    def measure(self, params):
        """
        Silhouette overlap and depth residual of one parameter vector, with no
        bound or guard terms.

        The depth residual is read over the mask, not over the overlap: every
        pixel of the mask the depth map can speak for is charged, either with
        what the render disagrees by there or, where the render does not reach,
        with the fixed miss. Residuals are capped at the same value, so one
        badly reconstructed corner cannot outvote the rest.
        """
        depth = self.renderer.depth(self.placement(params))
        silhouette = depth > 1e-6
        union = np.logical_or(silhouette, self.mask).sum()
        overlap = np.logical_and(silhouette, self.mask)
        iou = 0.0 if union == 0 else float(overlap.sum() / union)
        measurable = self.mask & self.depth_valid
        if measurable.sum() >= 25:
            residual = np.full(int(measurable.sum()), MISSED_DEPTH_ERROR)
            covered = silhouette[measurable]
            residual[covered] = np.abs(
                depth[measurable][covered] - self.observed_depth[measurable][covered])
            error = float(np.median(np.minimum(residual, MISSED_DEPTH_ERROR)))
        else:
            error = MISSED_DEPTH_ERROR
        return {"silhouette_iou": iou, "depth_error_m": error}

    def score(self, params):
        """
        Cost of one parameter vector, with the terms that make it up.

        On top of the two measured terms the fit pays for leaving its bounds
        and for any silhouette it loses against the placement it was handed.
        That last one is a guard, not a preference: the depth map of a badly
        reconstructed object can be worth more than the mask to the sum, and
        then a fit that is visibly worse still wins on cost (OPE-231).
        """
        terms = self.measure(params)
        cost = (1.0 - terms["silhouette_iou"]) + self.depth_weight * terms["depth_error_m"]
        cost += self.penalty(params)
        if self.iou_floor is not None:
            cost += PENALTY * max(0.0, self.iou_floor - terms["silhouette_iou"])
        terms["cost"] = cost
        return cost, terms

    def start(self):
        """The all-zero parameter vector: the placement exactly as it came in."""
        return np.zeros(5 if self.fit_thickness else 4)

    def coarse(self):
        """
        Scored grid over yaw, the uniform scale and the thickness.

        Deliberately coarse and deliberately not over the placement: the
        placement is already on the right ray, and a grid over five axes would
        not be. The simplex refinement moves it.
        """
        yaws = np.radians(np.arange(-np.degrees(self.yaw_bound),
                                    np.degrees(self.yaw_bound) + 1e-6, 15.0))
        if len(yaws) == 0:
            yaws = np.array([0.0])
        scales = [factor for factor in (0.8, 0.9, 1.0, 1.1, 1.25)
                  if factor <= self.scale_bound and factor >= 1.0 / self.scale_bound]
        low, high = self.thickness_floor, self.thickness_max
        thicknesses = [1.0]
        if self.fit_thickness:
            thicknesses = sorted({float(np.clip(value, low, high))
                                  for value in (high, 0.5 * (low + high), low)})
        grid = []
        for yaw in yaws:
            for scale in scales:
                for thickness in thicknesses:
                    params = [float(yaw), float(np.log(scale)), 0.0, 0.0]
                    if self.fit_thickness:
                        params.append(float(np.log(thickness)))
                    cost, _terms = self.score(np.array(params))
                    grid.append((cost, np.array(params)))
        grid.sort(key=lambda entry: entry[0])
        return grid

    def refine(self, params):
        """Nelder-Mead on the same cost, from one starting vector."""
        from scipy.optimize import minimize

        steps = [0.15, 0.1, 0.01, 0.01, 0.3][:len(params)]
        simplex = [params] + [
            [value + (steps[axis] if axis == index else 0.0)
             for axis, value in enumerate(params)]
            for index in range(len(params))
        ]
        result = minimize(lambda vector: self.score(vector)[0], params,
                          method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-5,
                                   "maxiter": self.iterations,
                                   "initial_simplex": np.array(simplex, dtype=np.float64)})
        return result.x, float(result.fun)

    # ---- reporting

    def report(self, params):
        """Silhouette, depth error and thickness ratio of one parameter vector."""
        _cost, terms = self.score(params)
        placement = self.placement(params)
        vertices = self.local @ placement[:3, :3].T + placement[:3, 3]
        mesh_shape = plane_frame_measure(vertices, self.plane_offset)
        cloud_shape = shape_measure(self.cloud, self.basis, self.plane_offset,
                                    percentile=99.0)
        terms["height_m"] = mesh_shape["height_m"]
        terms["footprint_m"] = mesh_shape["diameter_m"]
        terms["thickness_ratio"] = (mesh_shape["height_over_diameter"]
                                    / max(cloud_shape["height_over_diameter"], 1e-9))
        terms["depth_height_over_diameter"] = cloud_shape["height_over_diameter"]
        return terms


def fit_object_to_image(key, scene_dict, obj_info, mesh, cloud, cfg):
    """
    Fit the placed object to frame 0 and return the moved mesh.

    Returns the mesh, the 4x4 that was applied to it, and the record for the
    report. A fit is kept only when it both lowers the cost and does not lose
    silhouette against the placement it started from, so this step cannot make
    the one thing it is measured on worse.
    """
    oid = obj_info["oid"]
    name = obj_info["name"]
    fit = ObjectImageFit(scene_dict, obj_info, mesh, cloud, cfg)
    try:
        before = fit.report(fit.start())
        if fit.thickness_note is not None:
            print(f"[Info] [{key}] object {oid}_{name} thickness axis not fitted "
                  f"({fit.thickness_note})")
        grid = fit.coarse()
        best_params, best_cost = fit.start(), before["cost"]
        for cost, params in grid[:3]:
            refined, refined_cost = fit.refine(params)
            if refined_cost < best_cost:
                best_params, best_cost = refined, refined_cost
        cheaper = best_cost < before["cost"] - 1e-6
        candidate = fit.report(best_params)
        keeps_silhouette = (candidate["silhouette_iou"]
                            >= before["silhouette_iou"] - SILHOUETTE_TOLERANCE)
        accepted = bool(cheaper and keeps_silhouette)
        if not accepted:
            reason = ("it costs no less than the placement" if not cheaper else
                      f"it loses silhouette ({before['silhouette_iou']:.3f} -> "
                      f"{candidate['silhouette_iou']:.3f})")
            print(f"[Info] [{key}] object {oid}_{name} image fit dropped: {reason}")
            best_params = fit.start()
        after = fit.report(best_params)
        transform = fit.transform(best_params)
        scale = float(np.exp(best_params[1]))
        thickness = float(np.exp(best_params[4])) if len(best_params) > 4 else 1.0
        yaw_deg = float(np.degrees(best_params[0]))
        shift_mm = 1000.0 * float(np.hypot(best_params[2], best_params[3]))
        record = {
            "accepted": bool(accepted),
            "scale": scale,
            "thickness_scale": thickness,
            "thickness_fitted": bool(fit.fit_thickness),
            "thickness_note": fit.thickness_note,
            "height_floor_mm": 1000.0 * fit.height_floor,
            "thickness_floor": fit.thickness_floor,
            "bounds_hit": fit.bounds_hit(best_params) if accepted else [],
            "yaw_deg": yaw_deg,
            "shift_mm": shift_mm,
            "iou_before": before["silhouette_iou"],
            "iou_after": after["silhouette_iou"],
            "depth_error_before_mm": 1000.0 * before["depth_error_m"],
            "depth_error_after_mm": 1000.0 * after["depth_error_m"],
            "thickness_ratio_before": before["thickness_ratio"],
            "thickness_ratio_after": after["thickness_ratio"],
            "height_before_mm": 1000.0 * before["height_m"],
            "height_after_mm": 1000.0 * after["height_m"],
            "footprint_before_mm": 1000.0 * before["footprint_m"],
            "footprint_after_mm": 1000.0 * after["footprint_m"],
            "cost_before": before["cost"],
            "cost_after": after["cost"],
        }
        hits = record["bounds_hit"]
        print(f"[Info] [{key}] object {oid}_{name} image fit: scale {scale:.4f}, thickness "
              f"{thickness:.3f} (floor {fit.thickness_floor:.3f} from a "
              f"{1000.0 * fit.height_floor:.1f} mm depth cloud), yaw {yaw_deg:.1f} deg, moved "
              f"{shift_mm:.1f} mm in the plane"
              f"{', on the ' + ', '.join(hits) + ' bound' if hits else ''}")
        print(f"[Info] [{key}] object {oid}_{name} silhouette IoU "
              f"{before['silhouette_iou']:.3f} -> {after['silhouette_iou']:.3f}, depth error "
              f"{1000.0 * before['depth_error_m']:.1f} -> {1000.0 * after['depth_error_m']:.1f} mm, "
              f"thickness ratio {before['thickness_ratio']:.2f} -> {after['thickness_ratio']:.2f} "
              f"(height {1000.0 * before['height_m']:.0f} -> {1000.0 * after['height_m']:.0f} mm "
              f"over a {1000.0 * after['footprint_m']:.0f} mm footprint)")
    finally:
        fit.close()

    determinant = float(np.linalg.det(transform[:3, :3]))
    assert determinant > 0.0, (
        f"object {oid} image fit produced a reflection, determinant {determinant}")
    placed = mesh.copy()
    placed.apply_transform(transform)
    return placed, transform, record
