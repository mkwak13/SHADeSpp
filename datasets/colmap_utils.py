"""Reader for COLMAP sparse reconstructions (cameras/images/points3D).

Supports both COLMAP's binary (.bin) and text (.txt) export formats. This is a
self-contained reimplementation of the format documented at
https://colmap.github.io/format.html (no dependency on pycolmap).

Coordinate convention (matches COLMAP): each Image stores a world-to-camera
transform as a quaternion `qvec` (w, x, y, z) and translation `tvec`, such
that for a 3D point in world coordinates X_world:

    X_cam = R(qvec) @ X_world + tvec

`X_cam[2]` is the point's depth in that camera.
"""
from __future__ import absolute_import, division, print_function

import os
import struct
import collections

import cv2
import numpy as np


Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


# model_id -> (model_name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_NAME_TO_NUM_PARAMS = {name: n for name, n in CAMERA_MODELS.values()}


def qvec2rotmat(qvec):
    """(w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2],
    ])


def camera_intrinsics_matrix(camera):
    """Build the 3x3 pixel-space intrinsics matrix K for a Camera.

    Handles the camera models COLMAP commonly produces (PINHOLE family,
    (SIMPLE_)RADIAL, OPENCV, FOV, *_FISHEYE). Distortion params are ignored
    here (we only need the linear projection for reprojecting already
    triangulated/undistorted-ish sparse points) -- if you need to account for
    lens distortion explicitly, extend this using `camera.params` and
    `camera.model`.
    """
    p = camera.params
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif camera.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                           "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif camera.model == "RADIAL" or camera.model == "RADIAL_FISHEYE":
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif camera.model == "FOV":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise NotImplementedError(
            "Unsupported COLMAP camera model '{}'; add it to "
            "camera_intrinsics_matrix() in datasets/colmap_utils.py".format(camera.model))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_flag="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_flag + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height,
                params=np.array(params, dtype=np.float64))
    return cameras


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model_name = elems[1]
            width, height = int(elems[2]), int(elems[3])
            params = np.array(list(map(float, elems[4:])), dtype=np.float64)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=params)
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = _read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            name = ""
            next_char = _read_next_bytes(fid, 1, "c")[0]
            while next_char != b"\x00":
                name += next_char.decode("utf-8")
                next_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            xy_id_data = _read_next_bytes(
                fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([
                tuple(map(float, xy_id_data[0::3])),
                tuple(map(float, xy_id_data[1::3]))])
            point3D_ids = np.array(tuple(map(int, xy_id_data[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        lines = [l.strip() for l in fid if l.strip() and not l.strip().startswith("#")]
    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        image_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        camera_id = int(elems[8])
        name = elems[9]

        points_elems = lines[i + 1].split()
        xys = np.column_stack([
            list(map(float, points_elems[0::3])),
            list(map(float, points_elems[1::3]))]) if points_elems else np.zeros((0, 2))
        point3D_ids = np.array(list(map(int, points_elems[2::3]))) if points_elems else \
            np.zeros((0,), dtype=int)

        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name,
            xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = _read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            track_elems = _read_next_bytes(
                fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D


def read_points3D_text(path):
    points3D = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            point3D_id = int(elems[0])
            xyz = np.array(list(map(float, elems[1:4])))
            rgb = np.array(list(map(int, elems[4:7])))
            error = float(elems[7])
            track = elems[8:]
            image_ids = np.array(list(map(int, track[0::2]))) if track else np.zeros((0,), dtype=int)
            point2D_idxs = np.array(list(map(int, track[1::2]))) if track else np.zeros((0,), dtype=int)
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D


def detect_model_format(path):
    """Return '.bin' or '.txt' depending on which files exist in `path`."""
    if all(os.path.exists(os.path.join(path, f + ".bin"))
           for f in ("cameras", "images", "points3D")):
        return ".bin"
    if all(os.path.exists(os.path.join(path, f + ".txt"))
           for f in ("cameras", "images", "points3D")):
        return ".txt"
    raise FileNotFoundError(
        "Could not find a full set of cameras/images/points3D.{{bin,txt}} in {}".format(path))


def read_model(path, ext=None):
    """Read a COLMAP sparse model directory. Returns (cameras, images, points3D).

    Args:
        path: directory containing cameras/images/points3D.bin or .txt
        ext: force ".bin" or ".txt"; if None, auto-detected from the directory
    """
    ext = ext or detect_model_format(path)
    if ext == ".bin":
        cameras = read_cameras_binary(os.path.join(path, "cameras.bin"))
        images = read_images_binary(os.path.join(path, "images.bin"))
        points3D = read_points3D_binary(os.path.join(path, "points3D.bin"))
    elif ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras.txt"))
        images = read_images_text(os.path.join(path, "images.txt"))
        points3D = read_points3D_text(os.path.join(path, "points3D.txt"))
    else:
        raise ValueError("ext must be '.bin' or '.txt', got {!r}".format(ext))
    return cameras, images, points3D


def project_points_to_image(image, camera, points3D, min_depth=1e-6):
    """Reproject the sparse 3D points visible in `image` into pixel space.

    Uses the *triangulated* 3D points + the image's pose/intrinsics -- i.e. a
    fresh projection, not simply the raw detected keypoint locations stored
    in `image.xys` (though for a converged COLMAP solve the two should be
    almost identical up to reprojection error).

    Returns:
        pixels: (N, 2) array of (u, v) reprojected pixel coordinates
        depths: (N,) array of camera-space depth (Z) for each point
        point3D_ids: (N,) array of the corresponding point3D ids
    """
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    K = camera_intrinsics_matrix(camera)

    valid_ids = image.point3D_ids[image.point3D_ids != -1]
    if valid_ids.size == 0:
        return (np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=int))

    xyz_world = np.stack([points3D[pid].xyz for pid in valid_ids], axis=0)  # (N, 3)
    xyz_cam = (R @ xyz_world.T).T + t[None, :]  # (N, 3)
    depths = xyz_cam[:, 2]

    in_front = depths > min_depth
    xyz_cam = xyz_cam[in_front]
    depths = depths[in_front]
    valid_ids = valid_ids[in_front]

    uvw = (K @ xyz_cam.T).T
    pixels = uvw[:, :2] / uvw[:, 2:3]

    return pixels, depths, valid_ids


def observed_points_with_depth(image, points3D, min_depth=1e-6):
    """Sparse (pixel, depth) pairs for the 3D points visible in `image`, using
    the *actual detected keypoint locations* (`image.xys`) rather than a
    fresh projection through the camera's intrinsics/distortion model.

    This is what you want whenever the camera has lens distortion (e.g.
    EndoMapper's OPENCV_FISHEYE cameras): `image.xys` already lives in real,
    distorted pixel coordinates matching the raw frame on disk, so sampling a
    predicted depth map at these pixels is correct without needing to
    implement the camera model's distortion ourselves. Depth itself never
    involves the intrinsics/distortion -- it's just the point's Z coordinate
    after transforming into camera space with (R, t) -- so no camera object
    is required here, unlike `project_points_to_image`.

    Returns:
        pixels: (N, 2) array of (u, v) pixel coordinates, as detected by COLMAP
        depths: (N,) array of camera-space depth (Z) for each point
        point3D_ids: (N,) array of the corresponding point3D ids
    """
    R = qvec2rotmat(image.qvec)
    t = image.tvec

    valid = image.point3D_ids != -1
    if not valid.any():
        return (np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=int))

    pixels = image.xys[valid]
    point3D_ids = image.point3D_ids[valid]
    xyz_world = np.stack([points3D[pid].xyz for pid in point3D_ids], axis=0)
    depths = ((R @ xyz_world.T).T + t[None, :])[:, 2]

    in_front = depths > min_depth
    return pixels[in_front], depths[in_front], point3D_ids[in_front]


def get_fisheye_distortion_coeffs(camera):
    """Return the (k1, k2, k3, k4) OpenCV fisheye distortion coefficients for
    `camera`, or None if its model isn't fisheye-distorted (nothing to undo).
    """
    if camera.model == "OPENCV_FISHEYE":
        return np.array(camera.params[4:8], dtype=np.float64)
    return None


def build_undistort_map(camera, balance=0.0):
    """Precompute a cv2.fisheye undistortion remap for `camera`.

    `balance` trades off field of view vs. invalid (black) border pixels:
    0.0 keeps only the pixels valid in the original distorted image (no
    black borders, some FOV cropped at the edges); 1.0 keeps the full
    original FOV at the cost of black corners. 0.0 is the safer default
    here since the SHADeS/SHADeS++ encoder was trained on images with no
    black borders.

    Returns (map1, map2, new_K) for cv2.remap, or (None, None, K) if the
    camera has no fisheye distortion to undo (K unprojected as-is).
    """
    K = camera_intrinsics_matrix(camera)
    D = get_fisheye_distortion_coeffs(camera)
    if D is None:
        return None, None, K

    size = (camera.width, camera.height)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, size, cv2.CV_32FC1)
    return map1, map2, new_K


def undistort_image(image_bgr, map1, map2):
    """Apply a map from `build_undistort_map` to a BGR (cv2-convention) image."""
    return cv2.remap(image_bgr, map1, map2, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)


def undistort_points(pixels, camera, new_K):
    """Map (u, v) pixel coordinates from the original distorted image into the
    pixel space of the image `build_undistort_map`'s (map1, map2, new_K)
    produces -- i.e. the correct GT point locations to sample a depth map
    predicted on the undistorted image. No-op if the camera isn't fisheye.
    """
    D = get_fisheye_distortion_coeffs(camera)
    if D is None or pixels.shape[0] == 0:
        return pixels
    K = camera_intrinsics_matrix(camera)
    pts = pixels.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.fisheye.undistortPoints(pts, K, D, P=new_K)
    return undistorted.reshape(-1, 2)
