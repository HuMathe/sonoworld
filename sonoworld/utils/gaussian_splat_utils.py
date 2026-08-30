from __future__ import annotations

import math
from pathlib import Path
from typing import Any


SH_C0 = 0.28209479177387814


def _import_rasterization() -> Any:
    try:
        from gsplat.rendering import rasterization
    except ImportError:
        try:
            from gsplat import rasterization
        except ImportError as exc:
            raise ImportError(
                "Rendering Gaussian splat depth requires the optional `gsplat` package."
            ) from exc
    return rasterization


def load_gsplat_scene_from_ply(
    ply_path: str | Path,
    device: str | Any = "cuda",
    force_rgb: bool = False,
    min_alpha: float = 1 / 255,
    max_scale: float = 5.0,
) -> dict[str, Any]:
    import numpy as np
    from plyfile import PlyData
    import torch

    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    names = vertex.data.dtype.names or ()

    means = torch.tensor(
        np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1),
        dtype=torch.float32,
        device=device,
    )
    scales = torch.tensor(
        np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1),
        dtype=torch.float32,
        device=device,
    ).exp()
    quats = torch.tensor(
        np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=-1),
        dtype=torch.float32,
        device=device,
    )
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opacities = torch.tensor(vertex["opacity"], dtype=torch.float32, device=device).sigmoid()

    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1)
    rest_names = sorted(
        [name for name in names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )

    if rest_names and not force_rgb:
        rest = np.stack([vertex[name] for name in rest_names], axis=-1)
        per_channel = rest.shape[1] // 3
        sh_degree = int(np.sqrt(per_channel + 1) - 1)
        rest = rest.reshape(len(vertex), 3, per_channel).transpose(0, 2, 1)
        colors = torch.tensor(
            np.concatenate([f_dc[:, None, :], rest], axis=1),
            dtype=torch.float32,
            device=device,
        )
    else:
        sh_degree = None
        colors = torch.tensor(0.5 + SH_C0 * f_dc, dtype=torch.float32, device=device)

    valid = (opacities >= min_alpha) & (scales.max(dim=-1).values <= max_scale)
    if valid.any() and valid.sum() < valid.numel():
        means = means[valid]
        scales = scales[valid]
        quats = quats[valid]
        opacities = opacities[valid]
        colors = colors[valid]

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
    }


def _resolve_device(device: str | None) -> Any:
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _marble_cubemap_viewmats(
    front: Any,
    right: Any,
    up: Any,
    device: Any,
) -> Any:
    import torch

    directions = torch.tensor(
        [front, -front, right, -right, up, -up],
        dtype=torch.float32,
        device=device,
    )
    up_vectors = torch.tensor(
        [up, up, up, up, -front, front],
        dtype=torch.float32,
        device=device,
    )
    right_vectors = torch.cross(directions, up_vectors, dim=1)

    viewmats = torch.stack(
        [
            torch.stack(
                [
                    right_vectors[i],
                    -up_vectors[i],
                    directions[i],
                    torch.zeros(3, dtype=torch.float32, device=device),
                ],
                dim=1,
            )
            for i in range(6)
        ],
        dim=0,
    )
    bottom = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 4)
    viewmats = torch.cat([viewmats, bottom.repeat(6, 1, 1)], dim=1)
    return torch.linalg.inv(viewmats)


def cubemap_to_equirect(cubemap: Any, width: int, height: int | None = None) -> Any:
    import torch

    height = height or width // 2
    _, face_h, face_w, channels = cubemap.shape
    if face_h != face_w:
        raise ValueError(f"Cubemap faces must be square, got {face_h}x{face_w}.")

    if channels == 4:
        u = torch.arange(face_w, device=cubemap.device).view(1, face_w).expand(face_w, face_w)
        v = torch.arange(face_w, device=cubemap.device).view(face_w, 1).expand(face_w, face_w)
        x = (u - face_w / 2) / (face_w / 2)
        y = (v - face_w / 2) / (face_w / 2)
        cubemap[..., 3] = cubemap[..., 3] * torch.sqrt(1 + x**2 + y**2).to(cubemap.device)

    thetas = torch.linspace(-math.pi, math.pi, width + 1, device=cubemap.device, dtype=torch.float32)[:-1]
    phis = torch.linspace(math.pi / 2, -math.pi / 2, height, device=cubemap.device, dtype=torch.float32)
    thetas, phis = torch.meshgrid(thetas, phis, indexing="xy")

    dirs = torch.stack(
        [
            torch.cos(phis) * torch.sin(thetas),
            torch.cos(phis) * torch.cos(thetas),
            torch.sin(phis),
        ],
        dim=-1,
    )

    up = torch.tensor([0, 0, 1], dtype=torch.float32, device=cubemap.device)
    right = torch.tensor([1, 0, 0], dtype=torch.float32, device=cubemap.device)
    front = torch.tensor([0, 1, 0], dtype=torch.float32, device=cubemap.device)
    face_dirs = torch.stack([front, -front, right, -right, up, -up], dim=0)
    face_indices = torch.argmax(torch.einsum("hwc,dc->hwd", dirs, face_dirs), dim=-1)

    cam_ups = torch.stack([up, up, up, up, -front, front], dim=0)
    cam_rights = torch.cross(face_dirs, cam_ups, dim=1)
    cams = torch.stack(
        [torch.stack([r, -u, f], dim=0) for r, u, f in zip(cam_rights, cam_ups, face_dirs)],
        dim=0,
    )

    uvd = torch.einsum("hwc,dic->dhwi", dirs, cams)
    us = (uvd[..., 0] / (uvd[..., 2] + 1e-8) + 1) * (face_w / 2)
    vs = (uvd[..., 1] / (uvd[..., 2] + 1e-8) + 1) * (face_w / 2)

    equirect = torch.zeros((height, width, channels), dtype=torch.float32, device=cubemap.device)
    for face_idx in range(6):
        mask = face_indices == face_idx
        if not mask.any():
            continue

        u = us[face_idx][mask].clamp(0, face_w - 1)
        v = vs[face_idx][mask].clamp(0, face_w - 1)
        u0 = u.floor().long()
        v0 = v.floor().long()
        u1 = (u0 + 1).clamp(0, face_w - 1)
        v1 = (v0 + 1).clamp(0, face_w - 1)

        du = (u - u0.float()).unsqueeze(-1)
        dv = (v - v0.float()).unsqueeze(-1)
        c00 = cubemap[face_idx, v0, u0]
        c10 = cubemap[face_idx, v0, u1]
        c01 = cubemap[face_idx, v1, u0]
        c11 = cubemap[face_idx, v1, u1]
        equirect[mask] = (c00 * (1 - du) + c10 * du) * (1 - dv) + (
            c01 * (1 - du) + c11 * du
        ) * dv

    return equirect


def panorama_depth_to_points(depth: Any) -> Any:
    import numpy as np

    height, width = depth.shape
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    uv_u, uv_v = np.meshgrid(u, v, indexing="xy")

    theta = (1 - uv_u) * (2 * np.pi)
    phi = uv_v * np.pi
    directions = np.stack(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ],
        axis=-1,
    ).astype(np.float32)
    return directions * depth[..., None]


def _robust_normalize(values: Any, valid: Any, low_pct: float = 2.0, high_pct: float = 98.0) -> Any:
    import numpy as np

    out = np.zeros(values.shape, dtype=np.float32)
    if not valid.any():
        return out

    lo, hi = np.percentile(values[valid], [low_pct, high_pct])
    if hi <= lo:
        lo = float(values[valid].min())
        hi = float(values[valid].max())
    if hi <= lo:
        return out

    out[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _turbo_colormap(values: Any) -> Any:
    import numpy as np

    x = np.clip(values, 0.0, 1.0)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    rgb = np.stack(
        [
            0.13572138 + 4.61539260 * x - 18.13093517 * x2 + 26.44122614 * x3 - 14.81492069 * x4 + 3.01290902 * x5,
            0.09140261 + 3.91738400 * x - 8.26182999 * x2 - 14.72839253 * x3 + 27.70401665 * x4 - 11.05048485 * x5,
            0.10667330 + 5.41479921 * x - 27.81883754 * x2 + 66.24150003 * x3 - 60.16093189 * x4 + 20.64264706 * x5,
        ],
        axis=-1,
    )
    return np.clip(rgb, 0.0, 1.0)


def save_panorama_geometry_visualizations(
    depth: Any,
    points: Any,
    depth_out_path: str | Path | None = None,
    points_out_path: str | Path | None = None,
) -> None:
    import numpy as np
    from PIL import Image

    if depth_out_path is not None:
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        disparity = np.zeros_like(depth, dtype=np.float32)
        disparity[valid] = 1.0 / depth[valid]
        depth_rgb = (_turbo_colormap(_robust_normalize(disparity, valid)) * 255).astype(np.uint8)

        depth_out_path = Path(depth_out_path)
        depth_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(depth_rgb, mode="RGB").save(depth_out_path, quality=95)

    if points_out_path is not None:
        points = np.asarray(points, dtype=np.float32)
        valid = np.isfinite(points).all(axis=-1)
        point_rgb = np.zeros(points.shape, dtype=np.float32)
        for channel in range(3):
            point_rgb[..., channel] = _robust_normalize(points[..., channel], valid)

        points_out_path = Path(points_out_path)
        points_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((point_rgb * 255).astype(np.uint8), mode="RGB").save(points_out_path, quality=95)


def render_marble_panorama_depth(
    ply_path: str | Path,
    out_path: str | Path,
    width: int,
    height: int | None = None,
    cube_map_width: int = 512,
    device: str | None = "cuda",
    near_plane: float = 0.1,
    far_plane: float = 1000.0,
    points_out_path: str | Path | None = None,
    depth_vis_out_path: str | Path | None = None,
    points_vis_out_path: str | Path | None = None,
) -> Any:
    import numpy as np
    import torch

    device_obj = _resolve_device(device)
    rasterization = _import_rasterization()
    gaussians = load_gsplat_scene_from_ply(ply_path, device=device_obj, force_rgb=True)

    front = np.array([0, 0, 1], dtype=np.float32)
    right = np.array([1, 0, 0], dtype=np.float32)
    up = np.array([0, -1, 0], dtype=np.float32)

    viewmats = _marble_cubemap_viewmats(front, right, up, device_obj)
    intrinsics = torch.tensor(
        [
            [cube_map_width / 2, 0, cube_map_width / 2],
            [0, cube_map_width / 2, cube_map_width / 2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device_obj,
    ).unsqueeze(0).repeat(6, 1, 1)

    kwargs = {
        "means": gaussians["means"],
        "quats": gaussians["quats"],
        "scales": gaussians["scales"],
        "opacities": gaussians["opacities"],
        "colors": gaussians["colors"],
        "sh_degree": gaussians["sh_degree"],
        "width": cube_map_width,
        "height": cube_map_width,
        "render_mode": "RGB+ED",
        "viewmats": viewmats,
        "Ks": intrinsics,
        "near_plane": near_plane,
        "far_plane": far_plane,
        "camera_model": "pinhole",
        "rasterize_mode": "antialiased",
    }
    with torch.inference_mode():
        try:
            renders, _, _ = rasterization(**kwargs)
        except TypeError:
            kwargs.pop("rasterize_mode", None)
            try:
                renders, _, _ = rasterization(**kwargs)
            except TypeError:
                kwargs.pop("camera_model", None)
                renders, _, _ = rasterization(**kwargs)

        equirect = cubemap_to_equirect(renders, width, height)
        depth = equirect[..., 3].detach().cpu().numpy().astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=far_plane, neginf=0.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, depth)

    points = None
    if points_out_path is not None or points_vis_out_path is not None:
        points = panorama_depth_to_points(depth).astype(np.float32)
    if points_out_path is not None:
        points_out_path = Path(points_out_path)
        points_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(points_out_path, points)
    if depth_vis_out_path is not None or points_vis_out_path is not None:
        save_panorama_geometry_visualizations(
            depth,
            points if points is not None else panorama_depth_to_points(depth),
            depth_out_path=depth_vis_out_path,
            points_out_path=points_vis_out_path,
        )

    return depth


def marble_to_panorama_transform() -> Any:
    """Return the homogeneous transform used by the internal mic renderer."""
    import numpy as np

    front_marble = np.asarray([0, 0, 1], dtype=np.float32)
    right_marble = np.asarray([1, 0, 0], dtype=np.float32)
    up_marble = np.asarray([0, -1, 0], dtype=np.float32)
    front_panorama = np.asarray([-1, 0, 0], dtype=np.float32)
    right_panorama = np.asarray([0, 1, 0], dtype=np.float32)
    up_panorama = np.asarray([0, 0, 1], dtype=np.float32)
    marble_basis = np.stack([front_marble, right_marble, up_marble], axis=0)
    panorama_basis = np.stack([front_panorama, right_panorama, up_panorama], axis=0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = panorama_basis.T @ marble_basis
    return transform


def _opencv_cubemap_viewmats(world_to_camera: Any, device: Any) -> Any:
    """Compose six OpenCV cubemap faces with an arbitrary base camera pose."""
    import torch

    # Each matrix maps base OpenCV camera coordinates into one face camera.
    # Face order agrees with ``cubemap_to_equirect``.
    relative = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],       # front
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],     # back
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]],      # right
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],      # left
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]],      # up
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],      # down
        ],
        dtype=torch.float32,
        device=device,
    )
    relative_h = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(6, 1, 1)
    relative_h[:, :3, :3] = relative
    base = torch.as_tensor(world_to_camera, dtype=torch.float32, device=device).unsqueeze(0)
    return relative_h @ base


def render_marble_pinhole_view(
    ply_path: str | Path,
    world_to_camera_panorama: Any,
    rgb_out_path: str | Path,
    depth_out_path: str | Path | None = None,
    depth_vis_out_path: str | Path | None = None,
    width: int = 1024,
    height: int = 1024,
    focal_length: float | None = None,
    device: str | None = "cuda",
    near_plane: float = 0.4,
    far_plane: float = 1000.0,
    panorama_rgb_out_path: str | Path | None = None,
    panorama_depth_out_path: str | Path | None = None,
    panorama_depth_vis_out_path: str | Path | None = None,
    panorama_width: int = 2048,
    cube_map_width: int = 512,
) -> dict[str, Any]:
    """Rasterize a Marble Gaussian scene from a SonoScene360 mic pose.

    ``world_to_camera_panorama`` maps panorama-world coordinates into an
    OpenCV camera.  Marble Gaussians use a different world basis, so the basis
    transform is composed before calling gsplat.  The function writes both the
    view and its expected-depth audit assets.
    """
    import numpy as np
    from PIL import Image
    import torch

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid render size {width}x{height}")
    device_obj = _resolve_device(device)
    rasterization = _import_rasterization()
    gaussians = load_gsplat_scene_from_ply(ply_path, device=device_obj, force_rgb=True)

    world_to_camera = np.asarray(world_to_camera_panorama, dtype=np.float32)
    if world_to_camera.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 world-to-camera matrix, got {world_to_camera.shape}")
    world_to_camera_marble = world_to_camera @ marble_to_panorama_transform()
    viewmats = torch.as_tensor(world_to_camera_marble, device=device_obj).unsqueeze(0)
    focal = float(focal_length) if focal_length is not None else float(max(width, height)) / 2.0
    intrinsics = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device_obj,
    ).unsqueeze(0)
    kwargs = {
        "means": gaussians["means"],
        "quats": gaussians["quats"],
        "scales": gaussians["scales"],
        "opacities": gaussians["opacities"],
        "colors": gaussians["colors"],
        "sh_degree": gaussians["sh_degree"],
        "width": width,
        "height": height,
        "render_mode": "RGB+ED",
        "viewmats": viewmats,
        "Ks": intrinsics,
        "near_plane": near_plane,
        "far_plane": far_plane,
        "camera_model": "pinhole",
        "rasterize_mode": "antialiased",
    }
    with torch.inference_mode():
        try:
            renders, alphas, _ = rasterization(**kwargs)
        except TypeError:
            kwargs.pop("rasterize_mode", None)
            try:
                renders, alphas, _ = rasterization(**kwargs)
            except TypeError:
                kwargs.pop("camera_model", None)
                renders, alphas, _ = rasterization(**kwargs)

    render = renders[0].detach().cpu().numpy().astype(np.float32)
    alpha = alphas[0].detach().cpu().numpy().astype(np.float32)
    rgb = np.clip(render[..., :3], 0.0, 1.0)
    depth = np.nan_to_num(render[..., 3], nan=0.0, posinf=far_plane, neginf=0.0)

    rgb_out_path = Path(rgb_out_path)
    rgb_out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(rgb_out_path, quality=95)
    if depth_out_path is not None:
        depth_out_path = Path(depth_out_path)
        depth_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_out_path, depth.astype(np.float32))
    if depth_vis_out_path is not None:
        valid = np.isfinite(depth) & (depth > near_plane) & (alpha[..., 0] > 1.0e-4)
        disparity = np.zeros_like(depth, dtype=np.float32)
        disparity[valid] = 1.0 / depth[valid]
        colors = (_turbo_colormap(_robust_normalize(disparity, valid)) * 255).astype(np.uint8)
        depth_vis_out_path = Path(depth_vis_out_path)
        depth_vis_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colors).save(depth_vis_out_path, quality=95)

    panorama_metadata = None
    if panorama_rgb_out_path is not None:
        cubemap_viewmats = _opencv_cubemap_viewmats(world_to_camera_marble, device_obj)
        cubemap_intrinsics = torch.tensor(
            [
                [cube_map_width / 2.0, 0.0, cube_map_width / 2.0],
                [0.0, cube_map_width / 2.0, cube_map_width / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device_obj,
        ).unsqueeze(0).repeat(6, 1, 1)
        panorama_kwargs = {
            **{key: kwargs[key] for key in ("means", "quats", "scales", "opacities", "colors", "sh_degree")},
            "width": cube_map_width,
            "height": cube_map_width,
            "render_mode": "RGB+ED",
            "viewmats": cubemap_viewmats,
            "Ks": cubemap_intrinsics,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "camera_model": "pinhole",
            "rasterize_mode": "antialiased",
        }
        with torch.inference_mode():
            try:
                cubemap, _, _ = rasterization(**panorama_kwargs)
            except TypeError:
                panorama_kwargs.pop("rasterize_mode", None)
                try:
                    cubemap, _, _ = rasterization(**panorama_kwargs)
                except TypeError:
                    panorama_kwargs.pop("camera_model", None)
                    cubemap, _, _ = rasterization(**panorama_kwargs)
            panorama = cubemap_to_equirect(cubemap, panorama_width)
        panorama_array = panorama.detach().cpu().numpy().astype(np.float32)
        panorama_rgb = np.clip(panorama_array[..., :3], 0.0, 1.0)
        panorama_depth = np.nan_to_num(
            panorama_array[..., 3], nan=0.0, posinf=far_plane, neginf=0.0
        )
        panorama_rgb_out_path = Path(panorama_rgb_out_path)
        panorama_rgb_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((panorama_rgb * 255.0).astype(np.uint8)).save(
            panorama_rgb_out_path, quality=95
        )
        if panorama_depth_out_path is not None:
            panorama_depth_out_path = Path(panorama_depth_out_path)
            panorama_depth_out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(panorama_depth_out_path, panorama_depth.astype(np.float32))
        if panorama_depth_vis_out_path is not None:
            panorama_valid = np.isfinite(panorama_depth) & (panorama_depth > near_plane)
            panorama_disparity = np.zeros_like(panorama_depth, dtype=np.float32)
            panorama_disparity[panorama_valid] = 1.0 / panorama_depth[panorama_valid]
            panorama_colors = (
                _turbo_colormap(_robust_normalize(panorama_disparity, panorama_valid)) * 255
            ).astype(np.uint8)
            panorama_depth_vis_out_path = Path(panorama_depth_vis_out_path)
            panorama_depth_vis_out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panorama_colors).save(panorama_depth_vis_out_path, quality=95)
        panorama_metadata = {
            "width": panorama_width,
            "height": panorama_width // 2,
            "cube_map_width": cube_map_width,
        }

    return {
        "world_to_camera_marble": world_to_camera_marble.tolist(),
        "width": width,
        "height": height,
        "focal_length": focal,
        "panorama": panorama_metadata,
    }
