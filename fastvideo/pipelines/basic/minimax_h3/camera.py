# SPDX-License-Identifier: Apache-2.0
"""Plücker ray fields for MiniMax-H3 camera control.

A camera trajectory is turned into a dense, per-token geometric quantity rather than a pose vector,
because the ControlNet that consumes it is a token-space trunk laid over the packed sequence: every
row it owns is one ``(t, h, w)`` cell of the target video grid, and it needs a value there. The
Plücker coordinates of the ray through a cell — ``(d, o x d)`` for ray origin ``o`` and unit
direction ``d`` — are that value. They are invariant to where along the ray you sample, so two
frames that see the same world ray carry the same six numbers regardless of how far the camera
travelled, which is what makes translation and rotation separable to the trunk.

The field is built directly at the *latent* grid, not at pixel resolution and then downsampled. A
downsampled ray field is not the ray field of the downsampled image: averaging directions across a
16x16 pixel block shortens the vector and biases it toward the block's dominant direction. Sampling
one ray per latent cell centre is both cheaper and correct.

Trajectories are normalized before embedding (:func:`normalize_camera_trajectory`): the first frame
becomes the identity and translations are rescaled to a fixed radius. Without it the model would
have to learn the dataset's absolute world origin and unit scale, neither of which survives contact
with a different scene.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Plücker coordinates are six numbers per ray: the unit direction and its moment about the origin.
MINIMAX_H3_CAMERA_CHANNELS = 6
# Radius the first camera is rescaled to. Any positive constant works; this one keeps normalized
# translations in the same order of magnitude as the unit-length directions they sit beside.
MINIMAX_H3_CAMERA_SCALE = 2.0


def normalize_camera_trajectory(
    world_to_camera: torch.Tensor,
    *,
    camera_scale: float = MINIMAX_H3_CAMERA_SCALE,
) -> torch.Tensor:
    """Re-express a world-to-camera trajectory relative to its own first frame.

    Applies the three steps a scene-agnostic camera representation needs: rebase onto frame 0 so the
    world origin drops out, recentre on the mean camera position so a trajectory that drifts is not
    also translating in the embedding, and rescale so the representation does not encode the
    dataset's choice of unit.

    Args:
        world_to_camera: ``[F, 4, 4]`` world-to-camera extrinsics.
        camera_scale: Radius to rescale the first camera's translation to.

    Returns:
        ``[F, 4, 4]`` normalized world-to-camera extrinsics.
    """
    if world_to_camera.ndim != 3 or world_to_camera.shape[-2:] != (4, 4):
        raise ValueError(f"Extrinsics must have shape [frames, 4, 4], got {tuple(world_to_camera.shape)}.")
    if world_to_camera.shape[0] == 0:
        raise ValueError("A camera trajectory must contain at least one frame.")

    world_to_camera = world_to_camera.to(torch.float64)
    # Right-multiplying by frame 0's camera-to-world sends frame 0 to the identity and leaves every
    # other frame expressed as its motion relative to frame 0.
    world_to_camera = world_to_camera @ torch.linalg.inv(world_to_camera[:1])

    camera_to_world = torch.linalg.inv(world_to_camera)
    positions = camera_to_world[:, :3, 3]
    camera_to_world[:, :3, 3] = positions - positions.mean(dim=0, keepdim=True)

    # Frame 0 sits at the origin before recentring, so its post-recentring radius measures the
    # trajectory's spread. A stationary camera has no spread and keeps unit scale.
    radius = torch.linalg.vector_norm(camera_to_world[0, :3, 3])
    if float(radius) > 1e-5:
        camera_to_world[:, :3, 3] *= camera_scale / radius
    return torch.linalg.inv(camera_to_world).to(torch.float32)


def rescale_intrinsics(
    intrinsics: torch.Tensor,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Rescale pixel-unit intrinsics from the resolution they were measured at to a sampling grid."""
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsics must have shape [frames, 3, 3], got {tuple(intrinsics.shape)}.")
    source_height, source_width = source_size
    target_height, target_width = target_size
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("Intrinsic rescaling needs positive source and target sizes.")
    scale = intrinsics.new_tensor([
        [target_width / source_width, 1.0, target_width / source_width],
        [1.0, target_height / source_height, target_height / source_height],
        [1.0, 1.0, 1.0],
    ])
    return intrinsics * scale


def plucker_ray_field(
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sample one Plücker ray per grid cell centre.

    Args:
        world_to_camera: ``[F, 4, 4]`` extrinsics, already normalized and expressed in the grid's
            own coordinate frame.
        intrinsics: ``[F, 3, 3]`` pinhole intrinsics **in units of the requested grid**, not of the
            original pixel resolution. Use :func:`rescale_intrinsics` first when they differ.
        height: Grid height.
        width: Grid width.

    Returns:
        ``[F, 6, height, width]`` float32 Plücker field, direction channels first.
    """
    if world_to_camera.shape[0] != intrinsics.shape[0]:
        raise ValueError(f"Extrinsics and intrinsics must cover the same frames, got "
                         f"{world_to_camera.shape[0]} and {intrinsics.shape[0]}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"A ray field needs a positive grid, got {height}x{width}.")

    device = world_to_camera.device
    extrinsics = world_to_camera.to(torch.float32)
    intrinsics = intrinsics.to(device=device, dtype=torch.float32)
    num_frames = extrinsics.shape[0]

    camera_to_world = torch.linalg.inv(extrinsics.to(torch.float64)).to(torch.float32)
    rotation = camera_to_world[:, :3, :3]
    origin = camera_to_world[:, :3, 3]

    grid_y = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    grid_x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    homogeneous = torch.stack((mesh_x, mesh_y, torch.ones_like(mesh_x)), dim=-1).reshape(1, -1, 3)

    # Unproject to camera space at unit depth, then rotate into world space. Normalizing after the
    # rotation rather than before keeps the direction a unit vector under any intrinsic skew.
    camera_rays = homogeneous @ torch.linalg.inv(intrinsics).transpose(-1, -2)
    directions = F.normalize(camera_rays @ rotation.transpose(-1, -2), dim=-1)
    origins = origin[:, None, :].expand_as(directions)

    plucker = torch.cat((directions, torch.cross(origins, directions, dim=-1)), dim=-1)
    plucker = plucker.reshape(num_frames, height, width, MINIMAX_H3_CAMERA_CHANNELS).permute(0, 3, 1, 2)
    if not bool(torch.isfinite(plucker).all()):
        raise ValueError("The camera trajectory produced a non-finite ray field; check for singular intrinsics "
                         "or a degenerate extrinsic matrix.")
    return plucker.contiguous()


def build_camera_latent(
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    pixel_size: tuple[int, int],
    num_latent_frames: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Build the ``[1, 6, T, H, W]`` camera field the ControlNet consumes.

    Poses arrive per pixel frame; the trunk needs one per *latent* frame. Latent frames span
    unequal, non-contiguous runs of pixel frames under H3's chunked causal VAE, so rather than
    reproduce that geometry here the trajectory is resampled onto ``num_latent_frames`` evenly
    spaced source poses. The residual timing error is a fraction of a frame at 24 fps, well inside
    what a 16x-downsampled ray field can resolve.

    Args:
        world_to_camera: ``[F_pixel, 4, 4]`` world-to-camera extrinsics.
        intrinsics: ``[F_pixel, 3, 3]`` intrinsics in units of ``pixel_size``.
        latent_height: Latent grid height.
        latent_width: Latent grid width.
        pixel_size: ``(height, width)`` the intrinsics were measured at.
        num_latent_frames: Number of latent frames to emit.
        normalize: Whether to rebase and rescale the trajectory first.

    Returns:
        ``[1, 6, num_latent_frames, latent_height, latent_width]`` float32.
    """
    if num_latent_frames <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {num_latent_frames}.")
    num_pixel_frames = world_to_camera.shape[0]
    if num_pixel_frames == 0:
        raise ValueError("A camera trajectory must contain at least one frame.")

    if normalize:
        world_to_camera = normalize_camera_trajectory(world_to_camera)
    intrinsics = rescale_intrinsics(
        intrinsics,
        source_size=pixel_size,
        target_size=(latent_height, latent_width),
    )

    positions = torch.linspace(0.0, num_pixel_frames - 1, num_latent_frames, device=world_to_camera.device)
    indices = positions.round().to(torch.long).clamp_(0, num_pixel_frames - 1)
    field = plucker_ray_field(
        world_to_camera.index_select(0, indices),
        intrinsics.index_select(0, indices.to(intrinsics.device)),
        height=latent_height,
        width=latent_width,
    )
    return field.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


__all__ = [
    "MINIMAX_H3_CAMERA_CHANNELS",
    "MINIMAX_H3_CAMERA_SCALE",
    "build_camera_latent",
    "normalize_camera_trajectory",
    "plucker_ray_field",
    "rescale_intrinsics",
]
