"""
Compact statistical height representations for 2.5D elevation grid cells.
Stores [z_max, z_min, delta_z, variance, slope] without copying raw points.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np


@dataclass(slots=True)
class CellStats:
    """
    Compact 2.5D statistical descriptor of a spatial grid cell.
    Zero raw points stored to achieve edge memory efficiency.
    """
    z_min: float
    z_max: float
    delta_z: float
    variance: float
    slope: float        # Inclination angle in degrees relative to horizontal
    point_count: int
    mean_z: float

    def to_tuple(self):
        """Returns compact numerical tuple [z_max, z_min, delta_z, variance, slope]."""
        return (self.z_max, self.z_min, self.delta_z, self.variance, self.slope)

    def to_dict(self):
        return {
            "z_min": round(self.z_min, 3),
            "z_max": round(self.z_max, 3),
            "delta_z": round(self.delta_z, 3),
            "variance": round(self.variance, 4),
            "slope": round(self.slope, 2),
            "point_count": self.point_count,
            "mean_z": round(self.mean_z, 3),
        }


def filter_dynamic_points(
    points: np.ndarray,
    dynamic_tracks: Optional[List[Any]] = None,
    margin: float = 0.25,
    ground_z_estimate: Optional[float] = None,
) -> np.ndarray:
    """
    Filters out points falling inside bounding boxes or footprints of confirmed dynamic tracks.
    Prevents vehicle roofs and pedestrian heads from contaminating terrain height (z_min, z_max, delta_z).
    Ensures dynamic exclusion ONLY filters points elevated above the ground (z > ground_z_estimate + 0.15)
    so that drivable road points beneath or near tracked objects are preserved.
    """
    if dynamic_tracks is None or len(dynamic_tracks) == 0 or len(points) == 0:
        return points

    mask = np.ones(len(points), dtype=bool)
    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]

    if ground_z_estimate is None:
        z_cand = pz[(pz >= -2.2) & (pz <= -1.35)]
        if len(z_cand) >= 3:
            ground_z_estimate = float(np.median(z_cand))
        else:
            ground_z_estimate = -1.60

    elevated_cutoff = ground_z_estimate + 0.15

    for t in dynamic_tracks:
        if hasattr(t, "bbox") and t.bbox is not None and len(t.bbox) >= 6:
            min_x, max_x, min_y, max_y, min_z, max_z = t.bbox
        elif hasattr(t, "dimensions") and hasattr(t, "x") and hasattr(t, "y"):
            l, w, h = t.dimensions
            tz = getattr(t, "z", 0.0)
            min_x, max_x = t.x - l * 0.5, t.x + l * 0.5
            min_y, max_y = t.y - w * 0.5, t.y + w * 0.5
            min_z, max_z = tz - h * 0.5, tz + h * 0.5
        else:
            continue

        in_box = (
            (px >= min_x - margin) & (px <= max_x + margin) &
            (py >= min_y - margin) & (py <= max_y + margin) &
            (pz >= min_z - margin) & (pz <= max_z + margin)
        )
        is_dynamic = in_box & (pz > elevated_cutoff)
        mask &= ~is_dynamic

    return points[mask]


def is_within_bounds(points: np.ndarray, bounds: Optional[Any] = None) -> np.ndarray:
    """Returns boolean mask of points strictly within the specified or default SpatialBounds."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    from backend.config import BOUNDS
    b = bounds if bounds is not None else BOUNDS
    return (
        (points[:, 0] >= b.X_MIN) & (points[:, 0] <= b.X_MAX) &
        (points[:, 1] >= b.Y_MIN) & (points[:, 1] <= b.Y_MAX) &
        (points[:, 2] >= b.Z_MIN) & (points[:, 2] <= b.Z_MAX)
    )


def compute_cell_statistics(
    points: np.ndarray,
    cell_size: float = 0.50,
    dynamic_tracks: Optional[List[Any]] = None,
    bounds: Optional[Any] = None,
) -> Optional[CellStats]:
    """
    Computes statistical attributes for points falling inside a 2D cell.
    Excludes dynamic obstacle points if dynamic_tracks is provided.
    Guarantees points within the operational bounds are preserved and evaluated.

    Args:
        points: (N, 3) or (N, 4) point array containing [x, y, z, ...]
        cell_size: Spatial span of the cell in meters.
        dynamic_tracks: Optional list of active TrackedObject instances.
        bounds: Optional SpatialBounds instance to validate extent.

    Returns:
        CellStats instance or None if empty.
    """
    if bounds is not None and len(points) > 0:
        points = points[is_within_bounds(points, bounds)]

    if dynamic_tracks is not None and len(dynamic_tracks) > 0 and len(points) > 0:
        points = filter_dynamic_points(points, dynamic_tracks)

    n = len(points)
    if n == 0:
        return None

    z = points[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    delta_z = z_max - z_min
    mean_z = float(np.mean(z))
    variance = float(np.var(z)) if n > 1 else 0.0

    # Fast 2.5D geometric slope estimation: theta = arctan(delta_z / cell_size)
    if delta_z < 0.01 or cell_size <= 0.0:
        slope_deg = 0.0
    else:
        slope_deg = float(np.degrees(np.arctan2(delta_z, cell_size)))

    return CellStats(
        z_min=z_min,
        z_max=z_max,
        delta_z=delta_z,
        variance=variance,
        slope=slope_deg,
        point_count=n,
        mean_z=mean_z,
    )
