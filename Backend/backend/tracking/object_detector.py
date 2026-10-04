"""
PS 26053 3D Object Detection & Tracker Gating Interface.

Provides:
- ObjectDetector3D capable of consuming PointPillars-style 3D bounding boxes:
  [x, y, z, l, w, h, yaw, class, confidence]
- Geometric Fallback when detection weights are absent:
  DBSCAN (eps=0.45m, min_samples=6, spatial decimation=0.15m)
- Dynamic Point Exclusion:
  Excludes points within confirmed dynamic 3D track boxes before elevation statistics computation,
  preventing ghost walls, false terrain steps, and false elevation variance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    DBSCAN = None
    HAS_SKLEARN = False

from backend.tracking.clustering import DetectedCluster


@dataclass(slots=True)
class BoundingBox3D:
    """Represents a 3D Oriented Bounding Box detection."""
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    class_name: str
    confidence: float
    velocity: Optional[Tuple[float, float]] = None

    def to_cluster(self, points: Optional[np.ndarray] = None) -> DetectedCluster:
        min_x = self.x - self.l * 0.5
        max_x = self.x + self.l * 0.5
        min_y = self.y - self.w * 0.5
        max_y = self.y + self.w * 0.5
        min_z = self.z - self.h * 0.5
        max_z = self.z + self.h * 0.5
        pts = points if points is not None else np.empty((0, 3), dtype=np.float32)
        return DetectedCluster(
            centroid=(self.x, self.y, self.z),
            dimensions=(self.l, self.w, self.h),
            bbox=(min_x, max_x, min_y, max_y, min_z, max_z),
            point_count=len(pts) if len(pts) > 0 else 50,
            points=pts,
            yaw=self.yaw,
            velocity=self.velocity,
            label=self.class_name,
            source="pointpillars" if hasattr(self, "source") and getattr(self, "source") == "pointpillars" else "geometric",
        )


class ObjectDetector3D:
    """
    3D Object Detection interface for PS 26053.
    If no trained detector weights exist, cleanly reports:
        model_status["detection"] = "MODEL NOT LOADED"
    and routes through calibrated geometric fallback DBSCAN.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.model_status = "MODEL NOT LOADED"
        self.model_name = None
        self.execution_provider = None

        # Geometric Fallback Parameters mandated by PS 26053
        self.eps = 0.45
        self.min_samples = 6
        self.voxel_decimation = 0.15

        self._dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=1) if HAS_SKLEARN else None

    def consume_pointpillars_output(
        self, boxes_array: np.ndarray, class_names: Optional[List[str]] = None
    ) -> List[DetectedCluster]:
        """
        Consumes PointPillars-style 3D detection proposals:
        Shape: (M, >= 8) -> [x, y, z, l, w, h, yaw, class, (confidence)]
        """
        if boxes_array is None or len(boxes_array) == 0:
            return []

        clusters: List[DetectedCluster] = []
        default_names = ["car", "pedestrian", "cyclist", "truck", "barrier"]
        names = class_names or default_names

        for det in boxes_array:
            if len(det) < 7:
                continue
            x, y, z = float(det[0]), float(det[1]), float(det[2])
            l, w, h = float(det[3]), float(det[4]), float(det[5])
            yaw = float(det[6])
            cls_idx = int(det[7]) if len(det) > 7 else 0
            conf = float(det[8]) if len(det) > 8 else 1.0
            label = names[cls_idx] if 0 <= cls_idx < len(names) else "obstacle"

            min_x, max_x = x - l * 0.5, x + l * 0.5
            min_y, max_y = y - w * 0.5, y + w * 0.5
            min_z, max_z = z - h * 0.5, z + h * 0.5

            clusters.append(
                DetectedCluster(
                    centroid=(x, y, z),
                    dimensions=(l, w, h),
                    bbox=(min_x, max_x, min_y, max_y, min_z, max_z),
                    point_count=50,
                    points=np.empty((0, 3), dtype=np.float32),
                    yaw=yaw,
                    label=label,
                    source="pointpillars",
                )
            )

        return clusters

    def _decimate_points(self, points: np.ndarray) -> np.ndarray:
        """Applies 0.15m spatial voxel decimation to accelerate DBSCAN."""
        if len(points) == 0:
            return points
        xyz = points[:, :3]
        voxel_coords = np.floor(xyz / self.voxel_decimation).astype(np.int32)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        return points[unique_indices]

    def detect_geometric_fallback(self, obstacle_points: np.ndarray) -> List[DetectedCluster]:
        """
        Executes calibrated geometric fallback detection using:
        - Spatial decimation: 0.15 m
        - DBSCAN: eps = 0.45 m, min_samples = 6
        """
        if obstacle_points is None or len(obstacle_points) < self.min_samples:
            return []

        # 1. Spatial decimation
        decimated = self._decimate_points(obstacle_points)
        if len(decimated) < self.min_samples:
            return []

        xyz = decimated[:, :3]

        # 2. DBSCAN Clustering
        if self._dbscan is None:
            return []

        labels = self._dbscan.fit_predict(xyz)
        unique_labels = set(labels) - {-1}

        clusters: List[DetectedCluster] = []
        for lab in unique_labels:
            c_mask = labels == lab
            c_pts = xyz[c_mask]
            if len(c_pts) < self.min_samples:
                continue

            min_x, min_y, min_z = np.min(c_pts, axis=0)
            max_x, max_y, max_z = np.max(c_pts, axis=0)
            dim_x = max(0.2, max_x - min_x)
            dim_y = max(0.2, max_y - min_y)
            dim_z = max(0.2, max_z - min_z)
            cx = (min_x + max_x) * 0.5
            cy = (min_y + max_y) * 0.5
            cz = (min_z + max_z) * 0.5

            clusters.append(
                DetectedCluster(
                    centroid=(float(cx), float(cy), float(cz)),
                    dimensions=(float(dim_x), float(dim_y), float(dim_z)),
                    bbox=(float(min_x), float(max_x), float(min_y), float(max_y), float(min_z), float(max_z)),
                    point_count=len(c_pts),
                    points=c_pts,
                    yaw=0.0,
                    label="obstacle",
                    source="geometric",
                )
            )

        return clusters

    def detect(self, obstacle_points: np.ndarray) -> List[DetectedCluster]:
        """
        Primary detection method. Routes through geometric fallback when weights absent.
        """
        if self.model is not None:
            # Model inference route (if trained model available)
            pass
        return self.detect_geometric_fallback(obstacle_points)


def exclude_dynamic_points_from_elevation(
    points: np.ndarray,
    confirmed_dynamic_tracks: List[Any],
    margin: float = 0.20,
    ground_z_estimate: Optional[float] = None,
) -> np.ndarray:
    """
    Excludes points falling inside confirmed dynamic 3D track boxes BEFORE
    elevation statistics (Z_min, Z_max, delta_Z, sigma^2) are computed.
    Prevents moving vehicles from generating ghost walls or false terrain steps,
    while preserving drivable ground returns beneath and around the vehicle chassis.
    """
    if points is None or len(points) == 0 or not confirmed_dynamic_tracks:
        return points

    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    n_pts = len(pts)

    if ground_z_estimate is None:
        z_cand = z[(z >= -2.2) & (z <= -1.35)]
        if len(z_cand) >= 3:
            ground_z_estimate = float(np.median(z_cand))
        else:
            ground_z_estimate = -1.60

    elevated_cutoff = ground_z_estimate + 0.15
    dynamic_mask = np.zeros(n_pts, dtype=bool)

    for track in confirmed_dynamic_tracks:
        # Check if track is dynamic
        is_dyn = getattr(track, "is_dynamic", False)
        is_conf = getattr(track, "is_confirmed", True)
        if not (is_dyn and is_conf):
            continue

        bbox = getattr(track, "bbox", None)
        if bbox is not None and len(bbox) >= 6:
            min_x, max_x, min_y, max_y, min_z, max_z = bbox
        else:
            tx = getattr(track, "x", 0.0)
            ty = getattr(track, "y", 0.0)
            tz = getattr(track, "z", -1.5)
            dims = getattr(track, "dimensions", (2.0, 4.0, 1.5))
            l, w, h = dims[0], dims[1], dims[2]
            min_x, max_x = tx - l * 0.5, tx + l * 0.5
            min_y, max_y = ty - w * 0.5, ty + w * 0.5
            min_z, max_z = tz - h * 0.5, tz + h * 0.5

        # Check points in box with margin, elevated above ground
        in_box = (
            (x >= min_x - margin) & (x <= max_x + margin) &
            (y >= min_y - margin) & (y <= max_y + margin) &
            (z >= min_z - margin) & (z <= max_z + margin)
        )
        is_dynamic = in_box & (z > elevated_cutoff)
        dynamic_mask |= is_dynamic

    return pts[~dynamic_mask]
