"""
Traversability Costmap Evaluator.
Translates 2.5D cell elevation metrics (slope, delta_z, roughness)
and dynamic obstacle presences into traversability cost values [0, 255].
  - 0 to 50: Safe traversable terrain
  - 51 to 180: Caution / steep slope / rough ground
  - 181 to 255: Lethal obstacle / non-traversable
"""

from typing import Any, List, Optional
import numpy as np

from backend.config import COSTMAP
from backend.mapping.quadtree import QuadtreeNode


def compute_multifactor_traversability(
    delta_z: float,
    slope_deg: float,
    variance: float,
    s_class: float = 0.0,
    s_drop: float = 0.0,
    alpha: float = 80.0,
    beta: float = 60.0,
    gamma: float = 40.0,
    z_crit: float = 0.20,
    theta_crit: float = 25.0,
    sigma_sq_crit: float = 0.05,
) -> int:
    """
    PS 26053 Multi-Factor Traversability Computation:
    T = clip(alpha * (delta_z / Zcrit) + beta * (theta / theta_crit) + gamma * (sigma^2 / sigma^2_crit) + S_class + S_drop, 0, 255)

    Selected weights and critical thresholds:
    - alpha = 80.0 (step height contribution up to 80 at Zcrit)
    - beta = 60.0 (slope contribution up to 60 at theta_crit)
    - gamma = 40.0 (roughness/variance contribution up to 40 at sigma^2_crit)
    - Zcrit = 0.20 m
    - theta_crit = 25.0 degrees
    - sigma^2_crit = 0.05 m^2
    """
    term_step = alpha * (float(delta_z) / max(z_crit, 1e-6))
    term_slope = beta * (float(slope_deg) / max(theta_crit, 1e-6))
    term_roughness = gamma * (float(variance) / max(sigma_sq_crit, 1e-6))
    raw_cost = term_step + term_slope + term_roughness + float(s_class) + float(s_drop)
    return int(np.clip(raw_cost, 0, 255))


class CostmapEvaluator:
    def __init__(
        self,
        safe_max: int = COSTMAP.SAFE_MAX,
        caution_max: int = COSTMAP.CAUTION_MAX,
        lethal_val: int = COSTMAP.LETHAL_VAL,
        max_traversable_slope: float = COSTMAP.MAX_TRAVERSABLE_SLOPE_DEG,
        lethal_slope: float = COSTMAP.LETHAL_SLOPE_DEG,
        max_step_height: float = getattr(COSTMAP, "SAFE_STEP_HEIGHT", 0.12),
        lethal_step_height: float = getattr(COSTMAP, "CAUTION_STEP_HEIGHT", 0.25),
        ground_min: float = getattr(COSTMAP, "GROUND_MIN", -2.1),
        ground_max: float = getattr(COSTMAP, "GROUND_MAX", -1.35),
        elevated_obstacle_z: float = getattr(COSTMAP, "ELEVATED_OBSTACLE_Z", -1.2),
    ):
        self.safe_max = safe_max
        self.caution_max = caution_max
        self.lethal_val = lethal_val
        self.max_slope = max_traversable_slope
        self.lethal_slope = lethal_slope
        self.max_step = max_step_height
        self.lethal_step = lethal_step_height
        self.ground_min = ground_min
        self.ground_max = ground_max
        self.elevated_obstacle_z = elevated_obstacle_z

        from backend.config import BOUNDS
        self._res = 0.25
        self._inv_res = 4.0
        self._x_min = BOUNDS.X_MIN
        self._y_min = BOUNDS.Y_MIN
        self._num_x = int(np.ceil((BOUNDS.X_MAX - BOUNDS.X_MIN) * 4.0))
        self._num_y = int(np.ceil((BOUNDS.Y_MAX - BOUNDS.Y_MIN) * 4.0))
        self._total_cells = self._num_x * self._num_y

        from backend.mapping.cell_statistics import CellStats
        self._inpaint_pool: List[QuadtreeNode] = [
            QuadtreeNode(
                x=0.0,
                y=0.0,
                size=0.50,
                depth=0,
                stats=CellStats(0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0),
                cost=0,
                is_leaf=True,
            )
            for _ in range(500)
        ]

    def evaluate_node_cost(self, node: QuadtreeNode) -> int:
        """
        Computes cost in [0, 255] for a single quadtree leaf node based on its CellStats
        and absolute elevation relative to the true ground baseline.
        """
        if node.stats is None:
            return 0

        stats = node.stats
        z_mean = stats.mean_z
        z_max = stats.z_max
        z_mean = float(getattr(node, 'z_mean', getattr(node, 'z', -1.68)))
        delta_z = float(getattr(node, 'delta_z', 0.0))
        z_max = float(getattr(node, 'z_max', z_mean))
        slope = stats.slope

        if node.cost == 100:
            return 100

        # Drivable road surface envelope:
        # Asphalt elevation sits between -2.20m and -1.25m with step height <= 0.18m
        if (-2.20 <= z_mean <= -1.25) and (delta_z <= 0.18) and (z_max <= -1.15):
            node.cost = 0
            node.is_obstacle = False
            node.is_hazard = False
            return 0

        # 1. LETHAL HAZARD / RED (Cost 181 to 255):
        # Any cell with confirmed elevated obstacle returns, large step > 0.25m, or ditch drop-off < -2.15m
        if (
            (z_mean > self.elevated_obstacle_z and getattr(stats, "point_count", 0) >= 3)
            or delta_z > self.lethal_step
            or z_mean < self.ground_min
            or slope >= self.lethal_slope
        ):
            return self.lethal_val

        # 2. CAUTION / YELLOW (Cost 51 to 180):
        # Sloped terrain, curbs, or unconfirmed transition areas (delta_z between 0.12m and 0.25m, or -1.35m < z_mean <= -1.2m)
        if delta_z > self.max_step or z_mean > self.ground_max or slope > self.max_slope:
            step_ratio = (delta_z - self.max_step) / max(0.001, self.lethal_step - self.max_step) if delta_z > self.max_step else 0.0
            z_ratio = (z_mean - self.ground_max) / max(0.001, self.elevated_obstacle_z - self.ground_max) if z_mean > self.ground_max else 0.0
            slope_ratio = (slope - self.max_slope) / max(0.001, self.lethal_slope - self.max_slope) if slope > self.max_slope else 0.0
            ratio = min(1.0, max(0.0, max(step_ratio, z_ratio, slope_ratio)))
            val = int(self.safe_max + 1 + ratio * (self.caution_max - self.safe_max - 1))
            return min(self.caution_max, max(self.safe_max + 1, val))

        # 3. SAFE / GREEN (Cost 0 to 50):
        # Cell elevation must be strictly within the ground baseline (-2.1m <= z_mean <= -1.35m)
        # and internal step height delta_z <= 0.12m.
        if node.cost == 100:
            return 100
        if self.max_slope > 0:
            slope_cost = int((slope / self.max_slope) * self.safe_max)
            return min(self.safe_max, max(0, slope_cost))
        return 0

    def evaluate_leaves(self, leaves: List[QuadtreeNode]):
        """Evaluates cost for all leaf nodes in place incorporating absolute ground baseline."""
        lethal_step = self.lethal_step
        max_step = self.max_step
        lethal_slope = self.lethal_slope
        max_slope = self.max_slope
        safe_max = self.safe_max
        caution_max = self.caution_max
        lethal_val = self.lethal_val
        ground_min = self.ground_min
        ground_max = self.ground_max
        elev_obs_z = self.elevated_obstacle_z

        step_diff = max(0.001, lethal_step - max_step)
        slope_diff = max(0.001, lethal_slope - max_slope)
        z_diff = max(0.001, elev_obs_z - ground_max)
        caution_range = caution_max - safe_max - 1
        inv_max_slope = (safe_max / max_slope) if max_slope > 0 else 0.0

        for leaf in leaves:
            if leaf.cost == 100:
                continue

            z_mean = float(getattr(leaf, 'z_mean', getattr(leaf, 'z', -1.68)))
            delta_z = float(getattr(leaf, 'delta_z', 0.0))
            z_max = float(getattr(leaf, 'z_max', z_mean))

            # Lock ground cells to 0 regardless of adjacent obstacle dilation
            if (-2.20 <= z_mean <= -1.25) and (delta_z <= 0.18) and (z_max <= -1.15):
                leaf.cost = 0
                leaf.is_obstacle = False
                leaf.is_hazard = False
                continue

            if leaf.cost >= lethal_val:
                continue

            stats = leaf.stats
            slope = stats.slope if stats is not None else 0.0
            pt_count = stats.point_count if stats is not None else getattr(leaf, 'point_count', 0)

            # 1. LETHAL HAZARD / RED (Cost = 255):
            # Elevated objects in air (z_mean > -1.2m with >= 3 pts), large step (delta_z > 0.25m),
            # deep trench (z_mean < -2.15m), or rollover slope
            if (
                (z_mean > elev_obs_z and pt_count >= 3)
                or delta_z > lethal_step
                or z_mean < ground_min
                or slope >= lethal_slope
            ):
                leaf.cost = lethal_val
                continue


            # 2. CAUTION / YELLOW (Cost 51 to 180):
            # Sloped terrain, curbs, or unconfirmed transition areas (delta_z between 0.12m and 0.25m, or -1.35m < z_mean <= -1.2m)
            if delta_z > max_step or z_mean > ground_max or slope > max_slope:
                step_ratio = (delta_z - max_step) / step_diff if delta_z > max_step else 0.0
                z_ratio = (z_mean - ground_max) / z_diff if z_mean > ground_max else 0.0
                slope_ratio = (slope - max_slope) / slope_diff if slope > max_slope else 0.0
                ratio = min(1.0, max(0.0, max(step_ratio, z_ratio, slope_ratio)))
                leaf.cost = int(safe_max + 1 + ratio * caution_range)
                continue

            # 3. SAFE / GREEN (Cost 0 to 50):
            # Drivable ground strictly within -2.1m <= z_mean <= -1.35m with delta_z <= 0.12m
            if leaf.cost == 100:
                continue
            if slope <= max_slope:
                cost_val = int(slope * inv_max_slope)
                leaf.cost = max(0, min(255, cost_val))
            else:
                leaf.cost = safe_max

    def filter_dynamic_points(
        self,
        points: np.ndarray,
        dynamic_tracks: Optional[List[Any]] = None,
        margin: float = 0.25,
        ground_z_estimate: Optional[float] = None,
    ) -> np.ndarray:
        """Excludes points falling inside bounding boxes of active dynamic tracks."""
        from backend.mapping.cell_statistics import filter_dynamic_points as _fdp
        return _fdp(points, dynamic_tracks=dynamic_tracks, margin=margin, ground_z_estimate=ground_z_estimate)

    def apply_obstacle_occupancy(
        self,
        leaves: List[QuadtreeNode],
        obstacle_points: np.ndarray,
        min_obs_points: int = 3,
        dynamic_tracks: Optional[List[Any]] = None,
    ):
        """
        Flags quadtree cells containing non-ground obstacle points as lethal (255).
        Evaluated at 0.25m fine resolution matching the subdivided quadtree leaves
        with a minimum point count threshold to prevent single-point noise/outliers
        from falsely marking clear road terrain as lethal.
        Excludes points from dynamic_tracks to prevent baking moving objects into static terrain.
        High-speed spatial hashing in < 0.3ms.
        """
        if len(leaves) == 0 or len(obstacle_points) == 0:
            return

        if dynamic_tracks is not None and len(dynamic_tracks) > 0:
            obstacle_points = self.filter_dynamic_points(obstacle_points, dynamic_tracks)
            if len(obstacle_points) == 0:
                return

        # Filter out points that are within the drivable road ground elevation window.
        # Actual rigid obstacles project above the vehicle bumper / clearance line (Z > -1.30m)
        elevated_mask = obstacle_points[:, 2] > (getattr(self, "elevated_obstacle_z", -1.2) - 0.10)
        obstacle_points = obstacle_points[elevated_mask]
        if len(obstacle_points) == 0:
            return

        x_min = self._x_min
        y_min = self._y_min
        num_x = self._num_x
        num_y = self._num_y
        inv_res = self._inv_res

        ox = obstacle_points[:, 0]
        oy = obstacle_points[:, 1]
        oz = obstacle_points[:, 2]

        o_ix = np.clip(((ox - x_min) * inv_res).astype(np.int32), 0, num_x - 1)
        o_iy = np.clip(((oy - y_min) * inv_res).astype(np.int32), 0, num_y - 1)
        obs_grid = np.bincount(o_iy * num_x + o_ix, minlength=self._total_cells).reshape((num_y, num_x))

        sat = np.zeros((num_y + 1, num_x + 1), dtype=np.int32)
        np.cumsum(np.cumsum(obs_grid, axis=0, out=sat[1:, 1:]), axis=1, out=sat[1:, 1:])

        max_z_grid = np.full(self._total_cells, -999.0, dtype=np.float32)
        np.maximum.at(max_z_grid, o_iy * num_x + o_ix, oz)
        max_z_grid = max_z_grid.reshape((num_y, num_x))

        lx = np.fromiter((l.x for l in leaves), dtype=np.float32, count=len(leaves))
        ly = np.fromiter((l.y for l in leaves), dtype=np.float32, count=len(leaves))
        ls = np.fromiter((l.size for l in leaves), dtype=np.float32, count=len(leaves))
        half = ls * 0.5
        eps = 1e-4

        ix0 = np.clip(((lx - half + eps - x_min) * inv_res).astype(np.int32), 0, num_x - 1)
        ix1 = np.clip(((lx + half - eps - x_min) * inv_res).astype(np.int32), 0, num_x - 1)
        iy0 = np.clip(((ly - half + eps - y_min) * inv_res).astype(np.int32), 0, num_y - 1)
        iy1 = np.clip(((ly + half - eps - y_min) * inv_res).astype(np.int32), 0, num_y - 1)

        sums = sat[iy1 + 1, ix1 + 1] - sat[iy0, ix1 + 1] - sat[iy1 + 1, ix0] + sat[iy0, ix0]
        lethal_val = self.lethal_val
        lethal_indices = np.nonzero(sums >= min_obs_points)[0]
        for idx in lethal_indices:
            leaf = leaves[idx]
            cell_obs_max_z = float(np.max(max_z_grid[iy0[idx]:iy1[idx]+1, ix0[idx]:ix1[idx]+1]))
            if cell_obs_max_z > -990.0:
                if leaf.stats is not None:
                    leaf.stats.z_max = max(leaf.stats.z_max, cell_obs_max_z)
                    leaf.stats.delta_z = max(leaf.stats.delta_z, leaf.stats.z_max - leaf.stats.z_min)
            st = leaf.stats
            if st is not None and leaf.cost == 0 and -2.15 <= st.mean_z <= -1.35 and st.delta_z <= 0.15:
                # Flat asphalt road: only mark lethal if substantial obstacle cluster (>= 5 points or min_obs_points if explicitly lowered)
                req_pts = min_obs_points if min_obs_points < 3 else max(min_obs_points, 5)
                if sums[idx] >= req_pts:
                    leaf.cost = lethal_val
                    leaf.is_obstacle = True
                    leaf.is_hazard = True
            else:
                leaf.cost = lethal_val
                leaf.is_obstacle = True
                leaf.is_hazard = True

        # Guard against road cost inflation during dilation:
        for cell in leaves:
            z_mean = float(getattr(cell, 'z_mean', getattr(cell, 'z', -1.68)))
            delta_z = float(getattr(cell, 'delta_z', 0.0))
            z_max = float(getattr(cell, 'z_max', z_mean))

            # Lock ground cells to 0 regardless of adjacent obstacle dilation
            if (-2.20 <= z_mean <= -1.25) and (delta_z <= 0.18) and (z_max <= -1.15):
                cell.cost = 0
                cell.is_obstacle = False
                cell.is_hazard = False




    def apply_dynamic_hazards(self, leaves: List[QuadtreeNode], hazard_circles_or_ellipses: List[dict]):
        """
        Inflates cell costs within predicted dynamic obstacle hazard areas.
        Each hazard item contains {x, y, radius, cost}.
        Accelerated with spatial hashing in < 1.5 ms.
        """
        if not leaves or not hazard_circles_or_ellipses:
            return

        res = 0.50  # Spatial coarse bin size for fast O(1) leaf candidate indexing
        inv_res = 1.0 / res
        grid = {}
        for leaf in leaves:
            k = (int(round(leaf.x * inv_res)), int(round(leaf.y * inv_res)))
            grid.setdefault(k, []).append(leaf)

        for hazard in hazard_circles_or_ellipses:
            hx = hazard["x"]
            hy = hazard["y"]
            r = hazard["radius"]
            hz_cost = hazard.get("cost", self.lethal_val)
            r2 = r * r

            min_gx = int(round((hx - r) * inv_res))
            max_gx = int(round((hx + r) * inv_res))
            min_gy = int(round((hy - r) * inv_res))
            max_gy = int(round((hy + r) * inv_res))

            for gx in range(min_gx, max_gx + 1):
                for gy in range(min_gy, max_gy + 1):
                    cell_leaves = grid.get((gx, gy))
                    if cell_leaves:
                        for leaf in cell_leaves:
                            dx = leaf.x - hx
                            dy = leaf.y - hy
                            if (dx * dx + dy * dy) <= r2:
                                if leaf.cost < hz_cost:
                                    leaf.cost = hz_cost

    def interpolate_beam_gaps(
        self,
        leaves: List[QuadtreeNode],
        bounds=None,
        coarse_res: float = 0.50,
        max_dz: float = 0.05,
    ) -> List[QuadtreeNode]:
        """Method wrapper for planar beam gap dilation/inpainting."""
        from backend.config import BOUNDS
        b = bounds if bounds is not None else BOUNDS
        return interpolate_beam_gaps(leaves, bounds=b, coarse_res=coarse_res, max_dz=max_dz, pool=self._inpaint_pool)


def interpolate_beam_gaps(
    leaves: List[QuadtreeNode],
    bounds=None,
    coarse_res: float = 0.50,
    max_dz: float = 0.05,
    pool: Optional[List[QuadtreeNode]] = None,
) -> List[QuadtreeNode]:
    """
    2D spatial dilation / nearest-neighbor inpainting for coarse planar ground cells (50cm x 50cm).
    Bridges unscanned voids between consecutive concentric LiDAR rings where elevation variance
    is minimal (delta_z < 5cm), preventing black void fissures on flat ground.
    """
    if not leaves:
        return leaves

    from backend.config import BOUNDS
    from backend.mapping.cell_statistics import CellStats
    b = bounds if bounds is not None else BOUNDS

    inv_res = 1.0 / coarse_res
    num_cx = int(np.ceil((b.X_MAX - b.X_MIN) * inv_res))
    num_cy = int(np.ceil((b.Y_MAX - b.Y_MIN) * inv_res))
    x_min = b.X_MIN
    y_min = b.Y_MIN

    grid_present = np.zeros((num_cy, num_cx), dtype=bool)
    flat_mask = np.zeros((num_cy, num_cx), dtype=bool)
    grid_mean_z = np.zeros((num_cy, num_cx), dtype=np.float32)

    for leaf in leaves:
        if leaf.cost > 50 or abs(leaf.size - coarse_res) >= 0.05:
            continue
        gx = int((leaf.x - x_min) * inv_res)
        gy = int((leaf.y - y_min) * inv_res)
        if 0 <= gx < num_cx and 0 <= gy < num_cy:
            grid_present[gy, gx] = True
            st = leaf.stats
            if st is not None and st.delta_z < max_dz:
                flat_mask[gy, gx] = True
                grid_mean_z[gy, gx] = st.mean_z

    unscanned = ~grid_present
    gap_y = np.zeros((num_cy, num_cx), dtype=bool)
    gap_y[1:-1, :] = (
        unscanned[1:-1, :]
        & flat_mask[:-2, :]
        & flat_mask[2:, :]
        & (np.abs(grid_mean_z[:-2, :] - grid_mean_z[2:, :]) < 0.10)
    )

    gap_x = np.zeros((num_cy, num_cx), dtype=bool)
    gap_x[:, 1:-1] = (
        unscanned[:, 1:-1]
        & flat_mask[:, :-2]
        & flat_mask[:, 2:]
        & (np.abs(grid_mean_z[:, :-2] - grid_mean_z[:, 2:]) < 0.10)
    )

    total_gaps = gap_y | gap_x
    gy_coords, gx_coords = np.where(total_gaps)
    if len(gy_coords) == 0:
        return leaves

    new_nodes = []
    if pool is not None:
        n_fill = min(len(gy_coords), len(pool))
        for i in range(n_fill):
            gy, gx = gy_coords[i], gx_coords[i]
            node = pool[i]
            node.x = float(b.X_MIN + (gx + 0.5) * coarse_res)
            node.y = float(b.Y_MIN + (gy + 0.5) * coarse_res)
            node.size = coarse_res
            node.cost = 0
            st = node.stats
            if gap_y[gy, gx]:
                mz = float((grid_mean_z[gy - 1, gx] + grid_mean_z[gy + 1, gx]) * 0.5)
            else:
                mz = float((grid_mean_z[gy, gx - 1] + grid_mean_z[gy, gx + 1]) * 0.5)
            st.z_min = mz - 0.01
            st.z_max = mz + 0.01
            st.delta_z = 0.02
            st.variance = 0.0001
            st.slope = 0.0
            st.point_count = 1
            st.mean_z = mz
            new_nodes.append(node)
    else:
        for gy, gx in zip(gy_coords, gx_coords):
            cx = float(b.X_MIN + (gx + 0.5) * coarse_res)
            cy = float(b.Y_MIN + (gy + 0.5) * coarse_res)
            if gap_y[gy, gx]:
                mz = float((grid_mean_z[gy - 1, gx] + grid_mean_z[gy + 1, gx]) * 0.5)
            else:
                mz = float((grid_mean_z[gy, gx - 1] + grid_mean_z[gy, gx + 1]) * 0.5)
            inpainted_stats = CellStats(
                z_min=mz - 0.01,
                z_max=mz + 0.01,
                delta_z=0.02,
                variance=0.0001,
                slope=0.0,
                point_count=1,
                mean_z=mz,
            )
            new_nodes.append(QuadtreeNode(
                x=cx,
                y=cy,
                size=coarse_res,
                depth=0,
                stats=inpainted_stats,
                cost=0,
                is_leaf=True,
            ))

    leaves.extend(new_nodes)
    return leaves



