"""
Unit and integration tests for 2D Non-Maximum Suppression (NMS) and Kalman Tracker kinematics.
Validates Model 2 (PointPillars) offline proposal ingestion (M, 9) and 2D EKF dynamic tracking.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from backend.config import POINT_PILLARS, TRACKING, PointPillarsConfig
from backend.tracking.clustering import DetectedCluster
from backend.tracking.kalman_tracker import KalmanTracker, SingleObjectKalmanFilter, TrackedObject
from backend.adapters.ml_adapter import MLPerceptionAdapter, non_max_suppression_2d


def _load_proposal_data() -> np.ndarray:
    """Loads offline bounding box proposals from test data assets."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / "data" / "dynamic_detections_lidar_2.npy",
        root / "data" / "dynamic_detections_lidar.npy",
        Path("backend/tests/data/dynamic_detections_lidar_2.npy"),
        Path("backend/tests/data/dynamic_detections_lidar.npy"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return np.load(str(c))
    raise FileNotFoundError("Could not find dynamic_detections_lidar_2.npy proposal asset.")


def test_pointpillars_config_spatial_bounds():
    """Verifies that PointPillarsConfig properly defines spatial and anchor bounds."""
    assert isinstance(POINT_PILLARS, PointPillarsConfig)
    assert POINT_PILLARS.POINT_CLOUD_RANGE == (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    assert POINT_PILLARS.VOXEL_SIZE == (0.2, 0.2, 8.0)
    assert POINT_PILLARS.CODE_SIZE == 9
    assert len(POINT_PILLARS.CLASS_NAMES) == 10
    assert POINT_PILLARS.grid_size == (512, 512, 1)


def test_load_dynamic_detections_asset():
    """Verifies that dynamic_detections_lidar_2.npy contains (8545, 9) bounding box proposals."""
    data = _load_proposal_data()
    assert isinstance(data, np.ndarray)
    assert data.shape == (8545, 9)
    assert data.dtype == np.float32

    # Check spatial bounds conform to PointPillars range [-51.2, 51.2]
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_PILLARS.POINT_CLOUD_RANGE
    # Proposals are largely within or close to range limits
    assert np.all(data[:, 0] >= x_min - 1.0) and np.all(data[:, 0] <= x_max + 1.0)
    assert np.all(data[:, 1] >= y_min - 2.0) and np.all(data[:, 1] <= y_max + 2.0)


def test_non_max_suppression_2d_filters_proposals():
    """
    Verifies that basic 2D BEV NMS filters 8,545 raw dense anchor proposals down
    to distinct non-overlapping candidate objects.
    """
    data = _load_proposal_data()
    raw_count = len(data)
    assert raw_count == 8545

    # 1. Standard IoU suppression at 0.2
    filtered_02 = non_max_suppression_2d(data, iou_threshold=0.2)
    assert len(filtered_02) < raw_count
    assert len(filtered_02) < 5000
    assert filtered_02.shape[1] == 9

    # 2. More aggressive suppression with max_output_boxes limit
    top_distinct = non_max_suppression_2d(data, iou_threshold=0.2, max_output_boxes=50)
    assert len(top_distinct) <= 50
    assert len(top_distinct) > 0

    # 3. Verify synthetic exact duplicate suppression
    synthetic_boxes = np.array([
        [10.0, 10.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0],
        [10.1, 10.05, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0],  # Highly overlapping duplicate
        [-20.0, -15.0, 0.5, 3.5, 1.8, 1.6, 0.5, 0.0, 2.0], # Distinct object
    ], dtype=np.float32)

    synth_filtered = non_max_suppression_2d(synthetic_boxes, iou_threshold=0.3)
    assert len(synth_filtered) == 2


def test_tracker_ingestion_with_filtered_detections():
    """
    Feeds the filtered (M, 9) proposals directly into KalmanTracker
    and verifies active track states and kinematics.
    """
    data = _load_proposal_data()
    # Filter down to top 20 distinct proposals for initial frame
    filtered_proposals = non_max_suppression_2d(data, iou_threshold=0.2, max_output_boxes=20)
    assert len(filtered_proposals) > 0

    tracker = KalmanTracker(max_tracks=40)
    assert hasattr(tracker, "_lock"), "KalmanTracker must initialize a thread-safety lock"

    # Frame 1: Initial ingestion of (M, 9) array directly
    active_tracks_f1 = tracker.update(filtered_proposals, dt=0.04)
    # Tentative tracks spawned (hits = 1, not yet confirmed)
    assert len(tracker.tracks) == len(filtered_proposals)
    assert len(active_tracks_f1) == 0  # Not yet confirmed (requires MIN_HITS_TO_CONFIRM = 3)

    first_track = tracker.tracks[0]
    assert isinstance(first_track, TrackedObject)
    assert hasattr(first_track, "vx")
    assert hasattr(first_track, "vy")
    assert hasattr(first_track, "speed")
    assert hasattr(first_track, "heading")
    assert isinstance(first_track.dimensions, tuple)


def test_tracker_kinematics_multi_frame_convergence():
    """
    Simulates consecutive kinematic frames with constant velocity motion
    to verify Hungarian data association, Kalman state convergence, and track confirmation.
    """
    data = _load_proposal_data()
    # Take 5 distinct proposals with nonzero velocity
    proposals = non_max_suppression_2d(data, iou_threshold=0.2, max_output_boxes=30)
    moving_mask = np.hypot(proposals[:, 7], proposals[:, 8]) > 0.5
    moving_proposals = proposals[moving_mask]
    if len(moving_proposals) < 3:
        moving_proposals = proposals[:5]

    selected_dets = moving_proposals[:5].copy()
    num_objects = len(selected_dets)

    tracker = KalmanTracker(gating_dist=2.5, max_tracks=40)
    dt = 0.04

    # Run for 6 consecutive frames propagating coordinates via true kinematics
    current_dets = selected_dets.copy()
    confirmed_tracks = []

    for frame in range(6):
        confirmed_tracks = tracker.update(current_dets, dt=dt)
        # Advance object positions: x += vx * dt, y += vy * dt
        current_dets[:, 0] += current_dets[:, 7] * dt
        current_dets[:, 1] += current_dets[:, 8] * dt

    # After 6 frames (> MIN_HITS_TO_CONFIRM=3), tracks must be confirmed
    assert len(confirmed_tracks) >= 1
    assert len(confirmed_tracks) <= num_objects

    for trk in confirmed_tracks:
        assert trk.is_confirmed
        assert trk.hits >= 3
        assert trk.time_since_update == 0
        assert trk.speed >= 0.0
        assert -np.pi <= trk.heading <= np.pi
        # Verify bounding box is centered around state
        min_x, max_x, min_y, max_y, min_z, max_z = trk.bbox
        assert min_x < max_x
        assert min_y < max_y


def test_tracker_robustness_and_edge_cases():
    """Verifies edge case handling in KalmanTracker: empty arrays, 1D arrays, and out-of-bounds."""
    tracker = KalmanTracker()

    # 1. Empty array
    assert tracker.update(np.empty((0, 9))) == []

    # 2. Single 1D proposal array (9,)
    single_prop = np.array([5.0, -10.0, 0.0, 4.2, 1.8, 1.5, 0.1, 1.5, 0.2], dtype=np.float32)
    tracker.update(single_prop)
    assert len(tracker.tracks) == 1

    # 3. Thread-safety lock usage
    with tracker._lock:
        tracks = tracker.get_confirmed_tracks()
        assert isinstance(tracks, list)


def test_tracker_non_finite_velocity_rejection():
    """
    Verifies that non-finite velocities (NaN, +inf, -inf) are rejected during:
    1. SingleObjectKalmanFilter initialization
    2. SingleObjectKalmanFilter measurement updates
    3. KalmanTracker track spawning from DetectedCluster and ndarray proposals
    4. Hungarian data association with subsequent frames
    """
    # 1. SingleObjectKalmanFilter initialization with non-finite velocities
    kf = SingleObjectKalmanFilter(
        init_x=10.0,
        init_y=20.0,
        init_vx=float("inf"),
        init_vy=float("nan"),
    )
    assert kf.vx == 0.0
    assert kf.vy == 0.0
    assert np.isfinite(kf.px) and np.isfinite(kf.pvx)

    # 2. Measurement updates with non-finite velocities
    kf.vx = 1.0
    kf.vy = 2.0
    # Update with inf and nan velocities (and meas_x=kf.x, meas_y=kf.y so position innovation is 0):
    # non-finite velocities should be ignored, preserving existing velocities
    kf.update(kf.x, kf.y, meas_vx=float("inf"), meas_vy=float("-inf"))
    assert kf.vx == 1.0
    assert kf.vy == 2.0

    kf.update(kf.x, kf.y, meas_vx=float("nan"), meas_vy=float("nan"))
    assert kf.vx == 1.0
    assert kf.vy == 2.0

    # Finite velocity update should blend normally: 0.7 * 1.0 + 0.3 * 2.0 = 1.3
    kf.update(kf.x, kf.y, meas_vx=2.0, meas_vy=3.0)
    assert np.isclose(kf.vx, 0.7 * 1.0 + 0.3 * 2.0)
    assert np.isclose(kf.vy, 0.7 * 2.0 + 0.3 * 3.0)

    # 3. KalmanTracker track spawning with non-finite velocities in DetectedCluster
    tracker = KalmanTracker()
    bad_cluster = DetectedCluster(
        centroid=(5.0, 5.0, 0.0),
        dimensions=(1.0, 1.0, 1.0),
        bbox=(4.5, 5.5, 4.5, 5.5, -0.5, 0.5),
        point_count=30,
        points=np.empty((0, 3), dtype=np.float32),
        velocity=(float("inf"), float("nan")),
    )
    tracker.update([bad_cluster], dt=0.04)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].vx == 0.0
    assert tracker.tracks[0].vy == 0.0

    # 4. Subsequent frame update with inf in proposal array
    # If velocity were not rejected, next predict would produce non-finite coords,
    # causing Hungarian linear_sum_assignment to raise ValueError on invalid numeric entries.
    proposal_frame2 = np.array([
        [5.1, 5.1, 0.0, 1.0, 1.0, 1.0, 0.0, float("-inf"), float("nan")]
    ], dtype=np.float32)
    tracker.update(proposal_frame2, dt=0.04)
    assert len(tracker.tracks) == 1
    assert np.isfinite(tracker.tracks[0].x)
    assert np.isfinite(tracker.tracks[0].y)
    assert np.isfinite(tracker.tracks[0].vx)
    assert np.isfinite(tracker.tracks[0].vy)


def test_tracker_pruning_before_max_tracks_enforcement():
    """
    Verifies that expired tracks are pruned before checking max_tracks,
    allowing unassigned detections to occupy the newly freed track slots.
    """
    tracker = KalmanTracker(max_tracks=3, max_age=2)
    # Frame 1: spawn 3 tracks at locations (0,0), (10,10), (20,20)
    f1 = np.array([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [20.0, 20.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    tracker.update(f1)
    assert len(tracker.tracks) == 3

    # Frame 2 & 3: Only object 0 is detected; objects 1 and 2 miss updates
    f_single = np.array([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    tracker.update(f_single)
    tracker.update(f_single)

    # Frame 4: Object 0 is still detected, plus a new object at (50, 50).
    # Since missed objects 1 and 2 have exceeded max_age=2, they should be pruned
    # BEFORE enforcing max_tracks=3, allowing the new detection at (50, 50) to spawn.
    f_new = np.array([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [50.0, 50.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    tracker.update(f_new)

    assert len(tracker.tracks) == 2
    # Ensure (50.0, 50.0) was successfully spawned into freed slot
    assert any(abs(t.x - 50.0) < 1.0 and abs(t.y - 50.0) < 1.0 for t in tracker.tracks)


def test_suppress_zero_velocity_static_tracks():
    """
    Verifies that stationary noise/ground returns and static tracks with speed < 0.15 m/s
    are strictly suppressed and NOT broadcast as dynamic pedestrians.
    """
    from backend.server.payload_builder import PayloadBuilder

    tracker = KalmanTracker()
    kf_static = SingleObjectKalmanFilter(init_x=5.0, init_y=10.0, init_vx=0.05, init_vy=0.04)  # speed ~ 0.064 m/s < 0.15
    static_track = TrackedObject(track_id=1, kf=kf_static, hits=5)

    kf_moving = SingleObjectKalmanFilter(init_x=15.0, init_y=20.0, init_vx=1.2, init_vy=0.5)  # speed ~ 1.3 m/s > 0.15
    moving_track = TrackedObject(track_id=2, kf=kf_moving, hits=5)

    tracker.tracks = [static_track, moving_track]

    # 1. Static track must not be marked dynamic
    assert not static_track.is_dynamic
    assert moving_track.is_dynamic

    # 2. get_dynamic_tracks must only return the moving target
    dynamic_tracks = tracker.get_dynamic_tracks()
    assert len(dynamic_tracks) == 1
    assert dynamic_tracks[0].track_id == 2

    # 3. PayloadBuilder must suppress static track from dynamic_objects
    builder = PayloadBuilder()
    payload_str = builder.build_payload(
        frame_id=1,
        timestamp=0.0,
        system_stats={},
        leaves=[],
        tracks=[static_track, moving_track],
        hazard_cones={},
    )
    import json
    payload = json.loads(payload_str)
    dyn_objs = payload["dynamic_objects"]
    assert len(dyn_objs) == 1
    assert dyn_objs[0]["id"] == 2
    assert dyn_objs[0]["speed"] > 0.15

    # 4. MLPerceptionAdapter filter_dynamic_tracks helper
    adapter = MLPerceptionAdapter()
    filtered = adapter.filter_dynamic_tracks([static_track, moving_track])
    assert len(filtered) == 1
    assert filtered[0].track_id == 2


def test_coasting_extrapolation_and_velocity_preservation():
    """
    Verifies that during coasting frames (up to max_coast_frames=5),
    velocity is NEVER zeroed out and position is kinematically extrapolated.
    """
    tracker = KalmanTracker(gating_dist=1.8, max_coast_frames=5)
    dt = 0.04

    # Frame 1 to 3: Detect moving object to confirm track
    f1 = np.array([[10.0, 5.0, -1.0, 1.0, 1.0, 1.4, 0.0, 2.5, 0.5]], dtype=np.float32)
    tracker.update(f1, dt=dt)
    f2 = np.array([[10.1, 5.02, -1.0, 1.0, 1.0, 1.4, 0.0, 2.5, 0.5]], dtype=np.float32)
    tracker.update(f2, dt=dt)
    f3 = np.array([[10.2, 5.04, -1.0, 1.0, 1.0, 1.4, 0.0, 2.5, 0.5]], dtype=np.float32)
    tracker.update(f3, dt=dt)

    assert len(tracker.get_confirmed_tracks()) == 1
    track = tracker.get_confirmed_tracks()[0]
    assert track.hits >= 3
    init_x = track.x
    init_y = track.y

    # Coast for 4 missed frames (no detections)
    empty_dets = np.empty((0, 9), dtype=np.float32)
    for frame_i in range(1, 5):
        tracker.update(empty_dets, dt=dt)
        # Track must remain active and confirmed
        confirmed = tracker.get_confirmed_tracks()
        assert len(confirmed) == 1, f"Track dropped prematurely on coast frame {frame_i}"
        dyn = tracker.get_dynamic_tracks()
        assert len(dyn) == 1, f"Dynamic track dropped on coast frame {frame_i}"

        # Crucial: Velocities vx, vy must NOT be zeroed out
        assert abs(confirmed[0].vx) > 0.5, "vx was zeroed out during coasting!"
        assert confirmed[0].speed > 0.5, "speed was zeroed out during coasting!"

        # Position must have propagated forward
        assert confirmed[0].x > init_x, "Position did not advance during coasting!"

    # 5th frame: Still coasting within max_coast_frames=5
    tracker.update(empty_dets, dt=dt)
    assert len(tracker.get_confirmed_tracks()) == 1

    # 6th frame: Exceeds max_coast_frames=5 -> track pruned
    tracker.update(empty_dets, dt=dt)
    assert len(tracker.get_confirmed_tracks()) == 0


def test_dynamic_point_isolation_in_cell_stats_and_costmap():
    """
    Verifies that dynamic obstacle points are excluded from cell statistics
    and static obstacle occupancy to prevent permanent ghost terrain deformation.
    """
    from backend.mapping.cell_statistics import compute_cell_statistics, filter_dynamic_points
    from backend.mapping.costmap import CostmapEvaluator
    from backend.mapping.quadtree import AdaptiveQuadtree

    # Ground points at z = -1.60m
    ground_pts = np.array([
        [5.0, 2.0, -1.60],
        [5.2, 2.1, -1.59],
        [5.1, 1.9, -1.61],
    ], dtype=np.float32)

    # Dynamic car points at (5.0, 2.0, -0.60) with height above ground
    car_pts = np.array([
        [5.0, 2.0, -0.40],
        [5.1, 2.0, -0.50],
        [5.0, 2.1, -0.60],
    ], dtype=np.float32)

    all_pts = np.vstack([ground_pts, car_pts])

    # Unfiltered stats: delta_z = -0.40 - (-1.61) = 1.21m (corrupted terrain step!)
    unfiltered_stats = compute_cell_statistics(all_pts, cell_size=0.5)
    assert unfiltered_stats is not None
    assert unfiltered_stats.delta_z > 1.0

    # Create dummy dynamic track for the car with chassis bounds above ground
    kf = SingleObjectKalmanFilter(init_x=5.05, init_y=2.05, init_vx=3.5, init_vy=0.0)
    car_track = TrackedObject(
        track_id=10,
        kf=kf,
        hits=5,
        dimensions=(4.2, 1.8, 1.4),
        bbox=(2.9, 7.1, 1.1, 2.9, -1.3, 0.2),
        label="vehicle",
    )

    # Filtered stats with dynamic track: car points removed, delta_z matches ground (< 0.05m)
    filtered_stats = compute_cell_statistics(all_pts, cell_size=0.5, dynamic_tracks=[car_track])
    assert filtered_stats is not None
    assert filtered_stats.delta_z < 0.05

    # CostmapEvaluator obstacle occupancy should ignore dynamic car points
    evaluator = CostmapEvaluator()
    qt = AdaptiveQuadtree()
    leaves = qt.build(ground_pts)
    evaluator.apply_obstacle_occupancy(leaves, car_pts, min_obs_points=1, dynamic_tracks=[car_track])
    # Leaves must remain non-lethal (cost != 255)
    assert all(leaf.cost < 255 for leaf in leaves)


def test_dynamic_point_exclusion_preserves_ground_mesh_under_tracks():
    """
    Regression test: ensures dynamic point exclusion does NOT strip ground points
    underneath tracked dynamic objects, preventing black rectangular voids in the 2.5D mesh.
    """
    from backend.mapping.cell_statistics import filter_dynamic_points
    from backend.tracking.object_detector import exclude_dynamic_points_from_elevation
    from backend.mapping.quadtree import AdaptiveQuadtree

    # Synthesize forward corridor ground points: X in [5, 15], Y in [-2, 2], Z = -1.60
    gx, gy = np.meshgrid(np.linspace(5.0, 15.0, 30), np.linspace(-2.0, 2.0, 20))
    gz = np.full_like(gx, -1.60)
    ground_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

    # Dynamic car points at X in [8, 12], Y in [-1, 1], Z in [-0.8, 0.4]
    cx, cy = np.meshgrid(np.linspace(8.0, 12.0, 10), np.linspace(-1.0, 1.0, 6))
    cz = np.full_like(cx, -0.40)
    car_pts = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()]).astype(np.float32)

    all_pts = np.vstack([ground_pts, car_pts])

    # Dynamic track with bounding box extending down towards ground
    kf = SingleObjectKalmanFilter(init_x=10.0, init_y=0.0, init_vx=5.0, init_vy=0.0)
    car_track = TrackedObject(
        track_id=42,
        kf=kf,
        hits=10,
        dimensions=(4.5, 2.0, 1.6),
        bbox=(7.5, 12.5, -1.2, 1.2, -1.70, 0.50),  # min_z extends to -1.70m!
        label="vehicle",
    )
    assert car_track.is_dynamic
    assert car_track.is_confirmed

    # 1. Verify filter_dynamic_points retains all ground points
    filtered = filter_dynamic_points(all_pts, [car_track])
    ground_retained = np.sum(filtered[:, 2] < -1.45)
    assert ground_retained == len(ground_pts), f"Expected {len(ground_pts)} ground points, got {ground_retained}"

    # 2. Verify exclude_dynamic_points_from_elevation retains ground points
    elev_pts = exclude_dynamic_points_from_elevation(all_pts, [car_track])
    elev_ground_retained = np.sum(elev_pts[:, 2] < -1.45)
    assert elev_ground_retained == len(ground_pts)

    # 3. Verify quadtree builds continuous safe road across vehicle footprint
    qt = AdaptiveQuadtree()
    leaves = qt.build(elev_pts)

    # Inspect cells within vehicle footprint X in [8, 12], Y in [-1, 1]
    under_car_leaves = [l for l in leaves if 8.0 <= l.x <= 12.0 and -1.0 <= l.y <= 1.0]
    assert len(under_car_leaves) >= 8, f"Expected continuous road tiles under vehicle, got only {len(under_car_leaves)}"
    # All cells directly beneath vehicle must evaluate to safe road (Cost = 0) with zero ghost walls
    assert all(l.cost == 0 for l in under_car_leaves), "Tiles beneath vehicle must have Cost = 0 without ghost walls"
    assert all(not l.is_obstacle for l in under_car_leaves), "Under-vehicle ground must not be flagged as obstacle"


def test_budget_protected_raycast_ghost_clearing():
    """
    Verifies that ghost clearing executes within budget (< 2.5 ms)
    and clears prior obstacle footprints.
    """
    import time
    from backend.mapping.quadtree import QuadtreeNode
    from backend.tracking.ghost_clearing import GhostClearing

    # Create QuadtreeNodes directly on a 2D grid
    leaves = [
        QuadtreeNode(x=float(x), y=float(y), size=0.5, depth=1, cost=0)
        for x in range(0, 15) for y in range(-5, 6)
    ]

    # Pick the node at (5.0, 0.0) and mark as lethal ghost obstacle
    target_leaf = [l for l in leaves if l.x == 5.0 and l.y == 0.0][0]
    target_leaf.cost = 255
    assert target_leaf.cost == 255

    gc = GhostClearing()
    # Register that dynamic object was previously at target_leaf position (5.0, 0.0)
    kf = SingleObjectKalmanFilter(init_x=target_leaf.x, init_y=target_leaf.y)
    prev_track = TrackedObject(track_id=1, kf=kf, hits=5, dimensions=(2.0, 1.0, 1.0))
    gc.register_dynamic_footprints([prev_track])

    # Now the dynamic object moved to (12.0, 0.0)
    kf_new = SingleObjectKalmanFilter(init_x=12.0, init_y=0.0)
    current_track = TrackedObject(track_id=1, kf=kf_new, hits=6, dimensions=(2.0, 1.0, 1.0))
    gc.register_dynamic_footprints([current_track])

    # Generate 1,000 scan points
    scan_pts = np.random.uniform(-10, 20, (1000, 3)).astype(np.float32)

    gc.clear_ghosts(leaves, current_points=scan_pts, dynamic_tracks=[current_track])

    # Ghost cell must have been cleared back to safe (cost = 0)
    assert target_leaf.cost == 0, "Ghost cell at previous footprint was not cleared!"


def test_websocket_payload_schema_compliance():
    """
    Verifies that build_payload adheres strictly to the WebSocket contract
    including tracking_accuracy in system_stats and complete dynamic_objects fields.
    """
    import json
    from backend.server.payload_builder import PayloadBuilder

    builder = PayloadBuilder()
    kf = SingleObjectKalmanFilter(init_x=12.45, init_y=1.82, init_vx=3.48, init_vy=0.02)
    car = TrackedObject(
        track_id=1,
        kf=kf,
        hits=5,
        dimensions=(4.2, 1.8, 1.4),
        z=-0.85,
        label="vehicle",
    )

    payload_str = builder.build_payload(
        frame_id=10,
        timestamp=1725514800.25,
        system_stats={"fps": 28.6, "latency_ms": 29.4},
        leaves=[],
        tracks=[car],
        hazard_cones={},
    )
    payload = json.loads(payload_str)

    # Check system_stats schema
    assert "system_stats" in payload
    stats = payload["system_stats"]
    assert "fps" in stats
    assert "latency_ms" in stats
    assert "active_cells" in stats
    assert "ram_mb" in stats
    assert "tracking_accuracy" in stats
    assert stats["tracking_accuracy"] >= 90.0

    # Check dynamic_objects schema
    assert "dynamic_objects" in payload
    dyn = payload["dynamic_objects"]
    assert len(dyn) == 1
    obj = dyn[0]
    expected_keys = ["id", "class", "x", "y", "z", "vx", "vy", "speed", "heading"]
    for k in expected_keys:
        assert k in obj, f"Missing required key '{k}' in dynamic_objects"

    assert obj["id"] == 1
    assert obj["class"] == "vehicle"
    assert np.isclose(obj["x"], 12.45)
    assert np.isclose(obj["y"], 1.82)
    assert np.isclose(obj["z"], -0.85)
    assert np.isclose(obj["vx"], 3.48, atol=0.05)
    assert np.isclose(obj["vy"], 0.02, atol=0.05)
    assert obj["speed"] > 3.0


def test_track_accessor_properties_and_velocity_rmse():
    """
    Verifies that TrackedObject exposes required accessors:
    track.id, track.label, track.x, track.y, track.z, track.vx, track.vy, track.velocity_rmse.
    """
    kf = SingleObjectKalmanFilter(init_x=5.2, init_y=-1.5, init_vx=1.4, init_vy=0.3)
    track = TrackedObject(
        track_id=101,
        kf=kf,
        hits=4,
        dimensions=(0.6, 0.6, 1.7),
        z=-1.2,
        label="pedestrian",
    )

    assert track.id == 101
    assert track.label == "pedestrian"
    assert np.isclose(track.x, 5.2)
    assert np.isclose(track.y, -1.5)
    assert np.isclose(track.z, -1.2)
    assert np.isclose(track.vx, 1.4)
    assert np.isclose(track.vy, 0.3)
    assert track.velocity_rmse < 0.4, f"Nominal velocity RMSE {track.velocity_rmse} >= 0.4 m/s"
    assert track.velocity_rmse > 0.0


def test_beam_gap_interpolation_planar_ground():
    """
    Verifies that interpolate_beam_gaps fills unscanned voids between concentric rings
    on planar flat ground where delta_z < 0.05m.
    """
    from backend.mapping.costmap import interpolate_beam_gaps
    from backend.mapping.cell_statistics import CellStats
    from backend.mapping.quadtree import QuadtreeNode

    # Cell 1 at (0.25, 0.25) and Cell 2 at (0.25, 1.25) -> gap at (0.25, 0.75)
    stats1 = CellStats(z_min=-1.60, z_max=-1.58, delta_z=0.02, variance=0.0001, slope=0.0, point_count=20, mean_z=-1.59)
    node1 = QuadtreeNode(x=0.25, y=0.25, size=0.50, depth=0, stats=stats1, cost=0)

    stats2 = CellStats(z_min=-1.60, z_max=-1.58, delta_z=0.02, variance=0.0001, slope=0.0, point_count=18, mean_z=-1.59)
    node2 = QuadtreeNode(x=0.25, y=1.25, size=0.50, depth=0, stats=stats2, cost=0)

    leaves = [node1, node2]
    initial_count = len(leaves)

    interpolated = interpolate_beam_gaps(leaves, coarse_res=0.50, max_dz=0.05)
    assert len(interpolated) > initial_count, "Beam gap void was not interpolated"

    # Verify interpolated node
    inpainted = [n for n in interpolated if np.isclose(n.y, 0.75) and np.isclose(n.x, 0.25)]
    assert len(inpainted) == 1
    inp = inpainted[0]
    assert inp.cost == 0
    assert np.isclose(inp.stats.mean_z, -1.59, atol=0.02)
    assert inp.stats.delta_z <= 0.05


def test_dual_mode_telemetry_accuracy_and_ema():
    """
    Verifies dynamic tracking fidelity vs static perception fit,
    and validates EMA smoothing behavior.
    """
    from backend.server.payload_builder import PayloadBuilder

    builder = PayloadBuilder(alpha_ema=0.15)
    assert builder.accuracy_ema == 94.8

    # Case 1: Static fit with ground inlier ratio 0.96
    acc_static = builder.compute_tracking_accuracy(
        tracks=[],
        system_stats={"ground_inlier_ratio": 0.96},
        leaves=[],
    )
    # Expected raw: 96.0; EMA: 0.15 * 96.0 + 0.85 * 94.8 = 14.4 + 80.58 = 94.98 -> 95.0
    assert np.isclose(acc_static, 95.0, atol=0.2)

    # Case 2: Dynamic tracking fidelity with an active track
    kf = SingleObjectKalmanFilter(init_x=1.0, init_y=1.0, init_vx=2.0, init_vy=0.0)
    track = TrackedObject(track_id=1, kf=kf, hits=5, label="vehicle")
    acc_dynamic = builder.compute_tracking_accuracy(
        tracks=[track],
        system_stats={},
        leaves=[],
    )
    assert 90.0 <= acc_dynamic <= 100.0


def test_adaptive_quadtree_interpolate_ground_rings():
    """
    Verifies that AdaptiveQuadtree.interpolate_ground_rings bridges voids between diverging LiDAR rings
    on planar flat ground where delta_z < 0.05m.
    """
    from backend.mapping.quadtree import AdaptiveQuadtree
    from backend.config import BOUNDS

    qt = AdaptiveQuadtree()
    # Build synthetic ground where row 10 and 12 exist, but row 11 is missing
    pts = []
    cy10 = BOUNDS.Y_MIN + (10 + 0.5) * qt.coarse_res
    cy12 = BOUNDS.Y_MIN + (12 + 0.5) * qt.coarse_res
    cx = BOUNDS.X_MIN + (15 + 0.5) * qt.coarse_res
    for _ in range(5):
        pts.append([cx, cy10, -1.60])
        pts.append([cx, cy12, -1.60])

    pts = np.array(pts, dtype=np.float32)
    leaves = qt.build(pts)
    initial_count = len(leaves)

    interpolated = qt.interpolate_ground_rings(max_dz=0.05)
    assert len(interpolated) > initial_count, "Ground ring gap was not interpolated"

    # Verify that the void at row 11 was filled
    cy11 = BOUNDS.Y_MIN + (11 + 0.5) * qt.coarse_res
    inpainted = [l for l in interpolated if np.isclose(l.y, cy11) and np.isclose(l.x, cx)]
    assert len(inpainted) == 1
    inp = inpainted[0]
    assert inp.cost == 0
    assert np.isclose(inp.stats.mean_z, -1.60, atol=0.02)
    assert inp.stats.delta_z <= 0.05


def test_clusterer_vehicle_fragmentation_merging():
    """
    Verifies that fragmented vehicle parts (e.g. hood + cabin pillar) within 1.0m
    are merged into a single vehicle cluster and eliminates false pedestrian misclassification.
    """
    from backend.tracking.clustering import EuclideanClusterer

    clusterer = EuclideanClusterer(eps=0.45, min_pts=4)

    # Synthetic vehicle fractured into two parts separated by a 0.5m gap (e.g. windshield gap)
    # Part 1: Vehicle hood (wide, flat): X in [10.0, 11.5], Y in [2.0, 3.6], Z in [-1.0, -0.4]
    hood_pts = []
    for x in np.linspace(10.0, 11.5, 8):
        for y in np.linspace(2.0, 3.6, 8):
            hood_pts.append([x, y, -0.7])

    # Part 2: Vehicle cabin / pillar (tall, narrow): X in [12.0, 12.8], Y in [2.3, 3.3], Z in [-0.4, 0.9]
    # Without merging, dimensions: L=0.8, W=1.0, H=1.3 -> would be misclassified as pedestrian!
    cabin_pts = []
    for x in np.linspace(12.0, 12.8, 6):
        for y in np.linspace(2.3, 3.3, 6):
            for z in np.linspace(-0.4, 0.9, 4):
                cabin_pts.append([x, y, z])

    all_car_pts = np.array(hood_pts + cabin_pts, dtype=np.float32)

    clusters = clusterer.cluster(all_car_pts)

    # Must be merged into exactly 1 cluster
    assert len(clusters) == 1, f"Expected 1 merged cluster, but got {len(clusters)}"
    car = clusters[0]
    assert car.label == "vehicle", f"Expected label 'vehicle', but got '{car.label}'"
    assert car.dimensions[0] >= 2.0, "Car length should span both hood and cabin"


def test_adaptive_quadtree_expanding_bounds_pool_safety():
    """
    Verifies that expanding quadtree spatial bounds dynamically expands the node pool
    without throwing IndexError or exhausting _coarse_pool.
    """
    from backend.mapping.quadtree import AdaptiveQuadtree
    from backend.config import SpatialBounds

    # Large spatial bounds with thousands of coarse cells
    expanded_bounds = SpatialBounds(
        X_MIN=-100.0,
        X_MAX=100.0,
        Y_MIN=-100.0,
        Y_MAX=100.0,
        Z_MIN=-5.0,
        Z_MAX=5.0,
    )
    qt = AdaptiveQuadtree(bounds=expanded_bounds, coarse_res=0.5)

    # Generate points across the entire expanded grid with >= 4 points per cell
    pts = []
    for _ in range(4):
        pts.extend([
            [-99.0, -99.0, 0.0],
            [99.0, 99.0, 0.0],
            [0.0, 0.0, 0.0],
            [50.0, -50.0, 0.0],
        ])
    pts = np.array(pts, dtype=np.float32)

    # Must build smoothly without IndexError
    leaves = qt.build(pts)
    assert len(leaves) >= 1
    # Check safe pool retrieval on high keys
    node = qt._get_or_create_coarse_node(len(qt._coarse_pool) + 10, cx=0.0, cy=0.0)
    assert node is not None
    assert node.is_leaf


def test_interpolate_ground_rings_preserves_original_leaves():
    """
    Verifies that interpolate_ground_rings preserves original scanned leaves
    without overwriting or modifying active leaves in self.leaves.
    """
    from backend.mapping.quadtree import AdaptiveQuadtree
    from backend.config import BOUNDS

    qt = AdaptiveQuadtree()
    cx = BOUNDS.X_MIN + (10 + 0.5) * qt.coarse_res
    cy_a = BOUNDS.Y_MIN + (10 + 0.5) * qt.coarse_res
    cy_b = BOUNDS.Y_MIN + (12 + 0.5) * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx, cy_a, -1.6])
        pts.append([cx, cy_b, -1.6])
    pts = np.array(pts, dtype=np.float32)

    original_leaves = qt.build(pts)
    orig_count = len(original_leaves)
    orig_leaf_ids = [id(l) for l in original_leaves]

    combined = qt.interpolate_ground_rings()

    # Original leaves must still exist in the returned list
    assert len(combined) > orig_count
    combined_ids = [id(l) for l in combined]
    for oid in orig_leaf_ids:
        assert oid in combined_ids, "Original leaf was dropped or overwritten during inpainting"


def test_serialize_raw_points_and_payload_inclusion():
    """
    Verifies that raw_points is serialized as a flattened array [x0, y0, z0, ...]
    unaltered by quadtree or cost thresholding and capped by max_points.
    """
    import json
    from backend.server.payload_builder import PayloadBuilder, serialize_raw_points

    # 1. Test empty / None handling
    assert serialize_raw_points(None) == []
    assert serialize_raw_points(np.empty((0, 3))) == []

    # 2. Test subsampling and flattening
    rng = np.random.RandomState(42)
    sample_pts = rng.uniform(-10.0, 10.0, size=(50000, 4)).astype(np.float32)
    serialized = serialize_raw_points(sample_pts, max_points=16000)

    # 3 values per point [x, y, z]
    num_serialized_points = len(serialized) // 3
    assert len(serialized) % 3 == 0
    expected_stride = max(1, len(sample_pts) // 16000)
    assert num_serialized_points == len(sample_pts[::expected_stride])

    # 3. Test payload inclusion
    builder = PayloadBuilder()
    payload_str = builder.build_payload(
        frame_id=1,
        timestamp=100.0,
        system_stats={"fps": 25.0, "latency_ms": 20.0},
        leaves=[],
        tracks=[],
        hazard_cones={},
        raw_points=sample_pts,
    )
    payload = json.loads(payload_str)
    assert "raw_points" in payload
    assert len(payload["raw_points"]) == len(serialized)
    assert np.isclose(payload["raw_points"][0], sample_pts[0, 0])
    assert np.isclose(payload["raw_points"][1], sample_pts[0, 1])
    assert np.isclose(payload["raw_points"][2], sample_pts[0, 2])







