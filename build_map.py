import sys
import os
import open3d as o3d
import numpy as np

DATA_DIR = "data/lidar-warehouse-dataset/data"
LOG_DIR = "logs"

VOXEL_SIZE = 0.15       # downsampling resolution (meters)
ICP_DISTANCE = 0.5      # max correspondence distance for ICP
NORMAL_RADIUS = 0.5     # radius for normal estimation
Z_CLAMP = 0.3           # max allowed Z drift per frame (sensor height shouldn't move much)


def load_and_downsample(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    return pcd


def constrain_to_ground_plane(T):
    """Keep only yaw rotation and x/y translation. Clamp Z drift."""
    R = T[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    cy, sy = np.cos(yaw), np.sin(yaw)
    T_clean = np.eye(4)
    T_clean[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T_clean[0, 3] = T[0, 3]
    T_clean[1, 3] = T[1, 3]
    T_clean[2, 3] = np.clip(T[2, 3], -Z_CLAMP, Z_CLAMP)
    return T_clean


def icp(source, target, init):
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=ICP_DISTANCE,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return constrain_to_ground_plane(result.transformation)


def colorize_by_height(pcd):
    points = np.asarray(pcd.points)
    z = points[:, 2]
    z_norm = (z - z.min()) / (np.ptp(z) + 1e-9)
    colors = np.stack([z_norm, 0.4 * np.ones_like(z_norm), 1 - z_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".pcd")
    ])[:n]

    if not files:
        print(f"No PCD files found in {DATA_DIR}")
        sys.exit(1)

    print(f"Building global map from {len(files)} frames...")

    global_pcd = o3d.geometry.PointCloud()
    sensor_positions = []

    # T_abs[i] = absolute pose of frame i in global coords (maps local → global)
    T_abs = np.eye(4)       # current frame's absolute pose
    T_abs_prev = np.eye(4)  # previous frame's absolute pose
    prev_frame_global = None

    for i, path in enumerate(files):
        frame_local = load_and_downsample(path)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(path)}  ({len(frame_local.points)} pts)", end="")

        if prev_frame_global is not None:
            # Constant-velocity prior: extrapolate next pose from last relative motion
            if i == 1:
                init = T_abs  # no history yet, use identity motion
            else:
                rel = T_abs @ np.linalg.inv(T_abs_prev)   # last relative motion
                init = rel @ T_abs                          # extrapolate

            T_abs_prev = T_abs.copy()
            # ICP: source=frame_local, target=prev_frame already in global space
            # result is the absolute transform T_abs for this frame
            T_abs = icp(frame_local, prev_frame_global, init=init)

        sensor_pos = T_abs[:3, 3].tolist()
        sensor_positions.append(sensor_pos)

        frame_global = o3d.geometry.PointCloud(frame_local)
        frame_global.transform(T_abs)
        global_pcd += frame_global
        prev_frame_global = frame_global

        print(f"  sensor @ ({sensor_pos[0]:.2f}, {sensor_pos[1]:.2f}, {sensor_pos[2]:.2f})")

    global_pcd = global_pcd.voxel_down_sample(VOXEL_SIZE)
    colorize_by_height(global_pcd)
    print(f"\nGlobal map: {len(global_pcd.points)} points")

    # Red spheres at each sensor position
    sensor_markers = o3d.geometry.TriangleMesh()
    for pos in sensor_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.translate(pos)
        sensor_markers += sphere
    sensor_markers.paint_uniform_color([1, 0, 0])
    sensor_markers.compute_vertex_normals()

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"global_map_{n:04d}frames.png")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Global Map ({n} frames)", width=1920, height=1080, visible=True)
    vis.add_geometry(global_pcd)
    vis.add_geometry(sensor_markers)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.3)
    ctr.set_front([0.0, -0.3, 0.95])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_lookat(np.asarray(global_pcd.points).mean(axis=0).tolist())

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.point_size = 1.5

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
