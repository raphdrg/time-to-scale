import sys
import os
import open3d as o3d
import numpy as np

DATA_DIR = "data/lidar-warehouse-dataset/data"
LOG_DIR = "logs"

def colorize(pcd):
    if not pcd.has_colors():
        points = np.asarray(pcd.points)
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-9)
        colors = np.stack([z_norm, 0.4 * np.ones_like(z_norm), 1 - z_norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def render(pcd, out_path, title):
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    origin.translate([0, 0, 0])
    origin.paint_uniform_color([1, 0, 0])
    origin.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720, visible=True)
    vis.add_geometry(pcd)
    vis.add_geometry(origin)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.8, 0.5])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_lookat(np.asarray(pcd.points).mean(axis=0).tolist())

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.point_size = 2.0

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()
    print(f"  Saved -> {out_path}")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python visualize_pcd.py <num> [num2] [num3] ...")
        print("Example: python visualize_pcd.py 0 439 1727")
        sys.exit(1)

    for arg in sys.argv[1:]:
        filename = f"{int(arg):06d}.pcd"
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue

        print(f"Loading: {path}")
        pcd = o3d.io.read_point_cloud(path)
        print(f"  Points: {len(pcd.points)}")
        pcd = colorize(pcd)

        out_path = os.path.join(LOG_DIR, f"{int(arg):06d}.png")
        render(pcd, out_path, filename)

if __name__ == "__main__":
    main()
