import numpy as np
import pyrealsense2 as rs
import open3d as o3d

NUM_VIEWS = 5
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
pcd_list = []
try:
    for i in range(NUM_VIEWS):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Create point cloud from depth and color image
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        pcd = rs.pointcloud()
        pcd.map_to(color_frame)
        pcd_data = pcd.calculate(depth_frame)
        xyz = np.ndarray(buffer=pcd_data.get_vertices(), dtype=np.float32,
                         shape=(depth_frame.height, depth_frame.width, 3))
        rgb = np.ndarray(buffer=color_frame.get_data(), dtype=np.uint8,
                         shape=(color_frame.height, color_frame.width, 3))
        pcd_points = []
        for row in range(depth_frame.height):
            for col in range(depth_frame.width):
                if not np.isnan(xyz[row][col][0]) and not np.isnan(xyz[row][col][1]) and not np.isnan(xyz[row][col][2]):
                    pcd_points.append(
                        [xyz[row][col][0], xyz[row][col][1], xyz[row][col][2], rgb[row][col][2], rgb[row][col][1],
                         rgb[row][col][0]])
        pcd_array = np.array(pcd_points)
        # Convert to open3d point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_array[:, 3:] / 255.0)
        pcd_list.append(pcd_o3d)
finally:
    pipeline.stop()
trans_init = np.identity(4)
for i in range(1, NUM_VIEWS):
    print("Registering point clouds {0} and {1}".format(i - 1, i))
    pcd0 = pcd_list[i - 1]
    pcd1 = pcd_list[i]
    threshold = 0.02
    trans = o3d.registration.registration_icp(
        pcd0, pcd1, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=200))
    trans_init = trans.transformation
    pcd1.transform(trans_init)
pcd_combined = o3d.geometry.PointCloud()
for pcd in pcd_list:
    pcd_combined += pcd
pcd_combined_downsampled, _ = pcd_combined.voxel_down_sample(voxel_size=0.005)
o3d.io.write_point_cloud("model.ply", pcd_combined_downsampled)