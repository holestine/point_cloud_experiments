'''
@article{Zhou2018,
	author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
	title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
	journal   = {arXiv:1801.09847},
	year      = {2018},
}
'''
## IMPORT LIBRARIES
import numpy as np
import time
import open3d
import pandas as pd
import matplotlib.pyplot as plt
import pptk

## USE http://www.open3d.org/docs/release/tutorial/Basic/

# Open a PCD file and visualize with Open3D
# Supported extensions are: pcd, ply, xyz, xyzrgb, xyzn, pts.
pcd = open3d.io.read_point_cloud("test_files\sdc.pcd")

# Visualize the point cloud
#open3d.visualization.draw_geometries([pcd])
v1 = pptk.viewer(pcd.points)

# Print out some of the 3D points
for i in range(10):
	print(pcd.points[i])

# Downsample the voxel grid
print(f"Points before downsampling: {len(pcd.points)} ")
pcd = pcd.voxel_down_sample(voxel_size=0.25)
print(f"Points after downsampling: {len(pcd.points)}")

# Visualize the downsampled data
#open3d.visualization.draw_geometries([pcd])
v2 = pptk.viewer(pcd.points)

# Segmentation
t1 = time.time()
plane_model, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0,1,1])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([1,0,0])
t2 = time.time()
print(f"Time to segment points using RANSAC {t2 - t1}")
open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# Cluster using DBSCAN
labels = np.array(outlier_cloud.cluster_dbscan(eps=.5, min_points=5, print_progress=False))
max_label= labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])
t3 = time.time()
print(f"Time to cluster outliers using DBSCAN {t3 - t2}")
open3d.visualization.draw_geometries([outlier_cloud])

## BONUS CHALLENGE - CLUSTERING USING KDTREE AND KNN INSTEAD
#anchor = 1000
#outlier_cloud.paint_uniform_color([.5,.5,.5])
#outlier_cloud.colors[anchor] = [0, 1, 0]
#pcd_tree = open3d.geometry.KDTreeFlann(outlier_cloud)
#[k, idx, _] = pcd_tree.search_radius_vector_3d(outlier_cloud.points[anchor], 3)
#np.asarray(outlier_cloud.colors)[idx[1:], :] = [0, 0, 1]
#open3d.visualization.draw_geometries([outlier_cloud])

## CHALLENGE 5 - BOUNDING BOXES IN 3D
obbs = []
indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

MAX_POINTS = 300
MIN_POINTS = 20
for i in range(len(indexes)):
	nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
	if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
		sub_cloud = outlier_cloud.select_by_index(indexes[i])
		obb = sub_cloud.get_axis_aligned_bounding_box()
		obb.color = (0,0,1)
		obbs.append(obb)
print(f"Number of Bound Boxes calculated {len(obbs)}")



## CHALLENGE 6 - VISUALIZE THE FINAL RESULTS
list_of_visuals = []
list_of_visuals.append(outlier_cloud)
list_of_visuals.extend(obbs)
list_of_visuals.append(inlier_cloud)

t4 = time.time()
print(type(pcd))
print(type(list_of_visuals))
print(f"Time to compute bounding boxes {t4-t3}")
open3d.visualization.draw_geometries(list_of_visuals)
## BONUS CHALLENGE 2 - MAKE IT WORK ON A VIDEO
