import numpy as np
import h5py
import sys
import os
import argparse
import yaml
import itertools
import networkx as nx
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
import rosbag
import std_msgs
import scipy.misc
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from class_util import class_mapping, class_to_color_rgb

parser = argparse.ArgumentParser("")
parser.add_argument( '--dataset', '-d', type=str, required=True, help='path to semantic KITTI dataset')
parser.add_argument( '--bag', '-b', type=str, required=True, help='path to output .bag file')
parser.add_argument( '--output', '-o', type=str, required=True, help='path to output .h5 file')
parser.add_argument( '--sequences', '-s', type=str, default='00,01,02,03,04,05,06,07,08,09,10', help='')
parser.add_argument( '--interval', '-i', type=int, default=20, help='')
parser.add_argument( '--min-cluster', '-m', type=int, default=50, help='')
parser.add_argument( '--voxel-resolution', '-v', type=float, default=0.3, help='')
parser.add_argument( '--downsample-resolution', '-r', type=float, default=0.1, help='')
parser.add_argument( '--distance-span', '-p', type=float, default=50.0, help='')
parser.add_argument( '--skip', '-k', type=int, default=10, help='')
FLAGS, unparsed = parser.parse_known_args()

def downsample(cloud, resolution):
    voxel_coordinates = [tuple(p) for p in np.round((cloud[:,:3] / resolution)).astype(int)]
    voxel_set = set()
    downsampled_cloud = []
    for i in range(len(cloud)):
        if not voxel_coordinates[i] in voxel_set:
            voxel_set.add(voxel_coordinates[i])
            downsampled_cloud.append(cloud[i])
    return np.array(downsampled_cloud)

# get class names
#yaml_file = open(os.path.join(FLAGS.dataset, "semantic-kitti.yaml"), 'r')
#config = yaml.full_load(yaml_file)
#class_names = config['labels']
#yaml_file.close()

all_points = []
count = []
bag = rosbag.Bag(FLAGS.bag, 'w')
for sequence in FLAGS.sequences.split(','):
    # get camera calibration
    calib_file = open(os.path.join(FLAGS.dataset, "sequences", sequence, "calib.txt"), 'r')
    calib = {}
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()

    # get poses
    poses = []
    Tr = calib["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    pose_file = open(os.path.join(FLAGS.dataset, "sequences", sequence, "poses.txt"), 'r')
    for line in pose_file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    pose_file.close()

    # get timestamps
    times = np.loadtxt(os.path.join(FLAGS.dataset, "sequences", sequence, "times.txt"))

    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            sequence, "velodyne")
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()
    label_paths = os.path.join(FLAGS.dataset, "sequences",
                             sequence, "labels")
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()
    image_paths = os.path.join(FLAGS.dataset, "sequences",
                             sequence, "image_2")
    image_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(image_paths)) for f in fn]
    image_names.sort()

    rgb_map = {}
    stacked_points = []
    pose_arr = []
    offset = 0
#    while offset < len(scan_names):
    while offset < 500:
        scan = np.fromfile(scan_names[offset], dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # get XYZ in world coordinate frame
        xyz_local = scan[:, 0:3]
        R_world_local = poses[offset][:3, :3]
        t_world_local = poses[offset][:3, 3]
        xyz_world = xyz_local.dot(R_world_local.T) + t_world_local
        xyz_voxels = [tuple(v) for v in np.round(xyz_world / FLAGS.voxel_resolution).astype(int)]
        
        # project RGB colors
        rgb = np.zeros((len(xyz_local), 3))
        image = scipy.misc.imread(image_names[offset])
        xyz_homogenous = np.hstack((xyz_local, np.ones(len(xyz_local)).reshape(-1, 1)))
        xyz_homogenous = calib['P2'].dot(calib['Tr'].dot(xyz_homogenous.T)).T
        uv = np.round(xyz_homogenous[:, :2] / xyz_homogenous[:, 2:3]).astype(int)
        valid = xyz_homogenous[:, 2] > 0
        valid = np.logical_and(valid, uv[:, 0] >= 0)
        valid = np.logical_and(valid, uv[:, 0] < image.shape[1])
        valid = np.logical_and(valid, uv[:, 1] >= 0)
        valid = np.logical_and(valid, uv[:, 1] < image.shape[0])
        for i in np.arange(len(xyz_homogenous))[valid]:
            rgb[i, :] = image[uv[i,1], uv[i,0], :]
#            if not xyz_voxels[i] in rgb_map:
#                rgb_map[xyz_voxels[i]] = rgb[i, :]
#        for i in np.arange(len(xyz_homogenous))[~valid]:
#            if xyz_voxels[i] in rgb_map:
#                rgb[i, :] = rgb_map[xyz_voxels[i]]
        rgb = rgb / 255.0 - 0.5

        # get point labels
        label = np.fromfile(label_names[offset], dtype=np.uint32)
        obj_id = [l >> 16 for l in label]
        cls_id = [l & 0xFFFF for l in label]

        # stack in Nx8 array
        points = np.zeros((len(xyz_world), 8))
        points[:, :3] = xyz_world
        points[:, 3:6] = rgb
        points[:, 6] = obj_id
        points[:, 7] = cls_id
        # filter out points with no valid color mapping
        points = points[~np.all(rgb == -0.5, axis=1), :]
        # filter out points from moving objects
#        points = points[points[:, 7] < 250]
        stacked_points.extend(points)
        print('Processing %d points from %s'%(len(points), scan_names[offset][len(FLAGS.dataset):]))

        if offset % FLAGS.interval == FLAGS.interval - 1:
            stacked_points = np.array(stacked_points)
            stacked_points = downsample(stacked_points, FLAGS.downsample_resolution)

            # get equalized resolution for connected components
            equalized_idx = []
            unequalized_idx = []
            equalized_map = {}
            point_voxels = [tuple(v) for v in np.round(stacked_points[:,:3]/FLAGS.voxel_resolution).astype(int)]
            for i in range(len(stacked_points)):
                k = point_voxels[i]
                if not k in equalized_map:
                    equalized_map[k] = len(equalized_idx)
                    equalized_idx.append(i)
                unequalized_idx.append(equalized_map[k])
            points = stacked_points[equalized_idx, :]
            point_voxels = [tuple(v) for v in np.round(points[:,:3]/FLAGS.voxel_resolution).astype(int)]
            obj_id = points[:, 6]
            cls_id = points[:, 7]
            new_obj_id = np.zeros(len(obj_id), dtype=int)

            # connected components to label unassigned obj IDs
            original_obj_id = set(points[:, 6]) - set([0])
            cluster_id = 1
            for i in original_obj_id:
                new_obj_id[obj_id == i] = cluster_id
                cluster_id += 1 

            edges = []
            for i in range(len(point_voxels)):
                if obj_id[i] > 0:
                    continue
                k = point_voxels[i]
                for d in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
                    if d!=(0,0,0):
                        kk = (k[0]+d[0], k[1]+d[1], k[2]+d[2])
                        if kk in equalized_map and cls_id[i] == cls_id[equalized_map[kk]]:
                            edges.append([i, equalized_map[kk]])
            G = nx.Graph(edges)
            clusters = nx.connected_components(G)
            clusters = [list(c) for c in clusters]
            for i in range(len(clusters)):
                if len(clusters[i]) > FLAGS.min_cluster:
                    new_obj_id[clusters[i]] = cluster_id
                    cluster_id += 1

            stacked_points[:, 6] = new_obj_id[unequalized_idx]
#            stacked_points = stacked_points[stacked_points[:, 6] > 0, :]
            stacked_points[:, 7] = [class_mapping[c] for c in stacked_points[:, 7]]
            print('Creating data sample with %d->%d points %d->%d objects' % (len(stacked_points), len(points), len(original_obj_id), len(set(new_obj_id))))
            all_points.extend(stacked_points)
            count.append(len(stacked_points))

            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.from_sec(times[offset])
            header.frame_id = '/map'
            fields = [
                PointField('x',0,PointField.FLOAT32,1),
                PointField('y',4,PointField.FLOAT32,1),
                PointField('z',8,PointField.FLOAT32,1),
                PointField('r',12,PointField.UINT8,1),
                PointField('g',13,PointField.UINT8,1),
                PointField('b',14,PointField.UINT8,1),
                PointField('o',15,PointField.INT32,1),
                PointField('c',19,PointField.INT32,1),
            ]
            stacked_points[:, 3:6] = (stacked_points[:, 3:6]+0.5) * 255
            pcd_with_labels = point_cloud2.create_cloud(header,fields, stacked_points)
            bag.write('laser_cloud_surround',pcd_with_labels,t=header.stamp)

            fields = [
                PointField('x',0,PointField.FLOAT32,1),
                PointField('y',4,PointField.FLOAT32,1),
                PointField('z',8,PointField.FLOAT32,1),
                PointField('rgb',12,PointField.INT32,1),
            ]
            cloud = [[p[0],p[1],p[2],(int(p[3])<<16)|int(p[4])<<8|int(p[5])] for p in stacked_points]
            pcd_with_labels = point_cloud2.create_cloud(header,fields, cloud)
            bag.write('rgb_cloud',pcd_with_labels,t=header.stamp)
            stacked_points[:, 3:6] = [class_to_color_rgb[c] for c in stacked_points[:, 7]]
            cloud = [[p[0],p[1],p[2],(int(p[3])<<16)|int(p[4])<<8|int(p[5])] for p in stacked_points]
            pcd_with_labels = point_cloud2.create_cloud(header,fields, cloud)
            bag.write('cls_cloud',pcd_with_labels,t=header.stamp)

            pose = PoseStamped()
            pose.header = header
            q = quaternion_from_matrix(poses[offset])
            pose.pose.position.x = t_world_local[0]
            pose.pose.position.y = t_world_local[1]
            pose.pose.position.z = t_world_local[2]
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            pose_arr.append(pose)
            bag.write('slam_out_pose',pose,t=header.stamp)

            path = Path()
            path.header = header
            path.poses = pose_arr
            bag.write('trajectory',path,t=header.stamp)

            offset += FLAGS.skip * FLAGS.interval + 1
            stacked_points = []
            rgb_map = {}

            # filter rgb map based on distance to reduce memory consumption
#            min_span = (t_world_local - FLAGS.distance_span) / FLAGS.voxel_resolution
#            max_span = (t_world_local + FLAGS.distance_span) / FLAGS.voxel_resolution
#            for v in list(rgb_map):
#                if np.any(min_span > v) or np.any(max_span < v):
#                    del rgb_map[v]
#            break
        else:
            offset += 1

bag.close()
#h5_fout = h5py.File(FLAGS.output,'w')
#h5_fout.create_dataset('points', data=all_points, compression='gzip', compression_opts=4, dtype=np.float32)
#h5_fout.create_dataset('count_room', data=count, compression='gzip', compression_opts=4, dtype=np.int32)
#h5_fout.close()

