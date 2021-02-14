Incremental Segmentation for Autonomous Driving
=================

KITTI dataset preparation
-----

First, download the Semantic KITTI [dataset](http://semantic-kitti.org/dataset.html#download). Then, run the following script for each sequence
to generate a labeled bag file for that sequence.

    python stage_semantic_kitti.py --dataset semantic_kitti/dataset/ --output data/00.h5 --bag data/00.bag --sequences 00 --interval 1 --skip 1
    python stage_semantic_kitti.py --dataset semantic_kitti/dataset/ --output data/01.h5 --bag data/01.bag --sequences 01 --interval 1 --skip 1
    python stage_semantic_kitti.py --dataset semantic_kitti/dataset/ --output data/02.h5 --bag data/02.bag --sequences 02 --interval 1 --skip 1

Training
------

    # Train PointNet with sequence 01,02 as training data and sequence 00 as validation data
    python train.py --train-area 01,02 --val-area 00 --net pointnet

Inference
-----

	#start the ROS node for incremental segmentation
	#select the trained model for sequence 00
	#select PointNet as the network architecture
	#use the flag --color to publish original color point cloud scans
	#use the flag --cluster to publish clustering results
	#use the flag --classify to publish classification results
    python inc_seg.py --net pointnet --area 00 --classify
	
	#use RViz as a visualization tool
	rviz -d inc_seg.rviz
	
	#publish the laser scan data from a ROS bag file
	rosbag play data/00.bag

Screenshots
-----

![screenshot](results/example.png?raw=true)
