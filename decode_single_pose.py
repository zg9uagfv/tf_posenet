#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np

PART_NAMES = [
    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
]

NUM_KEYPOINTS = len(PART_NAMES)  # 17


#############################################################
# argmax2d
#
# Input:
#   t (H, W, D)
# Returns:
#   matrix (D, 2), with each row as [y, x] of argmax for D
############################################################
def argmax2d(t):
    if len(t.shape) > 3:
        t = np.squeeze(t)

    if not len(t.shape) == 3:
        print("Input must be a 3D matrix, or be able to be squeezed into one.")
        return

    height, width, depth = t.shape

    reshaped_t = np.reshape(t, [height * width, depth])
    argmax_coords = np.argmax(reshaped_t, axis=0)
    y_coords = argmax_coords // width
    x_coords = argmax_coords % width

    return np.concatenate([np.expand_dims(y_coords, 1), np.expand_dims(x_coords, 1)], axis=1)


###################################################################################
# get_offset_vectors
#
# Input:
#   heatmap_coords (NUM_KEYPOINTS, 2)
#   offsets (height, width, NUM_KEYPOINTS * 2)
# Returns:
#   matrix (NUM_KEYPOINTS, 2), with each row as [y, x] of offset for each keypoint
###################################################################################
def get_offset_vectors(heatmaps_coords, offsets):
    result = []

    for keypoint in range(NUM_KEYPOINTS):
        heatmap_y = heatmaps_coords[keypoint, 0]
        heatmap_x = heatmaps_coords[keypoint, 1]

        offset_y = offsets[heatmap_y, heatmap_x, keypoint]
        offset_x = offsets[heatmap_y, heatmap_x, keypoint + NUM_KEYPOINTS]

        result.append([offset_y, offset_x])

    return result


############################################################################################
# get_offset_points
#
# Input:
#   heatmap_coords (NUM_KEYPOINTS, 2)
#   offsets (H, W, NUM_KEYPOINTS * 2)
#   output_stride (scalar)
# Returns:
#   matrix (NUM_KEYPOINTS, 2), with each row as [y, x] location prediction for each keypoint
#############################################################################################
def get_offset_points(heatmaps_coords, offsets, output_stride):
    offset_vectors = get_offset_vectors(heatmaps_coords, offsets)
    scaled_heatmap = heatmaps_coords * output_stride
    return scaled_heatmap + offset_vectors


##############################################################
# get_points_confidence
#
# Input:
#   heatmaps (H, W, NUM_KEYPOINTS)
#   heatmap_coords (NUM_KEYPOINTS, 2)
# Returns:
#   matrix (NUM_KEYPOINTS), with confidence for each keypoint
##############################################################
def get_points_confidence(heatmaps, heatmaps_coords):
    result = []

    for keypoint in range(NUM_KEYPOINTS):
        # Get max value of heatmap for each keypoint
        result.append(heatmaps[heatmaps_coords[keypoint, 0], \
                               heatmaps_coords[keypoint, 1], keypoint])

    return result


#####################################################################################################
# decode_single_pose
#
# Input:
#   heatmaps (H, W, NUM_KEYPOINTS)
#   offsets (H, W, NUM_KEYPOINTS * 2)
#   output_stride (scalar)
# Returns:
#   prediction in the form outlined at https://github.com/tensorflow/tfjs-models/tree/master/posenet
#####################################################################################################
def decode_single_pose(heatmaps, offsets, output_stride, width_factor, height_factor):
    # Squeeze into 3D arrays
    poses = []
    heatmaps = np.squeeze(heatmaps)
    offsets = np.squeeze(offsets)

    heatmaps_coords = argmax2d(heatmaps)
    offset_points = get_offset_points(heatmaps_coords, offsets, output_stride)
    keypoint_confidence = get_points_confidence(heatmaps, heatmaps_coords)

    keypoints = [{
        "position": {
            "y": offset_points[keypoint, 0]*height_factor,
            "x": offset_points[keypoint, 1]*width_factor
        },
        "part": PART_NAMES[keypoint],
        "score": score
    } for keypoint, score in enumerate(keypoint_confidence)]

    poses.append({"keypoints": keypoints, \
                  "score": (sum(keypoint_confidence) / len(keypoint_confidence))})
    return poses
