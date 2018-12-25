#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from decode_utils import squaredDistance, getImageCoords
from build_part_with_score_queue import buildPartWithScoreQueue
from decode_pose import decodePose
from max_heap import MaxHeap
import numpy as np

kLocalMaximumRadius = 1

def withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, coordinate, keypointId):
    x = coordinate['x']
    y = coordinate['y']
    for pose in poses:
        correspondingKeypoint = pose['keypoints'][keypointId]['position']
        distance = squaredDistance(y, \
                                   x, \
                                   correspondingKeypoint['y'], \
                                   correspondingKeypoint['x'])
        if distance <= squaredNmsRadius:
            return True
    return False

'''
/* Score the newly proposed object instance without taking into account
 * the scores of the parts that overlap with any previously detected
 * instance.
 */
'''
def getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints):
    notOverlappedKeypointScores = 0.0
    for idx in range(len(instanceKeypoints)):
        position = instanceKeypoints[idx]['position']
        score = instanceKeypoints[idx]['score']
        if withinNmsRadiusOfCorrespondingPoint(existingPoses, \
                                               squaredNmsRadius, \
                                               position, idx) is False:
            notOverlappedKeypointScores += score
    notOverlappedKeypointScores /= len(instanceKeypoints)
    return notOverlappedKeypointScores

def decodeMultiplePoses(scores, offsets, displacementsFwd, displacementsBwd, \
                        width_factor, height_factor, \
                        outputStride=16, maxPoseDetections=5, scoreThreshold= 0.5,
                        nmsRadius= 30):
    poses = []
    squaredNmsRadius = nmsRadius * nmsRadius
    scoresBuffer = np.squeeze(scores)
    offsetsBuffer = np.squeeze(offsets)
    displacementsBwdBuffer = np.squeeze(displacementsBwd)
    displacementsFwdBuffer = np.squeeze(displacementsFwd)
    height, width, numKeypoints = scoresBuffer.shape
    queue = MaxHeap(height * width * numKeypoints, scoresBuffer)
    buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius, scoresBuffer, queue)
    while len(poses) < maxPoseDetections and queue.empty() is False:
        root = queue.dequeue()
        rootImageCoords = getImageCoords(root['part'], outputStride, offsetsBuffer)
        if withinNmsRadiusOfCorrespondingPoint(poses, \
                                               squaredNmsRadius, \
                                               rootImageCoords, \
                                               root['part']['id']) is True:
            continue
        #Start a new detection instance at the position of the root.
        keypoints = decodePose(root, \
                               scoresBuffer, \
                               offsetsBuffer, \
                               outputStride, \
                               displacementsFwdBuffer, \
                               displacementsBwdBuffer)
        for keypoint in keypoints:
            keypoint['position']['y'] *= (height_factor)
            keypoint['position']['x'] *= (width_factor)
        score = getInstanceScore(poses, squaredNmsRadius, keypoints)
        poses.append({'keypoints':keypoints, 'score':score})
    return poses
