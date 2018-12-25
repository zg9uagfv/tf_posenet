#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import math
from keypoints import poseChain, partIds, partNames
from decode_utils import clamp, addVectors, getImageCoords, getOffsetPoint

parentChildrenTuples = []

for joint_name in poseChain:
    parent_joint_name = joint_name[0]
    child_joint_name = joint_name[1]
    parentChildrenTuples.append([partIds[parent_joint_name], \
                                partIds[child_joint_name]])
''''
parentChildrenTuples = poseChain.map(
    ([parentJoinName, childJoinName]): NumberTuple =>
        ([partIds[parentJoinName], partIds[childJoinName]]));
'''

parentToChildEdges = []
for joint_id in parentChildrenTuples:
    parentToChildEdges.append(joint_id[1]) #child_joint_id

'''
const parentToChildEdges: number[] =
    parentChildrenTuples.map(([, childJointId]) => childJointId);
'''

childToParentEdges = []
for joint_id in parentChildrenTuples:
    childToParentEdges.append(joint_id[0]) #child_joint_id

'''
const childToParentEdges: number[] =
    parentChildrenTuples.map(([
                               parentJointId,
                             ]) => parentJointId);
'''

def getDisplacement(edgeId, point, displacements):
    numEdges = int(displacements.shape[2] / 2)
    point_x = int(point['x'])
    point_y = int(point['y'])
    y = displacements[point_y][point_x][edgeId]
    x = displacements[point_y][point_x][numEdges+edgeId]
    return {'y':y, 'x':x}

def getStridedIndexNearPoint(point, outputStride, height, width):
    return {'y':clamp(round(float(point['y']) / float(outputStride)), 0, height - 1), \
           'x':clamp(round(float(point['x']) / float(outputStride)), 0, width - 1)}

'''
 * We get a new keypoint along the `edgeId` for the pose instance, assuming
 * that the position of the `idSource` part is already known. For this, we
 * follow the displacement vector from the source to target part (stored in
 * the `i`-t channel of the displacement tensor).
'''
def traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId,
                             scoresBuffer, offsets, outputStride, displacements):
    height, width, numberScores = scoresBuffer.shape
    #Nearest neighbor interpolation for the source->target displacements.
    #最近邻插值(Nearest Neighbor interpolation)
    sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypoint['position'], \
                                                   outputStride, \
                                                   height, \
                                                   width)
    displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements)
    displacedPoint = addVectors(sourceKeypoint['position'], displacement)
    displacedPointIndices = getStridedIndexNearPoint(displacedPoint, \
                                                   outputStride, \
                                                   height, \
                                                   width)
    x = int(displacedPointIndices['x'])
    y = int(displacedPointIndices['y'])
    offsetPoint = getOffsetPoint(y, x, targetKeypointId, offsets)
    score = scoresBuffer[y][x][targetKeypointId]
    targetKeypoint = addVectors( \
        {'x': displacedPointIndices['x'] * outputStride, 'y': displacedPointIndices['y'] * outputStride}, \
        {'x': offsetPoint['x'], 'y': offsetPoint['y']})

    return {"position": targetKeypoint, \
            "part": partNames[targetKeypointId], \
            "score":score}

'''
 * Follows the displacement fields to decode the full pose of the object
 * instance given the position of a part that acts as root.
 *
 * @return An array of decoded keypoints and their scores for a single pose
'''
def decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd):
    numParts = scores.shape[2]
    numEdges = len(parentToChildEdges)

    #const instanceKeypoints: Keypoint[] = new Array(numParts);
    instanceKeypoints = []
    for i in range(numParts):
        keypoint = {'position':{'x':0, 'y':0}, 'part':'null', 'score':0.0}
        instanceKeypoints.append(keypoint)
    #Start a new detection instance at the position of the root.
    rootPart, rootScore = root['part'], root['score']
    rootPoint = getImageCoords(rootPart, outputStride, offsets)
    id = rootPart['id']
    instanceKeypoints[id] = {'position':rootPoint, 'part':partNames[id], 'score':rootScore}

    #Decode the part positions upwards in the tree, following the backward displacements.
    for edge in reversed(range(numEdges)):
        sourceKeypointId = parentToChildEdges[edge]
        targetKeypointId = childToParentEdges[edge]
        if instanceKeypoints[sourceKeypointId]['score'] > 0.0  \
            and math.isclose(instanceKeypoints[targetKeypointId]['score'], 0.0, rel_tol=1e-9) is True:
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge,
                                                                           instanceKeypoints[sourceKeypointId],
                                                                           targetKeypointId,
                                                                           scores,
                                                                           offsets,
                                                                           outputStride,
                                                                           displacementsBwd)

    # Decode the part positions downwards in the tree, following the forward
    #displacements.
    for edge in range(numEdges):
        sourceKeypointId = childToParentEdges[edge]
        targetKeypointId = parentToChildEdges[edge]
        #if instanceKeypoints[sourceKeypointId] is not None and instanceKeypoints[targetKeypointId] is None:
        if instanceKeypoints[sourceKeypointId]['score'] > 0.0 \
            and math.isclose(instanceKeypoints[targetKeypointId]['score'], 0.0, rel_tol=1e-9) is True:
            instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, \
                                                                           instanceKeypoints[sourceKeypointId], \
                                                                           targetKeypointId, \
                                                                           scores, \
                                                                           offsets, \
                                                                           outputStride, \
                                                                           displacementsFwd)
    return instanceKeypoints
