#!/usr/bin/python3
# -*- coding: UTF-8 -*-

def scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores):
    height, width, numKeypoints = scores.shape
    localMaximum = True
    yStart = max(heatmapY - localMaximumRadius, 0)
    yEnd = min(heatmapY + localMaximumRadius + 1, height)
    for yCurrent in range(yStart, yEnd):
        xStart = max(heatmapX - localMaximumRadius, 0)
        xEnd = min(heatmapX + localMaximumRadius + 1, width)
        for xCurrent in range(xStart, xEnd):
            if scores[yCurrent][xCurrent][keypointId] > score:
                localMaximum = False
                break
        if False == localMaximum:
            break
    return localMaximum

'''
* Builds a priority queue with part candidate positions for a specific image in
* the batch. For this we find all local maxima in the score maps with score
* values above a threshold. We create a single priority queue across all parts.
'''
def buildPartWithScoreQueue(scoreThreshold, localMaximumRadius, scores, queue):
    height, width, numKeypoints = scores.shape
    for heatmapY in range(height):
        for heatmapX in range(width):
            for keypointId in range(numKeypoints):
                score = scores[heatmapY][heatmapX][keypointId]
                if score < scoreThreshold:
                    continue
                if scoreIsMaximumInLocalWindow(keypointId, score, \
                                               heatmapY, heatmapX, \
                                               localMaximumRadius, scores) is True:
                    keypoint = {'score':score, 'part':{'y':heatmapY, 'x':heatmapX, 'id':keypointId}}
                    queue.enqueue(keypoint)
    return None
