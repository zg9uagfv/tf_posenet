#!/usr/bin/python3
# -*- coding: UTF-8 -*-

NUM_KEYPOINTS = 17

def getOffsetPoint(y, x, keypointId, offsets):
  return {'y':offsets[y][x][keypointId],
          'x':offsets[y][x][keypointId + NUM_KEYPOINTS]}
  '''
  return offsets[y][x][keypointId], offsets[y][x][keypointId + NUM_KEYPOINTS]
  '''

def getImageCoords(part, outputStride, offsets):
  heatmapY, heatmapX, id = part['y'], part['x'], part['id']
  offsetPoint = getOffsetPoint(heatmapY, heatmapX, id, offsets)
  y = offsetPoint['y']
  x = offsetPoint['x']
  return {'x':float(part['x'])*outputStride + float(x),
          'y':float(part['y'])*outputStride + float(y)}

'''
def fillArray<T>(element: T, size: number):
  const result: T[] = new Array(size);

  for (let i = 0; i < size; i++) {
    result[i] = element;
  }

  return result;
}
'''

def clamp(a, min, max):
    if a < min:
        return min
    if a > max:
        return max
    return a

def squaredDistance(y1, x1, y2, x2):
    dy = y2 - y1
    dx = x2 - x1
    return dy * dy + dx * dx

def addVectors(a, b):
    return {'x':a['x']+b['x'], 'y':a['y']+b['y']}

def clampVector(a, min, max):
    return {'y':clamp(a.y, min, max), 'x':clamp(a.x, min, max)}
