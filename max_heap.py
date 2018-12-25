#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import math

def half(k):
    return math.floor(k/2)

class MaxHeap:
    def __init__(self, maxSize, value):
        self.priorityQueue = []
        for i in range(maxSize):
            self.priorityQueue.append({})
        self.maxSize = maxSize
        self.numberOfElements = -1
        self.getElementValue = value


    def enqueue(self, x):
        self.numberOfElements += 1
        self.priorityQueue[self.numberOfElements] = x
        self.swim(self.numberOfElements)

    def dequeue(self):
        max = self.priorityQueue[0]
        self.exchange(0, self.numberOfElements)
        self.numberOfElements -= 1
        self.sink(0)
        self.priorityQueue[self.numberOfElements + 1] = {}
        return max

    def empty(self):
        return self.numberOfElements == -1

    def size(self):
        return self.numberOfElements + 1

    def all(self):
        return self.priorityQueue.slice(0, self.numberOfElements + 1)

    def max(self):
        return self.priorityQueue[0]

    def swim(self, k):
        while k > 0 and self.less(half(k), k):
            self.exchange(k, half(k))
            k = half(k)

    def sink(self, k):
        while 2 * k <= self.numberOfElements:
            j = 2 * k
            if j < self.numberOfElements and self.less(j, j + 1):
                j += 1
            if self.less(k, j) is False:
                break
            self.exchange(k, j)
            k = j

    def getValueAt(self, i):
        #return self.getElementValue(self.priorityQueue[i])
        return self.priorityQueue[i]['score']

    def less(self, i, j):
        a = self.getValueAt(i)
        b = self.getValueAt(j)
        ret = math.isclose(a, b, rel_tol=1e-9)
        if ret is True:
            return False
        return a < b

    def exchange(self, i, j):
        t = self.priorityQueue[i]
        self.priorityQueue[i] = self.priorityQueue[j]
        self.priorityQueue[j] = t
