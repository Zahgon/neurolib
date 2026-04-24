import numpy as np

def adjustArrayShape(original, target):
    """
    Tiles and then cuts an array (or list or float) such that
    it has the same shape as target at the end.
    This is used to make sure that any input parameter like external current has
    the same shape as the rate array.
    """
    pass

def computeDelayMatrix(lengthMat, signalV, segmentLength=1):
    """
    Compute the delay matrix from the fiber length matrix and the signal
    velocity

        :param lengthMat:       A matrix containing the connection length in
            segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm

        :returns:    A matrix of connexion delay in ms
    """
    pass