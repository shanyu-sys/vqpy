"""The ground-level tracker base class"""

from typing import Dict, List, Tuple
from vqpy.operator.video_reader import FrameStream
from vqpy.obj.frame import FrameInterface


class GroundTrackerBase(object):
    """The ground level tracker base class.
    Objects of this class approve detections results and associate the
    results with necessary data fields.
    """

    input_fields = []       # the required data fields for this tracker
    output_fields = []      # the data fields generated by this tracker

    def __init__(self, stream: FrameStream):
        raise NotImplementedError

    def update(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Filter the detected data and associate output data
        returns: the current tracked data and the current lost data
        """
        raise NotImplementedError


class SurfaceTrackerBase(object):
    """The surface level tracker base class.
    Objects of this class integrate detections results and associate the
    results with necessary data fields.
    """

    input_fields = []       # the required data fields for this tracker

    def update(self, data: List[Dict]) -> FrameInterface:
        """Generate the video objects using ground tracker and detection result
        returns: the current tracked/lost VObj instances"""
        raise NotImplementedError