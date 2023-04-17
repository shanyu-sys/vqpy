from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Callable, Dict, Any, List
import pandas as pd
import numpy as np
from vqpy.utils.images import crop_image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class VObjProjector(Operator):
    def __init__(self,
                 prev: Operator,
                 property_name: str,
                 property_func: Callable[[Dict], Any],
                 dependencies: Dict[str, int],
                 class_name: str,
                 filter_index: int = 0,
                 ):
        """
        Filter vobjs based on the condition_func.
        :param prev: previous operator
        :param property_name: the name of the property to be computed.
        :param property_func: a callable function that takes in the dependency
         data and returns the property value. The dependency data is a
         dictionary with the key being the dependency property name and the
         value being a list of the historical data (from old to new:
         [last_n-1th_value, last_n-2th_value, ..., cur_value])
         with length n of history length specified in the dependencies.
        :param dependencies: a dict of the dependencies of the property.
            The key is the name of the dependency property and the value is
            the history length (a non-negative integer) of the dependency
            property. If the value is 0, it means current frame. If the value
            is 1, it means the last frame and the current frame.
        :param class_name: the name of the vobj class to compute the property.
        :param filter_index: the index of the filter.
        """
        self.property_name = property_name
        self.property_func = property_func
        self.dependencies = dependencies
        self.filter_index = filter_index
        self.class_name = class_name
        self._hist_dependencies = {name: hist_len
                                   for name, hist_len in
                                   self.dependencies.items()
                                   if hist_len > 0}
        self._non_hist_dependencies = {name: hist_len
                                       for name, hist_len in
                                       self.dependencies.items()
                                       if hist_len == 0}
        self._stateful = len(self._hist_dependencies) > 0
        self._max_hist_len = max(dependencies.values())
        columns = ["track_id", "frame_id", "vobj_index"] + \
            list(self._hist_dependencies.keys())
        self._hist_buffer = pd.DataFrame(columns=columns)

        super().__init__(prev)

    def _get_cur_frame_dependencies(self, frame):
        # TODO: Add support for video metadata and frame image dependencies
        # 1. get vobj indexes in filter index of class name
        if self.filter_index not in frame.filtered_vobjs:
            raise ValueError("filter_index is not in filtered_vobjs")
        vobj_indexes = frame.filtered_vobjs[self.filter_index][self.class_name]
        # 2. get dependencies that need to be saved as history and current
        # frame dependencies.
        hist_deps = []
        non_hist_deps = []
        for vobj_index in vobj_indexes:
            vobj_data = frame.vobj_data[self.class_name][vobj_index].copy()
            if self._stateful and "track_id" not in vobj_data:
                continue
            else:
                # dependency of "image", which is the frame image cropped with
                #  the vobj's bbox
                if "image" in self.dependencies:
                    assert "tlbr" in vobj_data, "vobj_data does not have tlbr."
                    vobj_image = crop_image(frame.image, vobj_data["tlbr"])
                    vobj_data["image"] = vobj_image

                # dependency in video metadata
                # including "frame_width", "frame_height", "fps", "n_frames"
                for key in frame.video_metadata:
                    if key in self.dependencies:
                        vobj_data[key] = frame.video_metadata[key]

                # sanity check: dependencies should be in vobj_data
                assert all([dep_name in vobj_data for dep_name in
                            self.dependencies.keys()]), \
                    "vobj_data does not have all dependencies. Keys of "
                f"vobj_data: {vobj_data.keys()}. Keys of dependencies: "
                f"{self.dependencies.keys()}"
                # dependency data as current frame depenedency
                cur_dep = {dep_name: vobj_data[dep_name]
                           for dep_name in self._non_hist_dependencies.keys()}
                cur_dep.update(
                    {"vobj_index": vobj_index})
                non_hist_deps.append(cur_dep)

                # dependency data to be saved as history
                if self._stateful:
                    hist_dep = {dep_name: vobj_data[dep_name]
                                for dep_name in self._hist_dependencies.keys()}
                    hist_dep.update(
                        {"vobj_index": vobj_index,
                            "track_id": vobj_data["track_id"],
                            "frame_id": frame.id})

                    hist_deps.append(hist_dep)

        # sanity check
        if not self._stateful:
            assert not hist_deps, "stateful_deps should be empty"
        return non_hist_deps, hist_deps

    def _update_hist_buffer(self, hist_deps):
        self._hist_buffer = self._hist_buffer.append(hist_deps)
        # remove data that older than max history length
        cur_frame_id = hist_deps[0]["frame_id"]
        # frame_id starts from 0
        oldest_frame_id = cur_frame_id + 1 - (self._max_hist_len + 1)
        if oldest_frame_id >= 0:
            self._hist_buffer = self._hist_buffer[
                self._hist_buffer["frame_id"] >= oldest_frame_id]

    def _get_hist_dependency(self,
                             dependency_name,
                             track_id,
                             frame_id,
                             hist_len):
        # todo: allow user to fill missing data with a default value
        # currently fill with None
        hist_start = frame_id - hist_len
        # return None if there isn't enough history
        if hist_start < 0:
            return None, False
        # if dependency is the property itself, get history data from
        # hist_start to hist_end-1, and append None to the end as current
        # frame data.
        if dependency_name == self.property_name:
            hist_end = frame_id - 1
        else:
            hist_end = frame_id
        # get dependency data from hist buffer
        row = (self._hist_buffer["track_id"] == track_id) & \
              (self._hist_buffer["frame_id"] >= hist_start) & \
              (self._hist_buffer["frame_id"] <= hist_end)
        hist_df = self._hist_buffer.loc[row, ["frame_id", dependency_name]].\
            set_index("frame_id").reindex(range(hist_start, hist_end + 1))
        # fill missing frames with None
        hist_df = hist_df.replace(np.nan, None)
        hist_data = hist_df[dependency_name].tolist()
        # hist_data contains both history data and current frame data
        assert len(hist_data) == hist_len + 1

        if dependency_name == self.property_name:
            hist_data = hist_data.append(None)
        return hist_data, True

    def _compute_property(self, non_hist_data, hist_data, frame):
        # Todo: allow user to fill property without enough history with a
        # default value. Currently fill with None
        for i, cur_dep in enumerate(non_hist_data):
            vobj_index = cur_dep["vobj_index"]

            dep_data_dict = dict()
            all_enough = True
            for dependency_name, hist_len in self._hist_dependencies.items():
                hist_dep = hist_data[i]
                assert hist_dep["vobj_index"] == vobj_index
                assert dependency_name in hist_dep, \
                    f"dependency {dependency_name} is not in hist_dep"
                track_id = hist_dep["track_id"]
                frame_id = hist_dep["frame_id"]
                dep_data, enough = self._get_hist_dependency(dependency_name,
                                                             track_id=track_id,
                                                             frame_id=frame_id,
                                                             hist_len=hist_len)
                if enough:
                    assert len(dep_data) == hist_len + 1
                all_enough = all_enough and enough
                dep_data_dict[dependency_name] = dep_data

            for dependency_name in self._non_hist_dependencies:
                assert dependency_name in cur_dep, \
                    f"dependency {dependency_name} is not in cur_dep"
                dep_data_dict[dependency_name] = cur_dep[dependency_name]

            # compute property
            if all_enough:
                property_value = self.property_func(dep_data_dict)
            else:
                property_value = None
            # update frame vobj_data with computed property value for
            # corresponding vobj
            frame.vobj_data[self.class_name][vobj_index][self.property_name] =\
                property_value
        return frame

    def next(self) -> Frame:
        if self.prev.has_next():
            frame = self.prev.next()
            non_hist_data, hist_data = self._get_cur_frame_dependencies(frame)
            if self._stateful and hist_data:
                self._update_hist_buffer(hist_deps=hist_data)
            frame = self._compute_property(non_hist_data, hist_data,
                                           frame=frame)
            return frame
        else:
            raise StopIteration

from collections import defaultdict
from copy import deepcopy
class FrameProjector(Operator):

    def __init__(self,
                 prev,
                 property_name,
                 property_func,
                 dependencies,
                 ) -> None:
        self.history_buffer = defaultdict(list)
        self.property_name = property_name
        self.property_func = property_func
        self.dependencies = dependencies

        super().__init__(prev)

    def next(self):
        if self.prev.has_next():
            frame = self.prev.next()
            input_data_0 = {}
            input_data_1 = {}

            enought_history = True
            (index_0, class_0), deps0 = self.dependencies[0]
            (index_1, class_1), deps1 = self.dependencies[1]
            if class_0 is not None:
                for vobj_id_0 in frame.filtered_vobjs[index_0][class_0]:
                    vobj_0_current_value = deepcopy(frame.vobj_data[class_0][vobj_id_0])
                    for dep, history_len in deps0.items():
                        if history_len > 0:
                            self.history_buffer[(class_0, vobj_0_current_value["track_id"], dep)].append(vobj_0_current_value[dep])
                            original_history = self.history_buffer[(class_0, vobj_0_current_value["track_id"], dep)]
                            if len(original_history) > history_len:
                                self.history_buffer[(class_0, vobj_0_current_value["track_id"], dep)] = original_history[-history_len:]
                            if len(original_history) < history_len:
                                enought_history = False
                            else:
                                history = deepcopy(self.history_buffer[(class_0, vobj_0_current_value["track_id"], dep)])
                                vobj_0_current_value[dep] = history

                    input_data_0[vobj_id_0] = vobj_0_current_value
            else:
                input_data_0[0] = deepcopy(frame.properties)
                for dep, history_len in deps0.items():
                    if history_len > 0:
                        self.history_buffer[("_frame", dep)].append(frame.properties[dep])
                        original_history = self.history_buffer[("_frame", dep)]
                        if len(original_history) > history_len:
                            self.history_buffer[("_frame", dep)] = original_history[-history_len:]
                        if len(original_history) < history_len:
                            enought_history = False
                        else:
                            history = deepcopy(self.history_buffer[("_frame", dep)])
                            input_data_0[0][dep] = history

            if class_1 is not None:
                for vobj_id_1 in frame.filtered_vobjs[index_1][class_1]:
                    vobj_1_current_value = deepcopy(frame.vobj_data[class_1][vobj_id_1])

                    for dep, history_len in deps1.items():
                        if history_len > 0:
                            self.history_buffer[(class_1, vobj_1_current_value["track_id"])].append(vobj_1_current_value[dep])
                            original_history = self.history_buffer[(class_1, vobj_1_current_value["track_id"])]
                            if len(original_history) > history_len:
                                self.history_buffer[(class_1, vobj_1_current_value["track_id"])] = original_history[-history_len:]
                            
                            if len(original_history) < history_len:
                                enought_history = False
                            else:
                                history = deepcopy(self.history_buffer[(class_1, vobj_1_current_value["track_id"])])
                                vobj_1_current_value[dep] = history
                    input_data_1[vobj_id_1] = vobj_1_current_value
            else:
                input_data_1[0] = deepcopy(frame.properties)
                for dep, history_len in deps1.items():
                    if history_len > 0:
                        self.history_buffer[("_frame", dep)].append(frame.properties[dep])
                        original_history = self.history_buffer[("_frame", dep)]
                        if len(original_history) > history_len:
                            self.history_buffer[("_frame", dep)] = original_history[-history_len:]
                        if len(original_history) < history_len:
                            enought_history = False
                        else:
                            history = deepcopy(self.history_buffer[("_frame", dep)])
                            input_data_1[0][dep] = history            
            if not enought_history:
                frame.properties[self.property_name] = None
            else:
                values = []
                if class_0 is not None:
                    vobj0_ids = frame.filtered_vobjs[index_0][class_0]
                else:
                    vobj0_ids = [0]
                
                if class_1 is not None:
                    vobj1_ids = frame.filtered_vobjs[index_1][class_1]
                else:
                    vobj1_ids = [0]

                for vobj_id_0 in vobj0_ids:
                    for vobj_id_1 in vobj1_ids:
                        v = self.property_func(input_data_0[vobj_id_0], input_data_1[vobj_id_1])
                        values.append(v)
                if len(values) == 1:
                    frame.properties[self.property_name] = values[0]
                else:
                    frame.properties[self.property_name] = any(values)
                
            return frame
        else:
            raise StopIteration
                        
from collections import deque
class FrameProjectorBatch(Operator):

    def __init__(self,
                 prev,
                 property_name,
                 property_batch_func,
                 dependencies,
                 batch_size,
                ) -> None:
        
        self.history_length = max([max(v.values()) for k, v in dependencies])
    
        self.property_name = property_name
        self.property_func = property_batch_func
        self.dependencies = dependencies
        self.batch_size = batch_size
        self.current_batch_idx = - self.history_length
        self.batch_buffer = []
        self.result_buffer = []
        self.history_buffer = deque()

        super().__init__(prev)

    def has_next(self) -> bool:

        if self.history_length > 0:
            if len(self.history_buffer) < self.history_length:
                while self.prev.has_next() and len(self.history_buffer) < self.history_length:
                    frame = self.prev.next()
                    frame.properties[self.property_name] = None
                    self.history_buffer.append(frame)

            if self.current_batch_idx + self.history_length < len(self.history_buffer):
                # still in the history buffer range
                return True
        
        # now we have enough history

        # we have items left in result buffer
        if self.current_batch_idx < len(self.result_buffer):
            return True
        
        # the result_buffer has been exhausted, we accumulate anther batch

        # first we move the last history_len items in result_buffer to history_buff
        if self.history_length > 0:
            self.history_buffer.extend(self.result_buffer[-self.history_length:])
            while len(self.history_buffer) > self.history_length:
                self.history_buffer.popleft()

        self.result_buffer.clear()
        self.batch_buffer.clear()
        self.current_batch_idx = 0
        batch_count = 0
        while self.prev.has_next() and batch_count < self.batch_size:
            frame = self.prev.next()
            self.batch_buffer.append(frame)
            batch_count += 1
        
        # if self.property_name == "emotion":
        #     print("inside")
        if batch_count > 0:
            result = self.property_func(list(self.history_buffer) + self.batch_buffer)
            for i, r in enumerate(result):
                self.batch_buffer[i].properties[self.property_name] = result[i]
            self.result_buffer.extend(self.batch_buffer)
            return True
        else:
            return False

    def next(self):
        if self.has_next():
            if self.current_batch_idx < 0:
                result = self.history_buffer[self.current_batch_idx + self.history_length]
            else:
                result = self.result_buffer[self.current_batch_idx]
            self.current_batch_idx += 1
            return result
        else:
            raise StopIteration
            

                

                    

# class CrossVobjProjector(VObjProjector):
#     def __init__(self,
#                  prev: Operator,
#                  property_name: str,
#                  property_func: Callable[[List[Dict]], Any],
#                  dependencies: List[Dict[str, int]],
#                  input_filter_index: List[int],
#                  input_class_name: List[str],
#                  class_name: str,
#                  filter_index: int = 0,
#                  ):
#     super
    