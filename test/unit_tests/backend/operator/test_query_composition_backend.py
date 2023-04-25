from vqpy.backend.operator.vobj_projector import VObjProjector, FrameProjector
from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.vobj_filter import VObjFilter, VObjPropertyFilter
from vqpy.backend.operator.tracker import Tracker
from vqpy.backend.operator.frame_filter import FrameRangeFilter, FrameFilter


import pytest
import os
# import fake_yolox  # noqa: F401
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")
import cv2
import numpy as np


@pytest.fixture
def video_reader():
    # video_path = os.path.join(resource_dir,
    #                           "right_turn_trim.mp4")
    # video_path = "/disk2/home/ubuntu/videos/auburn_raw000.mp4"
    video_path = "/data/Auburn/auburn_fri003.mp4"
    assert os.path.isfile(video_path)
    video_reader = VideoReader(video_path)
    return video_reader


@pytest.fixture
def frame_filter(video_reader):
    node = FrameFilter(
        prev=video_reader,
        condition_func=lambda frame: similarity(frame.image) < 200,
    )

    # node = FrameRangeFilter(
    #     prev=video_reader,
    #     frame_id_range=(0 * 30 , 15 * 30),
    # )

    # object_detector = ObjectDetector(
    #     prev=video_reader,
    #     class_names={"person", "car", "truck"},
    #     detector_name="yolox",
    #     # detector_kwargs={"device": "cpu"}
    # )
    return node


@pytest.fixture
def object_detector(frame_filter):

    # node = FrameRangeFilter(
    #     prev=video_reader,
    #     frame_id_range=(0 * 30 , 15 * 30),
    # )

    object_detector = ObjectDetector(
        prev=frame_filter,
        class_names={"person", "car", "truck"},
        detector_name="yolox",
        # detector_kwargs={"device": "cpu"}
    )
    return object_detector


@pytest.fixture
def tracker(video_reader, object_detector):
    fps = video_reader.metadata["fps"]
    tracker = Tracker(
        prev=object_detector,
        tracker_name="byte",
        class_name="person",
        fps=fps,
    )

    tracker = Tracker(
        prev=tracker,
        tracker_name="byte",
        class_name="car",
        fps=fps,
    )
    return tracker


def tlbr_score(values):
    tlbr = values["tlbr"]  # noqa: F841
    score = values["score"]
    return score > 0.5


def dep_tlbr_score(values):
    tlbr_score = values["tlbr_score"]
    return tlbr_score


def hist_tlbr(values):
    last_2_tlbrs = values["tlbr"]
    assert len(last_2_tlbrs) == 2
    for tlbr in last_2_tlbrs:
        if tlbr is None:
            return 0
    return last_2_tlbrs[0][0] - last_2_tlbrs[1][0]


def hist_tlbr_score(values):
    last_5_tlbrs = values["last_5_tlbrs"]
    cur_score = values["cur_score"]
    assert len(last_5_tlbrs) == 5
    return cur_score - 0.5


def dep_self(values):
    last_5_dep_selfs = values["dep_self"]
    assert len(last_5_dep_selfs) == 5
    assert last_5_dep_selfs[-1] is None
    assert all([dep_self == 1 for dep_self in last_5_dep_selfs[:-1]])
    return 1


image_resolution = (1513, 854)
video_resolution = (1920, 1080)
side_walk_image = [
    (576, 435), (757, 446),
    (344, 854), (0, 820)
]

right_turn_corner_image = [
    (759, 447), (985, 395), (1023, 525), (752, 542)
]

side_walk_video = [(int(x / 1513 * 1920), int(y / 854 * 1080)) for x, y in side_walk_image]
right_turn_corner_video = [(int(x / 1513 * 1920), int(y / 854 * 1080)) for x, y in right_turn_corner_image]

ref_image = cv2.imread("/data/Auburn/003_s1.jpg")

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def mse2(imageA, imageB):
    mse = ((imageA - imageB) ** 2).mean(axis=None)
    return mse


def extract_polygon_region(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    return cv2.bitwise_and(image, mask)

cropped_ref_image = extract_polygon_region(ref_image, np.array(right_turn_corner_video, np.int32))

def similarity(image):
    # import time
    # st = time.time()
    cropped_image = extract_polygon_region(image, np.array(right_turn_corner_video, np.int32))
    # t2 = time.time()
    # print("extract_polygon_region: ", t2 - st)
    similarity = mse(cropped_image, cropped_ref_image)
    # print("mse: ", time.time() - t2)

    # print("Similarity: ", similarity)
    return similarity


def in_polygon(tlbr, polygon):
    x1, y1, x4, y4 = tlbr
    x2, y2 = x1, y4
    x3, y3 = x4, y1

    upper = is_inside_polygon(polygon, (x1, y1)) or is_inside_polygon(polygon, (x3, y3))
    lower = is_inside_polygon(polygon, (x2, y2)) or is_inside_polygon(polygon, (x4, y4))
    return upper or lower

def point_in_triangle(pt, tri):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign(pt, tri[0], tri[1]) < 0
    b2 = sign(pt, tri[1], tri[2]) < 0
    b3 = sign(pt, tri[2], tri[0]) < 0

    return (b1 == b2) and (b2 == b3)

def is_inside_polygon(quad, pt):
    # Split the quadrilateral into two triangles
    tri1 = [quad[0], quad[1], quad[2]]
    tri2 = [quad[0], quad[2], quad[3]]

    # Check if the point lies inside either triangle
    return point_in_triangle(pt, tri1) or point_in_triangle(pt, tri2)

def in_side_walk(values):
    tlbr = values["tlbr"]
    return in_polygon(tlbr, side_walk_video)

def in_right_turn_corner(values):
    tlbr = values["tlbr"]
    x = tlbr[0]
    crossing = x > right_turn_corner_image[0][0]
    return crossing and in_polygon(tlbr, right_turn_corner_video)


def car_waiting(values):
    length = len(values["car_in_corner"])
    count = 0
    for value in values["car_in_corner"]:
        if value:
            count += 1
    return count == length

def car_waiting_person_in_side_walk(car_value, person_value):
    if "car_waiting" not in car_value:
        return False
    if "person_in_side_walk" not in person_value:
        return False
    car_waiting = car_value["car_waiting"]
    person_in_side_walk = person_value["person_in_side_walk"]
    return car_waiting and person_in_side_walk

def car_crossing_after_car_waiting_person_in_side_walk(scene_value, car_value):
    if "car_crosing_side_walk" not in car_value:
        return False
    if "car_waiting_person_in_side_walk" not in scene_value:
        return False
    crosing = car_value["car_crosing_side_walk"]
    waitinig = scene_value["car_waiting_person_in_side_walk"]
    waiting_before = any(waitinig)
    return crosing and waiting_before
    

def test_stateful_projector(tracker):

    # test projector with history
    node = VObjFilter(
        prev=tracker,
        condition_func="person",
        filter_index=0,
    )

    node = VObjFilter(
        prev=node,
        condition_func="car",
        filter_index=1,
    )

    node = VObjFilter(
        prev=node,
        condition_func="scene",
        filter_index=2,
    )

    hist_len = 1

    node = VObjProjector(
        prev=node,
        property_name="person_in_side_walk",
        property_func=in_side_walk,
        dependencies={"tlbr": 0},
        class_name="person",
        filter_index=0,
    )

    node = VObjProjector(
        prev=node,
        property_name="car_in_corner",
        property_func=in_right_turn_corner,
        dependencies={"tlbr": 0},
        class_name="car",
        filter_index=1,
    )

    node = VObjProjector(
        prev=node,
        property_name="car_waiting",
        property_func=car_waiting,
        dependencies={"car_in_corner": 30 * 3},
        class_name="car",
        filter_index=1,
    )

    node = VObjProjector(
        prev=node,
        property_name="car_waited",
        property_func=lambda x: any(x["car_waiting"]),
        dependencies={"car_waiting": 30 * 5},
        class_name="car",
        filter_index=1,
    )

    node = VObjProjector(
        prev=node,
        property_name="car_crosing_side_walk",
        property_func=in_side_walk,
        dependencies={"tlbr": 0},
        class_name="car",
        filter_index=1,
    )

    node = VObjPropertyFilter(
        prev=node,
        property_name="car_waited",
        property_condition_func=lambda x: x,
        filter_index=1
    )

    # node = VObjPropertyFilter(
    #     prev=node,
    #     property_name="person_in_side_walk",
    #     property_condition_func=lambda x: x,
    #     filter_index=0
    # )

    node = FrameProjector(
        prev=node,
        property_name="car_waiting_person_in_side_walk",
        property_func=car_waiting_person_in_side_walk,
        dependencies=[((1, "car"), {"car_waiting": 0}),
                      ((0, "person"), {"person_in_side_walk": 0})]
    )

    node = FrameProjector(
        prev=node,
        property_name="car_crossing_after_car_waiting_person_in_side_walk",
        property_func=car_crossing_after_car_waiting_person_in_side_walk,
        dependencies=[((-1, None), {"car_waiting_person_in_side_walk": 2 * 30}),
                      ((1, "car"), {"car_crosing_side_walk": 0})]
    )

    node = FrameFilter(
        prev=node,
        condition_func=lambda frame: frame.properties["car_crossing_after_car_waiting_person_in_side_walk"],
        )

    # node = SceneFrameFilter(
    #     prev=node,
    #     condition_func="car_crossing_after_car_waiting_person_in_side_walk")
    import time
    start = time.time()
    try:
        while node.has_next():
            frame = node.next()
            # print(frame)
            print(frame.id/30)
    except:
        print("time: ", time.time() - start)
            
        # if frame.filtered_vobjs[1]["car"]:
        #     # print(frame.vobj_data["person"][frame.filtered_vobjs[0]["person"][0]])
        #     for car_idx in frame.filtered_vobjs[1]["car"]:
        #         print(frame.vobj_data["car"][car_idx])
        # print(frame.properties)
            # print(frame.vobj_data["car"][frame.filtered_vobjs[1]["car"][0]])
            # print(frame.properties)
        # print(frame.properties)
        # print(frame.id // 30)
        # for vobj in frame.filtered_vobjs[1]["car"]:
        #     print(frame.vobj_data["car"][vobj])
            # print(frame.vobj_data["car"][vobj])
        # for person_idx in frame.filtered_vobjs[0]["person"]:
        #     print(frame.vobj_data["person"][person_idx])

    # projector2 = VObjProjector(
    # checked = False
    # while projector.has_next():
    #     frame = projector.next()
    #     for vobj in frame.vobj_data["person"]:
    #         track_id = vobj.get("track_id")
    #         if track_id:
    #             checked = True
    #             if frame.id < hist_len:
    #                 assert vobj["hist_tlbr"] is None
    #             else:
    #                 assert vobj["hist_tlbr"] is not None
    #                 # hist_buffer = projector._hist_buffer
    #                 # row = (hist_buffer["track_id"] == track_id) & \
    #                 #     (hist_buffer["frame_id"] == frame.id)
    #                 # assert hist_buffer.loc[row, "tlbr"] == vobj["tlbr"]
    # assert not projector._hist_buffer.empty
    # assert checked

#     # test delete history
    print("Total runtime:", time.time() - start)