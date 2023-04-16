from vqpy.backend.operator.vobj_projector import VObjProjector, FrameProjector, FrameProjectorBatch
from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader, VideoReaderBatchLoad
from vqpy.backend.operator.vobj_filter import VObjFilter, VObjPropertyFilter
from vqpy.backend.operator.tracker import Tracker
from vqpy.backend.operator.frame_filter import FrameRangeFilter, FrameFilter
import json
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
import numpy as np


import pytest
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")
from pytorchvideo.data.encoded_video import EncodedVideo


def load_id_to_class():
    json_filename = os.path.join("/home/ubuntu/viva-vldb23-artifact/data/", 'kinetics_classnames.json')
    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")
    return kinetics_id_to_classname

def load_face_labels():
    filename = os.path.join("/home/ubuntu/viva-vldb23-artifact/data/", 'rcmalli_vggface_labels_v2.npy')
    labels = np.load(filename)
    return labels

id_to_class = load_id_to_class()

face_labels = load_face_labels()

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo, NormalizeVideo,
)

@pytest.fixture
def video_reader():
    video_path = os.path.join(resource_dir,
                            #   "/home/ubuntu/viva-vldb23-artifact/output/test_lebron.mp4")
                            #   "/home/ubuntu/viva-vldb23-artifact/data/dunk_video.mp4")
                            "/home/ubuntu/vqpy/lebron_james_best_highlights_while_wearing_no_6.mp4")
    assert os.path.isfile(video_path)
    video_reader = VideoReaderBatchLoad(video_path)
    return video_reader

import torch
action = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True, verbose=False)
action.eval()
action.to("cuda")

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 40

transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
    )

dunk_batch_size = 20
dunk_history_length = 10
def detect_dunk_batch(frames):
    images = [frame.image for frame in frames]

    video_data = {
        "video": torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in images], dim=1)
    }
    video_data = transform(video_data)
    inputs = video_data["video"]
    inputs = inputs.to('cuda', non_blocking=True)
    preds = action(inputs[None, ...])
    # with torch.no_grad():
    #     preds = action(images)
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_class = preds.topk(k=1).indices[0]
    pred_label = id_to_class[int(pred_class)]
    pred_score = preds.topk(k=1).values[0].item()

    result = {
        'label' : pred_label,
        'cls'   : int(pred_class),
        'score' : pred_score
    }

    return [result] * (len(frames) - dunk_history_length)

model_time = 0.0
facenet = InceptionResnetV1(pretrained='vggface2').eval().to("cuda")
facenet.classify = True
mtcnn = MTCNN(keep_all=True, device='cuda')
import time
def detect_face(frames):
    import torch
    images = [frame.image for frame in frames]
    t1 = time.perf_counter()
    img_bboxes, _ = mtcnn.detect(images)
    img_crops = mtcnn(images)
    t2 = time.perf_counter()
    print(f"detect face using {t2 - t1}")
    t3 = time.perf_counter()
    result = []
    for idx, img_cropped in enumerate(img_crops):

        next_map = {
            'xmin'  : [],
            'ymin'  : [],
            'xmax'  : [],
            'ymax'  : [],
            'label' : [],
            'cls' : [],
            'score' : []
        }

        # If there are no faces, we skip
        pred = False
        if img_cropped is not None:
            prediction = facenet(img_cropped.to('cuda', non_blocking=True))
            # To convert to probability
            probabilities_raw = torch.nn.functional.softmax(prediction, dim=1)
            probabilities, _ = torch.topk(probabilities_raw, 1) # Only need the score

            # Get the output and labels
            max_vals = torch.max(prediction, dim=1)
            for i in range(len(max_vals.values)):
                bbox = img_bboxes[idx][i]
                xmin, ymin, xmax, ymax = bbox
                ind = max_vals.indices[i].item()
                label = face_labels[ind].item()
                # Get the probability from the softmax as the score
                score = probabilities[i].item()

                # next_map['xmin'].append(xmin)
                # next_map['ymin'].append(ymin)
                # next_map['xmax'].append(xmax)
                # next_map['ymax'].append(ymax)
                # next_map['label'].append(label.strip())
                # next_map['cls'].append(ind)
                # next_map['score'].append(score)
                if label.strip() == "LeBron_James":
                    pred = True
                    break
        # else:
            # for k in next_map.keys():
            #     next_map[k].append(None)
        result.append(pred)

    t4 = time.perf_counter()
    print(f"predict face using {t4 - t3}")

    # for m in result:
    #     if "LeBron_James" in m["label"]:
    #         return True
    return result

face_window_length = 30

dunk_window_length = 30

def face_exists_window(frames):
    # first 5 frames are history
    # the rest are batchs
    # assert len(frames) == 155
    faces = [frame.properties["face"] for frame in frames]
    result = []
    for i in range(face_window_length, len(frames)):
        result.append(any(faces[max(i-face_window_length, 0):i+face_window_length]))
    return result

def dunk_exists_window(frames):
    dunks = [frame.properties["dunk"] for frame in frames]
    result = []
    for i in range(dunk_window_length, len(frames)):
        result.append(any(dunks[max(i-dunk_window_length, 0):i+dunk_window_length]))
    return result
    

import cv2 
import pandas as pd
def _test_output_ground_truth(video_reader):
    
    final_result = []
    count = 0
    while count < 1 and video_reader.has_next():
        batch = []
        ids = []
        # count += 1
        while video_reader.has_next():
            frame = video_reader.next()
            if frame.id % 30 != 0:
                continue
            image = frame.image
            batch.append(image)
            ids.append(frame.id//30)
            print(frame.id//30)
            if len(batch) == 32:
                break
        result = detect_face({"image": batch}, None)
        outputs = []
        for m in result:
            if "LeBron_James" in m["label"]:
                outputs.append(True)
            else:
                outputs.append(False)
        final_result.extend(zip(ids, outputs))
    df = pd.DataFrame(final_result, columns=["id", "output"])
    df.to_csv("face_groud_output.csv", index=False)

# def _test_dunk_groud_truth(video_reader):
#     final_result = []
#     count = 0
#     while count < 1 and video_reader.has_next():
#         batch = []
#         ids = []
#         # count += 1
#         while video_reader.has_next():
#             frame = video_reader.next()
#             if frame.id % 30 != 0:
#                 continue
#             image = frame.image
#             batch.append(image)
#             ids.append(frame.id//30)
#             print(frame.id//30)
#             if len(batch) == 32:
#                 break
#         result = detect_dunk({"image": batch}, None)
#         outputs = []
#         for m in result:
#             if m["cls"] == 107:
#                 outputs.append(True)
#             else:
#                 outputs.append(False)
#         final_result.extend(zip(ids, outputs))
#     df = pd.DataFrame(final_result, columns=["id", "output"])
#     df.to_csv("dunk_groud_output.csv", index=False)
    
    
        

        
# [32, 87, 281, 331, 482, 532, 575, 580, 689] james face           

def frame_filter(frame):
    df = pd.read_csv("face_groud_output.csv")

    true_output_df = df[df["output"] == True]
    true_output = true_output_df["id"].tolist()
    # print(true_output)
    white_list = []
    for i in true_output:
        for j in range(int(i)-10, int(i)+10):
            white_list.append(j)
    if frame.id//30 in white_list:
        return True
    else:
        return False

def test_dunk(video_reader):
    node = video_reader

    node = FrameProjectorBatch(
        prev=node,
        property_name="face",
        property_batch_func=detect_face,
        dependencies=[((-1, None), {"image": 0})],
        batch_size=150
    )

    node = FrameProjectorBatch(
        prev=node,
        property_name="face_window",
        property_batch_func=face_exists_window,
        dependencies=[((-1, None), {"face": face_window_length})],
        batch_size=30
    )


    node = FrameFilter(
        prev=node,
        condition_func=lambda frame: frame.properties["face_window"],
        )


    node = FrameProjectorBatch(
        prev=node,
        property_name="dunk",
        property_batch_func=detect_dunk_batch,
        dependencies=[((-1, None), {"image": dunk_history_length})],
        batch_size=dunk_batch_size
    )

    def dunk_filter(frame):
        result = frame.properties["dunk"]
        return result is not None and result["cls"] == 107 # and result["score"] > 0.50


    # node = FrameProjectorBatch(
    #     prev=node,
    #     property_name="dunk_window",
    #     property_batch_func=dunk_exists_window,
    #     dependencies=[((-1, None), {"dunk": dunk_window_length})],
    #     batch_size=dunk_window_length
    # )

    # node = FrameFilter(
    #     prev=node,
    #     condition_func=dunk_filter,
    #     )


    # node = FrameFilter(
    #     prev=node,
    #     condition_func=lambda frame: frame.properties["dunk_window"],
    #     )
    

    import time
    start = time.perf_counter()
    count = 0
    # os.mkdir(f"/home/ubuntu/vqpy/dunk/")
    outputs = []
    while node.has_next():
        frame = node.next()
        count += 1
        result = frame.properties["dunk"]
        print(frame.id - dunk_history_length, result)
        outputs.append(frame.id)
    end = time.perf_counter()
    print(f"using time {end - start}")

    # vqpy output event, action threshold == 0.5
    # plus 30 and minus 30 frame, the same as viva
    # 30-76 1
    # 110-146 1
    # 230-302
    # 570-525 1
    # precision = 0.75, using hit event
    # recall = 0.5, using hit event
    # f1 = 0.6, using hit event
    # using time 45.88889806802035, including data ingestion time 13.3s


    ### viva output event, action threshold == 0.5
    # 129-131, 148-179, hit event: 1  
    # 316-388 hit event: 1
    # 473-520 hit event: 1
    # 537-552 
    # 573-620
    # using time 105.4555675983429, including data ingestion time 22.107012765976833
    # precision = 0.6
    # recall = 0.5
    # f1 = 0.55
