from vqpy.backend.operator.vobj_projector import VObjProjector, FrameProjector, FrameProjectorBatch
from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader, VideoReaderBatchLoad
from vqpy.backend.operator.vobj_filter import VObjFilter, VObjPropertyFilter
from vqpy.backend.operator.tracker import Tracker
from vqpy.backend.operator.frame_filter import FrameRangeFilter, FrameFilter
import json
from facenet_pytorch import MTCNN
import numpy as np
import torch


def annotate_video(detections, output_video_path, table_name):
    color1=(207, 248, 64)
    color2=(255, 49, 49)
    thickness=4

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #codec
    video=cv2.VideoWriter(output_video_path, fourcc, fps, (width,height))

    frame_id = 0
    length = 1800
    # Capture frame-by-frame
    # ret = 1 if the video is captured; frame is the image
    ret, frame = vcap.read() 

    while ret:
        df = detections
        df = df[[f'{table_name}.id',f'{table_name}.bbox', f'{table_name}.labels', f'{table_name}.scores']][df[f'{table_name}.id'] == frame_id]
        if df.size:
            
            x1, y1, x2, y2 = df[f'{table_name}.bbox'].values[0]
            label = df[f'{table_name}.labels'].values[0]
            score = df[f'{table_name}.scores'].values[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # object bbox
            frame=cv2.rectangle(frame, (x1, y1), (x2, y2), color1, thickness) 
            # object label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, thickness)
            # object score
            cv2.putText(frame, str(round(score, 5)), (x1+120, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, thickness)
            # frame label
            cv2.putText(frame, 'Frame ID: ' + str(frame_id), (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color2, thickness) 
        
            video.write(frame)
            # Show every fifth frame
            if frame_id % 30 == 0:
                im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.imshow(im_rgb)
                plt.show()
            
            if frame_id > length:
                break
        
        frame_id+=1
        ret, frame = vcap.read()

    video.release()
    vcap.release()


import pytest
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")

import cv2
video_path = os.path.join(resource_dir,
                            "/home/ubuntu/vqpy/eva_experiments/modern_family.mp4")
mtcnn = MTCNN(device='cuda')
import time

global_face_detection_time = 0.0
# maybe make this into a object detector
def detect_face(frames, resize=False):
    start = time.perf_counter()
    import torch
    images = [frame.image for frame in frames]
    images = [torch.from_numpy(cv2.cvtColor(cv2.resize(img, (640, 360)) if resize else img, cv2.COLOR_RGB2BGR)) for img in images]
    images = torch.stack(images)
    detections = mtcnn.detect(images)
    boxes, scores = detections
    outcome = []
    for frame_boxes, frame_scores in zip(boxes, scores):
        pred_boxes = []
        pred_scores = []
        if frame_boxes is not None and frame_scores is not None:
            if not np.isnan(pred_boxes):
                pred_boxes = np.asarray(frame_boxes, dtype="int")
                if resize:
                    pred_boxes = pred_boxes * 2
                pred_scores = frame_scores
        outcome.append(
            {"bboxes": pred_boxes, "scores": pred_scores},
        )
    end = time.perf_counter()
    global global_face_detection_time
    global_face_detection_time += (end - start)
    return outcome

def detect_face_cheap(frames):
    return detect_face(frames, resize=True)

import sys
sys.path.append("./emotion_detector.py")
from emotion_detector import EmotionDetector

emotion_detector = EmotionDetector()

def crop(frame, bbox) -> np.ndarray:

    x0, y0, x1, y1 = np.asarray(bbox, dtype="int")
    # make sure the bbox is valid
    x0 = max(0, x0)
    y0 = max(0, y0)

    if x1 == x0:
        x1 = x0 + 1

    if y1 == y0:
        y1 = y0 + 1

    return frame[y0:y1, x0:x1]

def emotion_classify(frames):

    images = [frame.image for frame in frames]
    bboxes = [frame.properties["face"]["bboxes"] for frame in frames]
    bboxes_lens = [len(frame.properties["face"]["bboxes"]) for frame in frames]

    crops = []
    for image, bboxes in zip(images, bboxes):
        for box in bboxes:
            crops.append(emotion_detector.transform(crop(image, box)))
    
    predictions = emotion_detector.forward(crops)
    results = []
    count = 0
    for i in range(len(images)):
        row = []
        for l in range(bboxes_lens[i]):
            row.append(predictions[count])
            count += 1
        results.append(row)
    # print(results)
    return results

def test_emotion():
    
    # optimizor profiling
    node = VideoReaderBatchLoad(video_path, limit=1800)

    node = FrameProjectorBatch(
        prev=node,
        property_name="face",
        property_batch_func=detect_face,
        dependencies=[((-1, None), {"image": 0})],
        batch_size=128
    )

    node = FrameProjectorBatch(
        prev=node,
        property_name="emotion",
        property_batch_func=emotion_classify,
        dependencies=[((-1, None), {"face": 0})],
        batch_size=128
    )

    import time
    start = time.perf_counter()
    count = 0
    # os.mkdir(f"/home/ubuntu/vqpy/dunk/")
    outputs = []
    with torch.inference_mode():
        while node.has_next() and count < 30:
            frame = node.next()
            count += 1
            if count % 1 == 0:
                print(frame.id, frame.properties["face"], frame.properties["emotion"])
            outputs.append(frame)
        end = time.perf_counter()
        print(f"using time {end - start}")
        print(f"face detection time using {global_face_detection_time}")
        print(f"emotion detection using time {emotion_detector.detection_time}")

    results = []
    for frame in outputs:
        emotion = None
        score = 0
        for x in frame.properties["emotion"]:
            emotion = x["labels"]
            score = x["scores"]
            results.append((frame.id, emotion, score))
    import pandas
    df = pandas.DataFrame(results, columns=["id","labels", "scores"])
    df.to_csv("tv_show_analysis_2.csv", index=False)


# measure the average sentiment of the video
# detect face and emotion, positive emotion assign 1, neutral assign 0, other -1
# average each emotion detections over each second, determine positive of negative
#
# vqpy using 45.69412806601031s, accuracy: 0.9,  (optimization dynamic lower resolution)
# eva using 58.9s, accuary: 1 (using eva as gound truth)