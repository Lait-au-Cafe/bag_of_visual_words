import itertools

import numpy as np
import cv2
import cv2.xfeatures2d

dataset_root_dir = "./input/Scene"

for id in itertools.count(0):
    frame = cv2.imread(f"{dataset_root_dir}/frame{id:04}.png")
    if frame is None: break

    fast_detector = cv2.FastFeatureDetector_create()
    keypoints = fast_detector.detect(frame)

    display = np.empty_like(frame)
    cv2.imshow("Key Points", cv2.drawKeypoints(frame, keypoints, display, (255,0,0)))
    if cv2.waitKey(0) == 27: break

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, desc = brief.compute(frame, keypoints)