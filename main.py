import itertools

import numpy as np
import cv2

import utils.brief

dataset_root_dir = "./input/Scene"

for id in itertools.count(0):
    print(f"#{id}")
    frame = cv2.imread(f"{dataset_root_dir}/frame{id:04}.png", cv2.IMREAD_GRAYSCALE)
    if frame is None: 
        print(f"{dataset_root_dir}/frame{id:04}.png: Not Found. ")
        break
    else: 
        print("Found. ")

    # Gauss Blur
    frame = cv2.GaussianBlur(frame, (21, 21), 0)

    fast_detector = cv2.FastFeatureDetector_create()
    keypoints = fast_detector.detect(frame)
#    display = np.empty_like(frame)
#    cv2.imshow("Key Points", cv2.drawKeypoints(frame, keypoints, display, (255,0,0)))
#    if cv2.waitKey(0) == 27: break

    brief_extractor = utils.brief.BriefDescriptorExtractor()
    keypoints, descriptors = brief_extractor.compute(frame, keypoints)
    print(f"{descriptors.shape}: \n{descriptors}")

    continue