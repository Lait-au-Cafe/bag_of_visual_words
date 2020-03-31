import itertools

import numpy as np
import cv2
from bitarray import bitarray
import bitarray.util as ba

import utils.brief
import utils.kmedian
import utils.vocabularytree

dataset_root_dir = "./input/Scene"

#======================================
# Extract Features
#======================================
frame_features = []
for id in itertools.count(0):
    #print(f"#{id}")
    frame = cv2.imread(f"{dataset_root_dir}/frame{id:04}.png", cv2.IMREAD_GRAYSCALE)
    if frame is None: 
        print(f"{dataset_root_dir}/frame{id:04}.png: Not Found. ")
        break
    else: 
        print(f"{dataset_root_dir}/frame{id:04}.png: Found. ")

    # Gauss Blur
    frame = cv2.GaussianBlur(frame, (21, 21), 0)

    fast_detector = cv2.FastFeatureDetector_create()
    keypoints = fast_detector.detect(frame)
#    display = np.empty_like(frame)
#    cv2.imshow("Key Points", cv2.drawKeypoints(frame, keypoints, display, (255,0,0)))
#    if cv2.waitKey(0) == 27: break

    brief_extractor = utils.brief.BriefDescriptorExtractor()
    keypoints, descriptors = brief_extractor.compute(frame, keypoints)
    frame_features.append((keypoints, descriptors))

    print(f"{len(keypoints)} keypoints are found for frame #{id}. ")
#    for kp, desc in zip(keypoints, descriptors):
#        #print(f"{kp.pt} => {ba.ba2int(bitarray(list(desc)))}")
#        print(f"{kp.pt} => {bitarray(list(desc)).to01()}")

#======================================
# Construct Vocabulary Tree
#======================================
#descriptors = np.empty((0, 256))
#for kps, descs in frame_features:
#    descriptors = np.append(descriptors, descs, axis=0)
#kw, lw = 10, 6
kw, lw = 10, 3
vocab_tree = utils.vocabularytree.VacabularyTree(kw, lw)
vocab_tree.build(frame_features)
#k_median = utils.kmedian.KMedian01(kw)
#centroids, clusters = k_median.run(descriptors)
#for i, (cntr, clster) in enumerate(zip(centroids, clusters)):
#    print(f"Cluster #{i} -> {clster.shape[0]}")
#    #print(f"\tCentroid: {bitarray(list(cntr)).to01()}")
#    #print(f"\tNum of data: {clster.shape[0]}")