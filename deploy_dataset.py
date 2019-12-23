import argparse
import urllib.request
import tarfile
import os
import glob
import shutil

import cv2

parser = argparse.ArgumentParser(description="Download dataset via Internet and format it. ")
parser.add_argument('kind', help="The kind of dataset you are to download. ")
args = parser.parse_args()

destination = "./input/Scene/"
os.makedirs(destination, exist_ok=True)

if args.kind == "NewCollege":
    # The New College Visionand Laser Data Set
    url = "https://ori.ox.ac.uk/NewCollegeData/TreeParklandSample/TreeParklandSample_StereoImages.tgz"
    urllib.request.urlretrieve(url, destination + "tmp.tgz")
    with tarfile.open(destination + "tmp.tgz") as tfile: tfile.extractall(destination)

    image_files = glob.glob(destination + "StereoImages/StereoImage__*-left.pnm")
    for id, image in enumerate(image_files):
        if os.path.isfile(image):
            cv2.imwrite(f"{destination}/frame{id:04}.png", cv2.imread(image))

    os.remove(destination + "tmp.tgz")
    shutil.rmtree(destination + "StereoImages")