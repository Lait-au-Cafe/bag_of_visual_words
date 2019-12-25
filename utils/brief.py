import typing
import os

import numpy as np
import cv2

class BriefDescriptorExtractor:
    PATCH_SIZE: typing.ClassVar[int] = 48
    DESCRIPTOR_LENGTH: typing.ClassVar[int] = 256
    PAIRS_FILENAME: typing.ClassVar[str] = "brief_pairs.npy"

    @classmethod
    def generate_pairs(cls):
        #sigma = (cls.PATCH_SIZE ** 2) / 25
        #sigma2 = 4 * (cls.PATCH_SIZE ** 2) / 625
        sigma = cls.PATCH_SIZE / 5
        sigma2 = 2 * cls.PATCH_SIZE / 25

        a = np.random.normal(0, sigma, cls.DESCRIPTOR_LENGTH * 2)
        a = np.clip(a, -cls.PATCH_SIZE / 2, cls.PATCH_SIZE / 2)
        b = np.random.normal(a, sigma2)
        b = np.clip(b, -cls.PATCH_SIZE / 2, cls.PATCH_SIZE / 2)

        np.save(f"{os.path.dirname(__file__)}/{cls.PAIRS_FILENAME}", [a.reshape(-1, 2), b.reshape(-1, 2)])

    @classmethod
    def display_pairs(cls):
        scale = 4
        a, b = np.load(f"{os.path.dirname(__file__)}/{BriefDescriptorExtractor.PAIRS_FILENAME}")
        print(f"a -- Min: {np.amin(a)}, Max: {np.amax(a)}")
        print(f"b -- Min: {np.amin(b)}, Max: {np.amax(b)}")
        a += cls.PATCH_SIZE / 2
        b += cls.PATCH_SIZE / 2
        #print(np.append(a, b, axis=1)[0])

        disp_img = np.zeros((cls.PATCH_SIZE * scale, cls.PATCH_SIZE * scale, 3))
        for pts in np.append(a, b, axis=1).astype(np.int32):
            disp_img = cv2.line(disp_img, (pts[0] * scale, pts[1] * scale), (pts[2] * scale, pts[3] * scale), (0, 0, 255))
            #disp_img = cv2.circle(disp_img, (pts[0] * scale, pts[1] * scale), scale, (0, 0, 255))

        cv2.imshow("Pairs", disp_img)
        cv2.waitKey(0)

    def compute(self, 
        image: np.ndarray, 
        keypoints: typing.List[cv2.KeyPoint]
        ) -> typing.Tuple[typing.List[cv2.KeyPoint], typing.List]:

        a, b = np.load(f"{os.path.dirname(__file__)}/{BriefDescriptorExtractor.PAIRS_FILENAME}")
        kps = cv2.KeyPoint_convert(keypoints)

        # Filter keypoints
        print(f"kps: {kps.shape}")
        kps = kps[np.all(np.append(
            kps >= np.array([0, 0]) + BriefDescriptorExtractor.PATCH_SIZE / 2, 
            kps <  np.array(image.shape) - 1 - BriefDescriptorExtractor.PATCH_SIZE / 2, 
            axis=1), axis=1)]
        print(f"kps: {kps.shape}")

        descs = np.empty((0, 256), dtype=np.bool)
        for kp in kps:
            avals = image[tuple((kp+a).astype(np.int32).T)]
            bvals = image[tuple((kp+b).astype(np.int32).T)]
            desc = np.greater(avals, bvals)
            descs = np.append(descs, desc.reshape(1, 256), axis=0)
        #print(f"{descs.shape}: \n{descs}")

        keypoints = cv2.KeyPoint_convert(kps.reshape(-1, 1, 2))
        return (keypoints, descs)




if __name__ == "__main__":
    #BriefDescriptorExtractor.generate_pairs()
    BriefDescriptorExtractor.display_pairs()