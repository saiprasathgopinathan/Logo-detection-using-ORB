import traceback
import cv2
import numpy as np


def createDetector(): # outputs the key points(coords) and pixel intensity from FAST and given by BRIEF respectively
    detector = cv2.ORB_create(nfeatures=2000)
    return detector


def getFeatures(img): # Feature are extracted for the given image using the above method
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None)
    return kps, descs, img.shape[:2][::-1]


def detectFeatures(img, train_features): # Bounding box for LOGO is drawn
    train_kps, train_descs, shape = train_features    # This is the input from he camera  
    kps, descs, _ = getFeatures(img)   # This is the LOGO feats
    # check if keypoints are extracted
    if not kps:
        return None

    
    # Since the output from BRIEF is going to be a binary vector, hamming distance is calculated
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(train_descs, descs, k=2)   # output coordinate in the camera input and distance between the logo and the detected image
   
    good = []
    #train KP matching KP of LOGO image it will be much closer than a non-matching KP. basically distance
    
    try:
        for m, n in matches:
            if m.distance < 0.9 * n.distance:  # Threshold set for finding good match distance
                good.append([m])

        # stop if we enough matching keypoints not found
        if len(good) < 0.1 * len(train_kps):
            return None

        # Getting a transformation matrix which maps keypoints from train image coordinates to sample image
        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)           
        
        
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        
        
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   # It's used for finding the perspective transformation(3Dimage - 2D image)

        if m is not None:
            # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
            scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1),
                                                                (shape[1] - 1, shape[0] - 1),
                                                                (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
            rect = cv2.minAreaRect(scene_points)
            # check resulting rect ratio knowing we have almost square train image
            if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
                return rect
    except:
        pass
    return None
