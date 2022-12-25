import os
import cv2

sample = cv2.imread("SOCOFing/Altered/Altered-Easy/23__M_Left_thumb_finger_CR.BMP")

best_score = counter = 0
filename = image = kp1 = kp2 = mp = None
for file in os.listdir(
        "SOCOFing/Real"
):
    # if counter % 10 == 0:
    #counter += 1

    fingerprint_img = cv2.imread(
        "SOCOFing/Real/" + file
    )
    sift = cv2.SIFT_create()
    keypoints_1, des1 = sift.detectAndCompute(sample, None)
    keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(
        des1, des2, k=2
    )

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    if len(match_points) / keypoints * 100 > best_score:
        print("score:  " + str(len(match_points) / keypoints * 100))
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_img
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

if(filename):
    print("Best Match: " + filename)
    print("Score: " + str(best_score))

    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

