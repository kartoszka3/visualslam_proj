import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_camera_matrix_simple(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    cap.release()

    h, w = frame.shape[:2]
    focal_length = w * 1.2
    cx = w / 2
    cy = h / 2

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist = np.array([-0.2, 0.05, 0, 0, 0], dtype=np.float32)

    return K, dist


def orb_descriptor(image):
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2, kp1, kp2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

    return np.int32(pts1), np.int32(pts2)


if __name__ == "__main__":
    video_file = r"C:\Users\weron\PycharmProjects\studia\obrazy\WIN_20260427_15_35_38_Pro.mp4"

    K, dist = get_camera_matrix_simple(video_file)
    print("Camera Matrix K:\n", K)
    print("Distortion Coefficients:\n", dist)

    cap = cv2.VideoCapture(video_file)

    ret1, img1 = cap.read()

    for _ in range(30):
        cap.read()

    ret2, img2 = cap.read()
    cap.release()

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb_descriptor(gray1)
    kp2, des2 = orb_descriptor(gray2)

    pts1, pts2 = match_features(des1, des2, kp1, kp2)

    # macierz fundamentalna
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    print(f"Fundamental Matrix:\n{F}")

    # macierz istotna
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
    print(f"Essential Matrix:\n{E}")

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    print(f"Recovered Rotation:\n{R}")
    print(f"Recovered Translation:\n{t}")
    print(mask)
    # Macierze projekcji
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # Konwertuj na float32
    pts1_inliers = pts1[mask.ravel() > 0].astype(np.float32)
    pts2_inliers = pts2[mask.ravel() > 0].astype(np.float32)
    print(pts1_inliers)
    print(pts2_inliers)
    pts4D_hom = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
    pts3D = (pts4D_hom[:3, :] / pts4D_hom[3, :]).T

    #print(f"\n3D Points: {pts3D.shape}")
    #print(f"Translation: {t.flatten()}")

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], s=1)
    ax.set_title("3D Reconstruction")
    plt.show()