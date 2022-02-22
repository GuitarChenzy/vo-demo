import numpy as np
import cv2 as cv
import math

from mplot2d import Mplot2d

prev_img = None
px_ref = None
px_cur = None
out_img = None
cur_R = None
cur_t = None
kitti_pose = None
true_x, true_y, true_z = 0.0, 0.0, 0.0
detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
lk_params = dict(winSize=(21, 21),
                 # maxLevel = 3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
focal = cam.fx
pp = (cam.cx, cam.cy)

with open('/media/czy/DATA/Share/Kitti/poses/00.txt') as f:
    kitti_pose = f.readlines()

for img_id in range(0, 200):
    print("process img: " + str(img_id) + "--------------------------")
    # img = cv.imread('/home/guitarchen/dataset/00/image_0/' +
    #                 str(img_id).zfill(6) + '.png', 0)
    img = cv.imread('/media/czy/DATA/Share/Kitti/color/00/image_2/' +
                    str(img_id).zfill(6) + '.png', 0)
    if img_id == 0:
        feat_points = detector.detect(img)
        out_img = cv.drawKeypoints(
            image=img, keypoints=feat_points, outImage=out_img)
        # cv.imshow("feat_img", out_img)
        # cv.waitKey(1)
        px_ref = np.array([x.pt for x in feat_points], dtype=np.float32)
    elif img_id == 1:
        kp2, st, err = cv.calcOpticalFlowPyrLK(
            prev_img, img, px_ref, None, **lk_params)
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]

        E, mask = cv.findEssentialMat(
            kp2, kp1, focal=focal, pp=pp, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, cur_R, cur_t, mask = cv.recoverPose(E, kp2, kp1, focal=focal, pp=pp)
        print("cur_R: " + str(cur_R) + "\n" + "cur_t: " + str(cur_t))

        px_ref = kp1
        px_cur = kp2

    else:
        kp2, st, err = cv.calcOpticalFlowPyrLK(
            prev_img, img, px_ref, None, **lk_params)
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]

        E, mask = cv.findEssentialMat(
            kp2, kp1, focal=focal, pp=pp, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, kp2, kp1, focal=focal, pp=pp)
        print("R: " + str(R) + "\n" + "t: " + str(t))

        line = kitti_pose[img_id - 1].strip().split()
        x_prev, y_prev, z_prev = float(
            line[3]), float(line[7]), float(line[11])
        line = kitti_pose[img_id].strip().split()
        x, y, z = float(line[3]), float(line[7]), float(line[11])
        absolute_scale = np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        print(absolute_scale)
        cur_t = cur_t + absolute_scale * cur_R.dot(t)
        cur_R = R.dot(cur_R)
        print("cur_R: " + str(cur_R) + "\n" + "cur_t: " + str(cur_t))

        px_ref = kp1
        px_cur = kp2

        errx = [img_id, math.fabs(x-cur_t[0])]
        erry = [img_id, math.fabs(y-cur_t[1])]
        errz = [img_id, math.fabs(z-cur_t[2])] 
        err_plt.draw(errx,'err_x',color='g')
        err_plt.draw(erry,'err_y',color='b')
        err_plt.draw(errz,'err_z',color='r')
        err_plt.refresh()          
    prev_img = img
