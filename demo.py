from time import time


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import math
import time

import OpenGL.GL as gl
import pangolin

from mplot2d import Mplot2d


def get_color(depth):

    up_th = 50
    low_th = 10
    th_range = up_th - low_th
    if (depth > up_th):
        depth = up_th
    if (depth < low_th):
        depth = low_th

    return (255 * depth / th_range, 0, 255 * (1 - depth / th_range))


# 视差计算


def disparity_SGBM(left_image, right_image, down_scale=False):

    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    param = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv.STEREO_SGBM_MODE_SGBM_3WAY
             }

    # 构建SGBM对象
    sgbm = cv.StereoSGBM_create(**param)
    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
    else:
        left_image_down = cv.pyrDown(left_image)
        right_image_down = cv.pyrDown(right_image)
        factor = size[0] / left_image_down.shape[1]
        disparity_left_half = sgbm.compute(left_image_down, right_image_down)
        disparity_right_half = sgbm.compute(right_image_down, left_image_down)
        disparity_left = cv.resize(
            disparity_left_half, size, interpolation=cv.INTER_AREA)
        disparity_right = cv.resize(
            disparity_right_half, size, interpolation=cv.INTER_AREA)
        disparity_left *= factor
        disparity_right *= factor

    return disparity_left, disparity_right


# 立体校正检验----画线

def draw_line2(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
    for k in range(15):
        cv.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)),
                (0, 255, 0), thickness=2, lineType=cv.LINE_AA)  # 直线间隔：100

    return output


prev_img = None
px_ref = None
px_cur = None
out_img = None
cur_R = None
cur_t = None
kitti_pose = None
true_x, true_y, true_z = 0.0, 0.0, 0.0
points = np.array([0.0, 0.0, 0.0])
detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
# detector = cv.xfeatures2d.SIFT_create()

lk_params = dict(winSize=(21, 21),
                 # maxLevel = 3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))


# err_plt = Mplot2d(xlabel='img id', ylabel='m', title='error')

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
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
focal = cam.fx
pp = (cam.cx, cam.cy)

# with open('/media/czy/DATA/Share/Kitti/poses/00.txt') as f:
with open('/home/guitarchen/dataset/poses/00.txt') as f:
    kitti_pose = f.readlines()

for img_id in range(0, 20):
    print("process img: " + str(img_id) + "--------------------------")
    img = cv.imread('/home/guitarchen/dataset/00/image_0/' +
                    str(img_id).zfill(6) + '.png', 0)
    cv.imshow("now img", img)
    # img = cv.imread('/media/czy/DATA/Share/Kitti/color/00/image_2/' +
    #                 str(img_id).zfill(6) + '.png', 0)

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
        _, cur_R, cur_t, mask = cv.recoverPose( E, kp2, kp1, focal=focal, pp=pp)
        print("cur_R: " + str(cur_R) + "\n" + "cur_t: " + str(cur_t))
        px_ref = kp1
        px_cur = kp2

    else:
        kp2, st, err = cv.calcOpticalFlowPyrLK( prev_img, img, px_ref, None, **lk_params)
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        E, mask = cv.findEssentialMat(
            kp2, kp1, focal=focal, pp=pp, method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, kp2, kp1, focal=focal, pp=pp)
        print("R: " + str(R) + "\n" + "t: " + str(t))

        projMatr1 = np.array( [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])    # 第一个相机参数
        projMatr2 = np.concatenate((R, t), axis=1)               # 第二个相机参数
        projMatr1 = np.matmul(cam.K, projMatr1)  # 相机内参 相机外参
        projMatr2 = np.matmul(cam.K, projMatr2)
        points4D = cv.triangulatePoints(projMatr1, projMatr2, kp1.T, kp2.T)
        points4D /= points4D[3]       # 归一化
        # print("points4D: " + str(points4D))
        points4D = points4D.T[:, 0:3]  # 取坐标点
        print("depth: " + str(points4D))

        line = kitti_pose[img_id - 1].strip().split()
        x_prev, y_prev, z_prev = float(line[3]), float(line[7]), float(line[11])
        line = kitti_pose[img_id].strip().split()
        x, y, z = float(line[3]), float(line[7]), float(line[11])
        absolute_scale = np.sqrt( (x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        print("absolute_scale: " + str(absolute_scale))

        cur_t = cur_t + absolute_scale * cur_R.dot(t)
        cur_R = R.dot(cur_R)
        print("cur_R: " + str(cur_R) + "\n" + "cur_t: " + str(cur_t))

        px_ref = kp1
        px_cur = kp2

        # errx = [img_id, math.fabs(x-cur_t[0])]
        # erry = [img_id, math.fabs(y-cur_t[1])]
        # errz = [img_id, math.fabs(z-cur_t[2])]
        # err_plt.draw(errx,'err_x',color='g')
        # err_plt.draw(erry,'err_y',color='b')
        # err_plt.draw(errz,'err_z',color='r')
        # err_plt.refresh()

        # 可视化
        # for i in range(kp1.shape[0]):
        #     # 第一幅图
        #     prev_img = cv.circle(prev_img, (kp1[i][0].astype(int),kp1[i][1].astype(int)), 10,get_color(points4D[i,2]), -1)
        #     # 第二幅图
        #     tmp_point = np.dot(R,points4D[i,:].reshape(3,1)) + t
        #     tmp_point = tmp_point.reshape(-1)
        #     img = cv.circle(img, (kp2[i][0].astype(int),kp2[i][1].astype(int)), 10,get_color(tmp_point[2]), -1)
        # plt.figure(figsize=(10,5))
        # plt.subplot(121)
        # plt.imshow(prev_img[:,:])
        # plt.subplot(122)
        # plt.imshow(img[:,:])
        # plt.show()

        lookdispL, lookdispR = disparity_SGBM(prev_img, img)
        linepic2 = draw_line2(lookdispL, lookdispR)
        # print(type(lookdispL))
        plt.imshow(lookdispL)

    prev_img = img
