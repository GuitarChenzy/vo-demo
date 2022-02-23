from random import random
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


# 视差计算
def sgbm(imgL, imgR):
    # SGBM参数设置
    blockSize = 9
    img_channels = 3
    stereo = cv.StereoSGBM_create(minDisparity=0,
                                  numDisparities=96,
                                  blockSize=blockSize,
                                  P1=8 * img_channels * blockSize * blockSize,
                                  P2=32 * img_channels * blockSize * blockSize,
                                  disp12MaxDiff=1,
                                  preFilterCap=63,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=100,
                                  mode=cv.STEREO_SGBM_MODE_HH)
    # 计算视差图
    disp = stereo.compute(imgL, imgR)
    disp = np.divide(disp.astype(np.float32), 16.)  # 除以16得到真实视差图
    return disp


prev_img = None
px_ref = None
px_cur = None
out_img = None
cur_R = None
cur_t = None
kitti_pose = None
true_x, true_y, true_z = 0.0, 0.0, 0.0
points = list()
detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
# detector = cv.xfeatures2d.SIFT_create()
lk_params = dict(winSize=(21, 21),
                 # maxLevel = 3,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

# err_plt = Mplot2d(xlabel='img id', ylabel='m', title='error')


class PinholeCamera:

    def __init__(self, width, height, fx, fy, cx, cy, b):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.b = b
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560,
                    607.1928, 185.2157, 0.573)
focal = cam.fx
pp = (cam.cx, cam.cy)

with open('/media/czy/DATA/Share/Kitti/poses/00.txt') as f:
    # with open('/home/guitarchen/dataset/poses/00.txt') as f:
    kitti_pose = f.readlines()

for img_id in range(0, 3):
    print("process img: " + str(img_id) + "--------------------------")
    # img = cv.imread('/home/guitarchen/dataset/00/image_0/' +
    #                 str(img_id).zfill(6) + '.png', 0)
    img = cv.imread('/media/czy/DATA/Share/Kitti/color/00/image_2/' +
                    str(img_id).zfill(6) + '.png', 0)
    img_r = cv.imread(
        '/media/czy/DATA/Share/Kitti/color/00/image_3/' + str(img_id).zfill(6) + '.png', 0)
    cv.imshow("now img", img)
    cv.waitKey(-1)
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

        projMatr1 = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])    # 第一个相机参数
        projMatr2 = np.concatenate((R, t), axis=1)               # 第二个相机参数
        projMatr1 = np.matmul(cam.K, projMatr1)  # 相机内参 相机外参
        projMatr2 = np.matmul(cam.K, projMatr2)
        points4D = cv.triangulatePoints(projMatr1, projMatr2, kp1.T, kp2.T)
        points4D /= points4D[3]       # 归一化
        # print("points4D: " + str(points4D))
        points4D = points4D.T[:, 0:3]  # 取坐标点
        print("depth: " + str(points4D))

        line = kitti_pose[img_id - 1].strip().split()
        x_prev, y_prev, z_prev = float(
            line[3]), float(line[7]), float(line[11])
        line = kitti_pose[img_id].strip().split()
        x, y, z = float(line[3]), float(line[7]), float(line[11])
        absolute_scale = np.sqrt(
            (x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
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

        # stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = sgbm(img, img_r)
        cv.imshow("disparity", disparity/96.0)
        cv.waitKey(-1)
        # plt.imshow(disparity,'gray')
        # plt.show()
        print(img.shape[1])
        for v in range(img.shape[0]):
            for u in range(img.shape[1]):
                if disparity[v, u] <= 0.0 and disparity[v, u] >= 96.0:
                    continue
                point = np.array([0, 0, 0, img[v, u] / 255.0])
                x = (u-cam.cx) / cam.fx
                y = (v - cam.cy) / cam.fy
                depth = cam.fx * cam.b / (disparity[v, u])
                point[0] = x * depth
                point[1] = y * depth
                point[2] = depth
                points.append(point)

    prev_img = img
# print(str(points))

pangolin.CreateWindowAndBind("Point Cloud Viewer", 1024, 768)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
s_cam = pangolin.OpenGlRenderState(
    pangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0))
d_cam = pangolin.CreateDisplay()
d_cam.SetBounds(0.0, 1.0, 175 / 1024., 1.0, -1024.0 / 768.0)
d_cam.SetHandler(pangolin.Handler3D(s_cam))
while pangolin.ShouldQuit() is False:
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    d_cam.Activate(s_cam)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    gl.glPointSize(2)
    gl.glBegin(gl.GL_POINTS)
    for p in points:
            gl.glColor3f(p[3], p[3], p[3])
            gl.glVertex3d(p[0], p[1], p[2])
    gl.glEnd()
    pangolin.FinishFrame()
    time.sleep(5)
