# coding:utf-8
import cv2
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(2)
#检查摄像头是否打开，值为TRUE or FALSE
flag = cap.isOpened()
index = 1
while (flag):
    ret, frame = cap.read()
    #flip(mat,mat,int)第三个参数：1左右翻  0 上下翻  -1 对角翻
    frame = cv2.flip(frame,1)
    cv2.imshow("Capture_Paizhao", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):  # 按下空格键，进入下面的保存图片操作(其他键位需要切换中英文和大小写太麻烦，直接空格简单)
        cv2.imwrite("/home/czy/Desktop/" + str(index) + ".jpg", frame)
        print("save" + str(index) + ".jpg successfuly!")
        print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
        index += 1
    elif k == 27:  # 按下Esc键，程序退出(Esc的ASCII值是27，即0001  1011)
        break
cap.release() # 释放摄像头
cv2.destroyAllWindows()# 释放并销毁窗口
