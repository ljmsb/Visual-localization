# https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html

import numpy as np
import cv2
import glob
import camera_realworldxyz


cameraXYZ = camera_realworldxyz.camera_realtimeXYZ()
calculatefromCam = True

imgdir = "/Users/mgl/PycharmProjects/vision_location/vision_location/camera_data"

writeValues = True

# test camera calibration against all points, calculating XYZ

# load camera calibration
savedir = "C://Users//Administrator//Desktop//bishe//camera_data//"
# cam_mtx= np.array([[599.72671355 , 0.0,        322.78665478],
#                [ 0.0,        614.86680349,304.61409241],
#                 [ 0.0,         0.0,         1.0        ]], dtype=np.float32)
cam_mtx = np.load(savedir + 'cam_mtx3.npy')
#dist = np.array([[-0.01739308, 0.22505005, 0.01253785, -0.00476795, -0.67140603]], dtype=np.float32)
dist = np.load(savedir + 'dist3.npy')
# newcam_mtx = np.array( [[410.80352783,   0.,         246.60587772]
#  [  0.,         975.50457764, 544.82168263]
#  [  0.,           0.,           1.        ]], dtype=np.float32)
newcam_mtx = np.load(savedir + 'newcam_mtx3.npy')
roi = np.load(savedir + 'roi2.npy')
# roi =  np.array( [[0,0,11,8]], dtype=np.float32)
# load center points from New Camera matrix
cx = newcam_mtx[0, 2]
cy = newcam_mtx[1, 2]
fx = newcam_mtx[0, 0]
print("cx: " + str(cx) + ",cy " + str(cy) + ",fx " + str(fx))

# MANUALLY INPUT YOUR MEASURED POINTS HERE
# ENTER (X,Y,d*)
# d* is the distance from your point to the camera lens. (d* = Z for the camera center)
# we will calculate Z in the next steps after extracting the new_cam matrix
#手动输入您的测量点在这里
#输入(X,Y,d*)
# d*是从你的点到相机镜头的距离。(d* = Z为相机中心)
#我们将在接下来的步骤中提取new_cam矩阵后计算Z

# world center + 9 world points

test_point=[230,182]

total_points_used = 4


worldPoints = np.array([[0, 0, 0],
                        [0, 300, 0],
                        [300, 0, 0],
                        [300, 300, 0]], dtype=np.float32)

# MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE
#手动输入坐标
# [u,v] center + 9 Image points
imagePoints = np.array([[209, 449],
                        [167, 45],
                        [600, 316],
                        [450, 77]],dtype=np.float32)
# imagePoints = np.array([[272, 427],
#                         [23, 98],
#                         [589, 186],
#                         [289, 48]], dtype=np.float32)

# imagePoints = np.array([[566, 636],
#                         [192, 147],
#                         [1050, 281],
#                         [599, 70]], dtype=np.float32)



# FOR REAL WORLD POINTS, CALCULATE Z from d*
#对于真实世界坐标的点，从d*计算Z
# for i in range(1, total_points_used):
#     # start from 1, given for center Z=d*
#     # to center of camera
#     wX = worldPoints[i, 0] - X_center
#     wY = worldPoints[i, 1] - Y_center
#     wd = worldPoints[i, 2]
#
#     d1 = np.sqrt(np.square(wX) + np.square(wY))
#     wZ = np.sqrt(np.square(wd) - np.square(d1))
#     worldPoints[i, 2] = wZ
#
# print(worldPoints)

# print(ret)
print("Camera Matrix")#摄像机矩阵
print(cam_mtx)
print("Distortion Coeff")#失真
print(dist)

print("Region of Interest")#感兴趣区
print(roi)
print("New Camera Matrix")#新相机矩阵
print(newcam_mtx)
inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
print("Inverse New Camera Matrix")#新相机逆矩阵,计算得出
print(inverse_newcam_mtx)
if writeValues == True: np.save(savedir + 'inverse_newcam_mtx.npy', inverse_newcam_mtx)

print(">==> Calibration Loaded")#标定加载

print("solvePNP")
ret, rvec1, tvec1 = cv2.solvePnP(worldPoints, imagePoints, newcam_mtx, dist)

print("pnp rvec1 - Rotation")#旋转
print(rvec1)
if writeValues == True: np.save(savedir + 'rvec1.npy', rvec1)

print("pnp tvec1 - Translation")#翻译
print(tvec1)
if writeValues == True: np.save(savedir + 'tvec1.npy', tvec1)

print("R - rodrigues vecs")
R_mtx, jac = cv2.Rodrigues(rvec1)
print(R_mtx)
if writeValues == True: np.save(savedir + 'R_mtx.npy', R_mtx)

print("R|t - Extrinsic Matrix")
Rt = np.column_stack((R_mtx, tvec1))
print(Rt)
if writeValues == True: np.save(savedir + 'Rt.npy', Rt)

print("newCamMtx*R|t - Projection Matrix")
P_mtx = newcam_mtx.dot(Rt)
print(P_mtx)
if writeValues == True: np.save(savedir + 'P_mtx.npy', P_mtx)

# [XYZ1]


# LETS CHECK THE ACCURACY HERE


s_arr = np.array([0], dtype=np.float32)
s_describe = np.array([0, 0, 0, 0], dtype=np.float32)

for i in range(0, total_points_used):
    print("=======POINT # " + str(i) + " =========================")

    print("Forward: From World Points, Find Image Pixel")#从世界坐标找到像素点
    XYZ1 = np.array([[worldPoints[i, 0], worldPoints[i, 1], worldPoints[i, 2], 1]], dtype=np.float32)
    XYZ1 = XYZ1.T
    print("{{-- XYZ1")
    print(XYZ1)
    suv1 = P_mtx.dot(XYZ1)
    print("//-- suv1")
    print(suv1)
    #得到相机坐标，乘以逆相机矩阵，减去tvec1
    s = suv1[2, 0]
    uv1 = suv1 / s
    print(">==> uv1 - Image Points")#像素点，为特征点在图像中的坐标
    print(uv1)
    print(">==> s - Scaling Factor")#比例因子
    print(s)
    s_arr = np.array([s / total_points_used + s_arr[0]], dtype=np.float32)
    s_describe[i] = s
    if writeValues == True: np.save(savedir + 's_arr.npy', s_arr)

    print("Solve: From Image Pixels, find World Points")

    uv_1 = np.array([[imagePoints[i, 0], imagePoints[i, 1], 1]], dtype=np.float32)
    uv_1 = uv_1.T
    print(">==> uv1")
    print(uv_1)
    suv_1 = s * uv_1
    print("//-- suv1")
    print(suv_1)

    print("get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")#得到相机坐标，乘以逆相机矩阵，减去tvec1
    xyz_c = inverse_newcam_mtx.dot(suv_1)
    xyz_c = xyz_c - tvec1
    print("      xyz_c")
    inverse_R_mtx = np.linalg.inv(R_mtx)
    XYZ = inverse_R_mtx.dot(xyz_c)
    print("{{-- XYZ")
    print(XYZ)

    if calculatefromCam == True:
        cXYZ = cameraXYZ.calculate_XYZ_real_z(imagePoints[i, 0], imagePoints[i, 1],0)
        print("camXYZ")
        print(cXYZ)

    test_XYZ = cameraXYZ.calculate_XYZ_real_z(test_point[0], test_point[1], 0)
    print("test point XYZ")
    print(test_XYZ)

s_mean, s_std = np.mean(s_describe), np.std(s_describe)

print(">>>>>>>>>>>>>>>>>>>>> S RESULTS")
print("Mean: " + str(s_mean))
# print("Average: " + str(s_arr[0]))
print("Std: " + str(s_std))

print(">>>>>> S Error by Point")

for i in range(0, total_points_used):
    print("Point " + str(i))
    print("S: " + str(s_describe[i]) + " Mean: " + str(s_mean) + " Error: " + str(s_describe[i] - s_mean))

