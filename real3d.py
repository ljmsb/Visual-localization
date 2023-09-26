import numpy as np
import cv2
height=0
savedir = "camera_data/"
cam_mtx = np.load(savedir + 'cam_mtx3.npy')
# dist = np.array([[-0.01739308, 0.22505005, 0.01253785, -0.00476795, -0.67140603]], dtype=np.float32)
dist = np.load(savedir + 'dist3.npy')
# newcam_mtx = np.array([[599.72671355 , 0.0,        322.78665478],
#               [ 0.0,        614.86680349,304.61409241],
#                [ 0.0,         0.0,         1.0        ]], dtype=np.float3     2)
newcam_mtx = np.load(savedir + 'newcam_mtx3.npy')
#newcam_mtx = np.load(savedir + 'cam_mtx.npy')
roi = np.load(savedir + 'roi2.npy')
worldPoints = np.array([[0, 0, 0],
                        [0, 3000, 0],
                        [3000, 0, 0],
                        [3000, 3000, 0]], dtype=np.float32)

# MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE
#手动输入坐标
# [u,v] center + 9 Image points
imagePoints = np.array([[209, 447],
                        [166, 45],
                        [600, 315],
                        [448, 77]],dtype=np.float32)
# imagePoints = np.array([[272, 427],
#                         [23, 98],
#                         [589, 186],
#                         [289, 48]], dtype=np.float32)
ret, rVec, tVec = cv2.solvePnP(worldPoints, imagePoints, newcam_mtx, dist)
def to3Dpts(point2D, rVec, tVec, cam_mtx, height):
 point3D = []
 point2D = (np.array(point2D, dtype='float32')).reshape(-1, 2)
 numPts = point2D.shape[0]
 point2D_op = np.hstack((point2D, np.ones((numPts, 1))))
 #按水平方向（列顺序）堆叠数组构成一个新的2d点数组
 rMat = cv2.Rodrigues(rVec)[0]
 rMat_inv = np.linalg.inv(rMat)
 kMat_inv = np.linalg.inv(cam_mtx)
 for point in range(numPts):
    uvPoint = point2D_op[point, :].reshape(3, 1)
    tempMat = np.matmul(rMat_inv, kMat_inv)
    tempMat1 = np.matmul(tempMat, uvPoint)
    tempMat2 = np.matmul(rMat_inv, tVec)
    s = (height + tempMat2[2]) / tempMat1[2]
    p = tempMat1 * s - tempMat2
    point3D.append(p)
 point3D = (np.array(point3D, dtype='float32')).reshape([-1, 1, 3])
 return point3D
print(to3Dpts([243,182],rVec,tVec,newcam_mtx,2))
