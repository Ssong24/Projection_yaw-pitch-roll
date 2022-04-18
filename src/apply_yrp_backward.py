import cv2
import numpy as np
import os

f = 500
rotXval = 90    # pitch
rotYval = 90    # yaw
rotZval = 88.522    # roll
distXval = 500
distYval = 500
distZval = 500

def onFchange(val):
    global f
    f = val
def onRotXChange(val):
    global rotXval
    rotXval = val
def onRotYChange(val):
    global rotYval
    rotYval = val
def onRotZChange(val):
    global rotZval
    rotZval = val
def onDistXChange(val):
    global distXval
    distXval = val
def onDistYChange(val):
    global distYval
    distYval = val
def onDistZChange(val):
    global distZval
    distZval = val

def create_meshgrid():
    


if __name__ == '__main__':
    print("current path: " ,os.getcwd())

    #Read input image, and create output image
    src_dir = "/Users/3i-21-331/workspace/stitching/mobile_stitching/dataset/PivoX_full/galaxy_zflip3/WideLens_20deg/correct_capture_1/"
    n_img = 9


    for i in range(0, n_img):
        src = cv2.imread(src_dir + 'd_' + str(i+1) + '.jpg')
        dst = np.zeros_like(src)
        h,w = src.shape[:2]

        if f <= 0: f = 1
        rotX = (rotXval - 90) * np.pi/180
        rotY = (rotYval - 90) * np.pi/180
        rotZ = (rotZval - 90) * np.pi/180
        distX = distXval - 500
        distY = distYval - 500
        distZ = distZval - 500

        # Camera intrinsic matrix
        K = np.array([[f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0,   1, 0]])

        # K inverse
        Kinv = np.zeros((4,3))
        Kinv[:3,:3] = np.linalg.inv(K[:3,:3])*f
        Kinv[-1,:] = [0, 0, 1]

        # Rotation matrices around the X,Y,Z axis
        RX = np.array([[1,           0,            0, 0],
                       [0,np.cos(rotX),-np.sin(rotX), 0],
                       [0,np.sin(rotX), np.cos(rotX), 0],
                       [0,           0,            0, 1]])

        RY = np.array([[ np.cos(rotY), 0, np.sin(rotY), 0],
                    [            0, 1,            0, 0],
                    [ -np.sin(rotY), 0, np.cos(rotY), 0],
                    [            0, 0,            0, 1]])

        RZ = np.array([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                    [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                    [            0,            0, 1, 0],
                    [            0,            0, 0, 1]])

        # Composed rotation matrix with (RX,RY,RZ)
        R = np.linalg.multi_dot([ RX , RY , RZ ])

        # Translation matrix
        T = np.array([[1,0,0,distX],
                      [0,1,0,distY],
                      [0,0,1,distZ],
                      [0,0,0,1]])

        # Overall homography matrix
        H = np.linalg.multi_dot([K, R, T, Kinv])

        # Apply matrix transformation
        cv2.warpPerspective(src, H, (w, h), dst, cv2.WARP_INVERSE_MAP, cv2.BORDER_CONSTANT, 0)

        cv2.imwrite('images/backward_' + str(i+1) + '.jpg', dst)
