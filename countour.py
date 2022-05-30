import cv2
import numpy as np
import SimpleITK as sitk


dcm_file = './raw/test/19.tif'
dcm_seg = './results/19.tif'



def drawContour():

    image = sitk.ReadImage(dcm_file)
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.squeeze(image_array)
    image_array = image_array.astype(np.float32)

    image_array = (image_array - (-200)) / 400.0
    image_array[image_array > 1] = 1.0
    image_array[image_array < 0] = 0.0

    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    seg = sitk.ReadImage(dcm_seg)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array = np.squeeze(seg_array)

    contours, hierarchy = cv2.findContours(seg_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    cnt = contours[0]
    cv2.drawContours(image_array, [cnt], 0, (0, 255, 0), 1)
    cv2.drawContours(image_array, contours, -1, (0, 255, 0), 1)
    cv2.imwrite("./track/"+"19.tif", image_array)


drawContour()

import cv2
img_root = './track/'
fps = 5
size=(512,512)
fourcc = cv2.VideoWriter_fourcc(*'X264')
videoWriter = cv2.VideoWriter('./track/track_results.mp4',fourcc,fps,size)
for i in range(1,20):
    frame = cv2.imread(img_root+str(i)+'.tif')
    videoWriter.write(frame)
videoWriter.release()

