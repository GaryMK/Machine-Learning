# @author: GaryMK
#  @EMAIL: chenxingmk@gmail.com
#  @Date: 2021/2/13 23:27
#  @Version: 1.0
#  @Description:
import cv2
vidcap = cv2.VideoCapture('video/origin.mp4')
success, image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    cv2.imwrite("source/frame%d.jpg" % count, image)  # save frame as JPEG file
    if cv2.waitKey(10) == 27:
        break
    count += 1
