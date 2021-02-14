# @author: GaryMK
#  @EMAIL: chenxingmk@gmail.com
#  @Date: 2021/2/14 0:29
#  @Version: 1.0
#  @Description:
import time
import os
import cv2
import re


def resort(list):
    for i in range(len(list)-1):
        for j in range(len(list)-1):
            if int(re.findall(r'\d+', list[j])[0]) > int(re.findall(r'\d+', list[j+1])[0]):
                list[j], list[j+1] = list[j+1], list[j]
    return list

# path = r'C:\Users\Administrator\Desktop\\'# 文件路径
def picvideo(path, size):
    filelist = os.listdir(path)  # 获取该目录下的所有文件名
    filelist = resort(filelist)


    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次]
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 27
    # size = (591,705) #图片的分辨率片
    file_path = 'output/output.mp4'  # 导出路径
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(file_path, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.jpg'):  # 判断图片后缀是否是.png
            item = path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频

    video.release()  # 释放

Dir = 'replaced/'
img = cv2.imread(Dir + 'replaced_frame0.jpg')
imgInfo = img.shape
size = (imgInfo[1], imgInfo[0])
print("imgeSize:")
print(size)
picvideo(Dir, size)
