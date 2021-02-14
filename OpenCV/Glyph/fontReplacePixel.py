# @author: GaryMK
#  @EMAIL: chenxingmk@gmail.com
#  @Date: 2021/2/14 0:28
#  @Version: 1.0
#  @Description:
from PIL import Image, ImageDraw, ImageFont
import cv2
import os


def draw(pic):
    img = cv2.imread('source/' + pic)
    img = img[:, :, (2, 1, 0)]

    blank = Image.new("RGB", [len(img[0]), len(img)], "white")
    drawObj = ImageDraw.Draw(blank)

    n = 10

    font = ImageFont.truetype('C:/Windows/Fonts/Microsoft YaHei UI/msyhbd.ttc', size=n - 1)

    for i in range(0, len(img), n):
        for j in range(0, len(img[i]), n):
            text = '晨星'
            drawObj.ink = img[i][j][0] + img[i][j][1] * 256 + img[i][j][2] * 256 * 256
            drawObj.text([j, i], text[int(j / n) % len(text)], font=font)
            print('完成处理——', i, j)

    blank.save('replaced/replaced_' + pic, 'jpeg')


filelist = os.listdir('source')
for file in filelist:
    draw(file)
