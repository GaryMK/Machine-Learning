---
typora-root-url: README
---

# OpenCV

OpenCV project practices.

### Glyph

表白神器，将输入的图像视频里的像素替换成你心仪对象的名字

#### 环境

1. Python 3.7

2. OpenCV

```python
# 在conda环境中通过下面的命令一键式安装OpenCV
conda install -c menpo opencv
```

3. PIL

~~~
# 安装pillow
conda install pillow
~~~

#### 运行

**videoToFrame.py**

提取出videa文件夹下指定视频的每一帧输出到source文件夹

**fontReplacePixel.py**

将每张图片的图案替换成相应的文字，输出到replace

**compositeVidea.py**

将多张图片合成相应视频，输出到output文件夹

#### 样图

![image-20210214125135168](./glyph.png)