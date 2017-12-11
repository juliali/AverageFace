本项目教你用opencv（dlib）构造平均脸

前提条件
=========
(1) 安装Python 2.7

(2) 安装pip

(3) 安装dlib,cv2等所需library

安装dlib有众多方法：

可以通过pip直接安装：
>   pip install dlib

也可以下载源代码后直接编译：
>   python setup.py install

推荐pip，因为这样所有dlib所需要的dependencies可以一并安装


构造平均脸
===========

[1] 基于多张单人照片求平均脸(不区分性别）

第一步：将要平均的照片放入${AverageFace_root}\input_data\${sub_dir_name}文档，确保图片为jpg格式。
    例如： input_data\president

第二步：在终端运行 face_landmark_detection.py
${AverageFace_root}> python scripts\face_landmark_detection.py input_data\president

第三步：在终端运行 faceAverage.py

${AverageFace_root}> python scripts\average_face_generation.py input_data\president output_data\president.jpg

这样就能看到制作成功的平均脸了！


[2] 基于多张单人照片求平均脸（区分性别）

前提条件：安装caffe， 参照 http://caffe.berkeleyvision.org/installation.html

第一步：将要平均的照片放入${AverageFace_root}\input_data\${sub_dir_name}文档，确保图片为jpg格式。
    例如： input_data\president

第二步：在终端运行 face_landmark_detection.py 加上'gender'参数

${AverageFace_root}> python scripts\face_landmark_detection.py input_data\president gender

第三步：在终端运行 faceAverage.py 指明gender目录

${AverageFace_root}> python scripts\average_face_generation.py input_data\president\male output_data\president_male.jpg


[3] 将一张大合影中的人脸切分成独立的照片

第一步：将要切分的大合影照片放入${AverageFace_root}\input_data\${sub_dir_name}
      例如： input_data\team\heying.jpg

第二步：运行 splitfaces.py 进行切分

${AverageFace_root}> python scripts\splitfaces.py input_data\team\heying.jpg input_data\team_splited

完成后在input_data\team_splited\目录下生成了众多jpg文件，每张包含一个人脸

NOTE：切分完之后，之后将input_data\team_splited 作为数据源，可以在其上进行[1]或[2]的操作