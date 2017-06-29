# TensorFlow Detection
> 这个项目是对原tensorflow models的一个剥离版本，原来的感觉有点乱，在这里我会一直维护，基本上会保持和官方repo一样的更新。

## 开箱即用的预测模型



官方提供了SSD， Faster-RCNN的预训练模型，并且使用了不同的特征提取网络。从[这里](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)可以找到对应的下载链接。目前支持这些组合：

* SSD_MobileNet
* SSD_Inception_V2
* RFCN_ResNet101
* faster_rcnn_ResNet101
* faster_rcnn_Inception_ResNet_v2

可想而知，最后一个模型最大也最慢，但是精度是最高的，第一个速度最快，下面的预测结果就是SSD和MobileNet预测的：

![PicName](http://ofwzcunzi.bkt.clouddn.com/naJu5I5d0iFafTD6.png)

我整理了一个一键预测的代码，从 `test_images/ `连续预测两张图像，可以稍微修改一下就可以直接预测单张图片或者调用camera进行预测。速度感觉…还可以，我的是CPU。预测代码是 `object_detection.py`.

## 训练KITTI数据集

有时间我在把训练KITTI数据集的部分写完，其实这很简单，就是把图片转成tfrecord然后写proto修改参数。不过说实话，检测网络普遍比较复杂。



## Copyright

本文由在当地较为英俊的男子原创，转载请注明出处。如有疑问或者想交流，可以加入微信: `jintianiloveu`

