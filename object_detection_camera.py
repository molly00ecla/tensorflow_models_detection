# Copyright 2017 Jin Fagang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================
"""
this file detect objects from live camera.
you gonna need OpenCV to run this code.
"""
import cv2
from PIL import Image
import os
import tensorflow as tf
import numpy as np
from utils import label_map_util
from utils import visualization_utils as vis_util


def load_image_to_numpy_array_uint(img):
    (im_width, im_height) = img.size
    return np.array(img.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)


def camera_live_detect():
    graph_path = 'out_of_box_graph/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
    label_file = 'data/mscoco_label_map.pbtxt'
    num_classes = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('load graph success!')
    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories=categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            try:
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    image = cv2.cvtColor(frame, cv2.CAP_MODE_RGB)
                    image_np = np.array(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded}
                    )
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index=category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8
                    )
                    cv2.imshow('video', image_np)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print('open camera fail: ', e)


if __name__ == '__main__':
    camera_live_detect()
