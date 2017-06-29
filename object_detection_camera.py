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
from object_detection import detect_single_image
import cv2
from PIL import Image


def camera_live_detect():
    graph_path = 'out_of_box_graph/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
    label_file = 'data/mscoco_label_map.pbtxt'
    num_classes = 90

    try:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.CAP_MODE_RGB)
            print(image)

            # detect_single_image(graph_path, label_file, num_classes)
    except Exception as e:
        print('no camera on this device: ', e)


if __name__ == '__main__':
    camera_live_detect()
