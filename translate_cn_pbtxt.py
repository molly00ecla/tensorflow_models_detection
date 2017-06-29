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
this file will load a *.pdtxt file to dict, and translate specific key
into Chinese. Automatically translate all display name into Chinese.
"""
import tensorflow as tf
from google.protobuf import text_format
from protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from translation import baidu, google, youdao, iciba
from google.protobuf import text_format
from protos import string_int_label_map_pb2


def load_label_map(file_path):
    with tf.gfile.GFile(file_path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    return label_map


def translate():

    label_map = load_label_map('data/pet_label_map.pbtxt')

    translated_label_map = StringIntLabelMap()
    i = 0
    for item in label_map.item:
        translated_item = translated_label_map.item.add()

        # TODO: edit the field name accordingly, name or display name
        r = youdao(str(item.name).replace('_', ' '), dst='zh-CN')
        print(r)
        translated_item.name = r.encode('utf-8')
        translated_item.id = item.id

        if i > 4:
            break
        i += 1

    with open('data/pet_cn_label_map.pbtxt', 'w') as f:
        text_format.PrintMessage(translated_label_map, out=f, as_utf8=True)
        print('translated!')

if __name__ == '__main__':
    translate()