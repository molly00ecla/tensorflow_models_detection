# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/faster_rcnn.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import anchor_generator_pb2 as protos_dot_anchor__generator__pb2
from protos import box_predictor_pb2 as protos_dot_box__predictor__pb2
from protos import hyperparams_pb2 as protos_dot_hyperparams__pb2
from protos import image_resizer_pb2 as protos_dot_image__resizer__pb2
from protos import losses_pb2 as protos_dot_losses__pb2
from protos import post_processing_pb2 as protos_dot_post__processing__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/faster_rcnn.proto',
  package='protos',
  syntax='proto2',
  serialized_pb=_b('\n\x18protos/faster_rcnn.proto\x12\x06protos\x1a\x1dprotos/anchor_generator.proto\x1a\x1aprotos/box_predictor.proto\x1a\x18protos/hyperparams.proto\x1a\x1aprotos/image_resizer.proto\x1a\x13protos/losses.proto\x1a\x1cprotos/post_processing.proto\"\xad\t\n\nFasterRcnn\x12\x1f\n\x10\x66irst_stage_only\x18\x01 \x01(\x08:\x05\x66\x61lse\x12\x13\n\x0bnum_classes\x18\x03 \x01(\x05\x12+\n\rimage_resizer\x18\x04 \x01(\x0b\x32\x14.protos.ImageResizer\x12=\n\x11\x66\x65\x61ture_extractor\x18\x05 \x01(\x0b\x32\".protos.FasterRcnnFeatureExtractor\x12=\n\x1c\x66irst_stage_anchor_generator\x18\x06 \x01(\x0b\x32\x17.protos.AnchorGenerator\x12\"\n\x17\x66irst_stage_atrous_rate\x18\x07 \x01(\x05:\x01\x31\x12G\n*first_stage_box_predictor_conv_hyperparams\x18\x08 \x01(\x0b\x32\x13.protos.Hyperparams\x12\x30\n%first_stage_box_predictor_kernel_size\x18\t \x01(\x05:\x01\x33\x12,\n\x1f\x66irst_stage_box_predictor_depth\x18\n \x01(\x05:\x03\x35\x31\x32\x12\'\n\x1a\x66irst_stage_minibatch_size\x18\x0b \x01(\x05:\x03\x32\x35\x36\x12\x32\n%first_stage_positive_balance_fraction\x18\x0c \x01(\x02:\x03\x30.5\x12*\n\x1f\x66irst_stage_nms_score_threshold\x18\r \x01(\x02:\x01\x30\x12*\n\x1d\x66irst_stage_nms_iou_threshold\x18\x0e \x01(\x02:\x03\x30.7\x12&\n\x19\x66irst_stage_max_proposals\x18\x0f \x01(\x05:\x03\x33\x30\x30\x12/\n$first_stage_localization_loss_weight\x18\x10 \x01(\x02:\x01\x31\x12-\n\"first_stage_objectness_loss_weight\x18\x11 \x01(\x02:\x01\x31\x12\x19\n\x11initial_crop_size\x18\x12 \x01(\x05\x12\x1b\n\x13maxpool_kernel_size\x18\x13 \x01(\x05\x12\x16\n\x0emaxpool_stride\x18\x14 \x01(\x05\x12\x38\n\x1asecond_stage_box_predictor\x18\x15 \x01(\x0b\x32\x14.protos.BoxPredictor\x12#\n\x17second_stage_batch_size\x18\x16 \x01(\x05:\x02\x36\x34\x12+\n\x1dsecond_stage_balance_fraction\x18\x17 \x01(\x02:\x04\x30.25\x12<\n\x1csecond_stage_post_processing\x18\x18 \x01(\x0b\x32\x16.protos.PostProcessing\x12\x30\n%second_stage_localization_loss_weight\x18\x19 \x01(\x02:\x01\x31\x12\x32\n\'second_stage_classification_loss_weight\x18\x1a \x01(\x02:\x01\x31\x12\x34\n\x12hard_example_miner\x18\x1b \x01(\x0b\x32\x18.protos.HardExampleMiner\"S\n\x1a\x46\x61sterRcnnFeatureExtractor\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\'\n\x1b\x66irst_stage_features_stride\x18\x02 \x01(\x05:\x02\x31\x36')
  ,
  dependencies=[protos_dot_anchor__generator__pb2.DESCRIPTOR,protos_dot_box__predictor__pb2.DESCRIPTOR,protos_dot_hyperparams__pb2.DESCRIPTOR,protos_dot_image__resizer__pb2.DESCRIPTOR,protos_dot_losses__pb2.DESCRIPTOR,protos_dot_post__processing__pb2.DESCRIPTOR,])




_FASTERRCNN = _descriptor.Descriptor(
  name='FasterRcnn',
  full_name='protos.FasterRcnn',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='first_stage_only', full_name='protos.FasterRcnn.first_stage_only', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='protos.FasterRcnn.num_classes', index=1,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_resizer', full_name='protos.FasterRcnn.image_resizer', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='feature_extractor', full_name='protos.FasterRcnn.feature_extractor', index=3,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_anchor_generator', full_name='protos.FasterRcnn.first_stage_anchor_generator', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_atrous_rate', full_name='protos.FasterRcnn.first_stage_atrous_rate', index=5,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor_conv_hyperparams', full_name='protos.FasterRcnn.first_stage_box_predictor_conv_hyperparams', index=6,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor_kernel_size', full_name='protos.FasterRcnn.first_stage_box_predictor_kernel_size', index=7,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor_depth', full_name='protos.FasterRcnn.first_stage_box_predictor_depth', index=8,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_minibatch_size', full_name='protos.FasterRcnn.first_stage_minibatch_size', index=9,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_positive_balance_fraction', full_name='protos.FasterRcnn.first_stage_positive_balance_fraction', index=10,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_nms_score_threshold', full_name='protos.FasterRcnn.first_stage_nms_score_threshold', index=11,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_nms_iou_threshold', full_name='protos.FasterRcnn.first_stage_nms_iou_threshold', index=12,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.7),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_max_proposals', full_name='protos.FasterRcnn.first_stage_max_proposals', index=13,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=300,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_localization_loss_weight', full_name='protos.FasterRcnn.first_stage_localization_loss_weight', index=14,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_objectness_loss_weight', full_name='protos.FasterRcnn.first_stage_objectness_loss_weight', index=15,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='initial_crop_size', full_name='protos.FasterRcnn.initial_crop_size', index=16,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maxpool_kernel_size', full_name='protos.FasterRcnn.maxpool_kernel_size', index=17,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='maxpool_stride', full_name='protos.FasterRcnn.maxpool_stride', index=18,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_stage_box_predictor', full_name='protos.FasterRcnn.second_stage_box_predictor', index=19,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_stage_batch_size', full_name='protos.FasterRcnn.second_stage_batch_size', index=20,
      number=22, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=64,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_stage_balance_fraction', full_name='protos.FasterRcnn.second_stage_balance_fraction', index=21,
      number=23, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.25),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_stage_post_processing', full_name='protos.FasterRcnn.second_stage_post_processing', index=22,
      number=24, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_stage_localization_loss_weight', full_name='protos.FasterRcnn.second_stage_localization_loss_weight', index=23,
      number=25, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_stage_classification_loss_weight', full_name='protos.FasterRcnn.second_stage_classification_loss_weight', index=24,
      number=26, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hard_example_miner', full_name='protos.FasterRcnn.hard_example_miner', index=25,
      number=27, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=201,
  serialized_end=1398,
)


_FASTERRCNNFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='FasterRcnnFeatureExtractor',
  full_name='protos.FasterRcnnFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='protos.FasterRcnnFeatureExtractor.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_stage_features_stride', full_name='protos.FasterRcnnFeatureExtractor.first_stage_features_stride', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1400,
  serialized_end=1483,
)

_FASTERRCNN.fields_by_name['image_resizer'].message_type = protos_dot_image__resizer__pb2._IMAGERESIZER
_FASTERRCNN.fields_by_name['feature_extractor'].message_type = _FASTERRCNNFEATUREEXTRACTOR
_FASTERRCNN.fields_by_name['first_stage_anchor_generator'].message_type = protos_dot_anchor__generator__pb2._ANCHORGENERATOR
_FASTERRCNN.fields_by_name['first_stage_box_predictor_conv_hyperparams'].message_type = protos_dot_hyperparams__pb2._HYPERPARAMS
_FASTERRCNN.fields_by_name['second_stage_box_predictor'].message_type = protos_dot_box__predictor__pb2._BOXPREDICTOR
_FASTERRCNN.fields_by_name['second_stage_post_processing'].message_type = protos_dot_post__processing__pb2._POSTPROCESSING
_FASTERRCNN.fields_by_name['hard_example_miner'].message_type = protos_dot_losses__pb2._HARDEXAMPLEMINER
DESCRIPTOR.message_types_by_name['FasterRcnn'] = _FASTERRCNN
DESCRIPTOR.message_types_by_name['FasterRcnnFeatureExtractor'] = _FASTERRCNNFEATUREEXTRACTOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FasterRcnn = _reflection.GeneratedProtocolMessageType('FasterRcnn', (_message.Message,), dict(
  DESCRIPTOR = _FASTERRCNN,
  __module__ = 'protos.faster_rcnn_pb2'
  # @@protoc_insertion_point(class_scope:protos.FasterRcnn)
  ))
_sym_db.RegisterMessage(FasterRcnn)

FasterRcnnFeatureExtractor = _reflection.GeneratedProtocolMessageType('FasterRcnnFeatureExtractor', (_message.Message,), dict(
  DESCRIPTOR = _FASTERRCNNFEATUREEXTRACTOR,
  __module__ = 'protos.faster_rcnn_pb2'
  # @@protoc_insertion_point(class_scope:protos.FasterRcnnFeatureExtractor)
  ))
_sym_db.RegisterMessage(FasterRcnnFeatureExtractor)


# @@protoc_insertion_point(module_scope)
