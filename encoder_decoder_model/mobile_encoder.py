#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-8
# @Author  : Shenhan Qian
# @Site    : https://github.com/ShenhanQian
# @File    : mobielNetV2_encoder.py
# @IDE: PyCharm Community Edition
"""
实现一个基于MobileNet_V2的特征编码类
"""
from collections import OrderedDict
import tensorflow as tf
from encoder_decoder_model import cnn_basenet

import sys

# sys.path.append('../tf_models/models/research/slim')

from nets.mobilenet.conv_blocks import expanded_conv
from nets.mobilenet import mobilenet_v2


class MobielnetV2Encoder(cnn_basenet.CNNBaseModel):
    """
    实现了一个基于v的特征编码类
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(MobielnetV2Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def encode(self, input_tensor, name):
        """
        根据MobileNet框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :param flags:
        :return: 输出MobileNet编码特征
        """
        ret = OrderedDict()

        with tf.variable_scope(name):
            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
                net, end_points = mobilenet_v2.mobilenet(input_tensor, base_only=True)

            # # Version B
            # ret['layer_5'] = dict()
            # ret['layer_5']['data'] = end_points['layer_5']
            # ret['layer_5']['shape'] = end_points['layer_5'].get_shape().as_list()
            #
            #
            # ret['layer_8'] = dict()
            # ret['layer_8']['data'] = end_points['layer_8']
            # ret['layer_8']['shape'] = end_points['layer_8'].get_shape().as_list()
            #
            #
            # ret['layer_18'] = dict()
            # ret['layer_18']['data'] = end_points['layer_18']
            # ret['layer_18']['shape'] = end_points['layer_18'].get_shape().as_list()



            # Version A
            ret['layer_7'] = dict()
            ret['layer_7']['data'] = end_points['layer_7']
            ret['layer_7']['shape'] = end_points['layer_7'].get_shape().as_list()


            ret['layer_14'] = dict()
            ret['layer_14']['data'] = end_points['layer_14']
            ret['layer_14']['shape'] = end_points['layer_14'].get_shape().as_list()


            ret['layer_19'] = dict()
            ret['layer_19']['data'] = end_points['layer_19']
            ret['layer_19']['shape'] = end_points['layer_19'].get_shape().as_list()

            # ret['end_points'] = end_points

        return ret

    def encode_0(self, input_tensor, name):
        """
        根据MobileNet框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :param flags:
        :return: 输出MobileNet编码特征
        """
        ret = OrderedDict()

        with tf.variable_scope(name):
            conv1 = self.conv2d(inputdata=input_tensor, out_channel=32,
                                kernel_size=3, stride=2, use_bias=False, name='conv1')  # 128

            e_conv_1 = expanded_conv(conv1, expansion_size=16, num_outputs=16)

            e_conv_2 = expanded_conv(e_conv_1, stride=2, num_outputs=24)  # 64
            e_conv_3 = expanded_conv(e_conv_2, stride=1, num_outputs=24)
            e_conv_4 = expanded_conv(e_conv_3, stride=2, num_outputs=32)  # 32

            ret['e_conv_4'] = dict()
            ret['e_conv_4']['data'] = e_conv_4
            ret['e_conv_4']['shape'] = e_conv_4.get_shape().as_list()

            e_conv_5 = expanded_conv(e_conv_4, stride=1, num_outputs=32)
            e_conv_6 = expanded_conv(e_conv_5, stride=1, num_outputs=32)
            e_conv_7 = expanded_conv(e_conv_6, stride=2, num_outputs=64)  # 16

            ret['e_conv_7'] = dict()
            ret['e_conv_7']['data'] = e_conv_7
            ret['e_conv_7']['shape'] = e_conv_7.get_shape().as_list()

            e_conv_8 = expanded_conv(e_conv_7, stride=1, num_outputs=64)
            e_conv_9 = expanded_conv(e_conv_8, stride=1, num_outputs=64)
            e_conv_10 = expanded_conv(e_conv_9, stride=1, num_outputs=64)
            e_conv_11 = expanded_conv(e_conv_10, stride=1, num_outputs=96)
            e_conv_12 = expanded_conv(e_conv_11, stride=1, num_outputs=96)
            e_conv_13 = expanded_conv(e_conv_12, stride=1, num_outputs=96)
            e_conv_14 = expanded_conv(e_conv_13, stride=2, num_outputs=160)  # 8
            e_conv_15 = expanded_conv(e_conv_14, stride=1, num_outputs=160)
            e_conv_16 = expanded_conv(e_conv_15, stride=1, num_outputs=160)
            e_conv_17 = expanded_conv(e_conv_16, stride=1, num_outputs=320)

            ret['e_conv_17'] = dict()
            ret['e_conv_17']['data'] = e_conv_17
            ret['e_conv_17']['shape'] = e_conv_17.get_shape().as_list()

        return ret

if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    encoder = MobielnetV2Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')

    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))


    # end_points = ret['end_points']
    # for item in end_points:
    #     print(item, end_points[item])


