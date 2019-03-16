#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition

"""
训练lanenet模型
"""

import argparse
import math
import os
import os.path as ops
import time

import cv2
import glog as log
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_merge_model
from data_provider import lanenet_data_processor


CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='vgg')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    parser.add_argument('--initial_step', type=str, help='The initial global step', default=0)

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def train_net(dataset_dir, weights_path=None, net_flag='vgg', initial_step=0):
    """

    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """

    train_dataset_file = ops.join(dataset_dir, '7-3_random_train.txt')
    val_dataset_file = ops.join(dataset_dir, '7-3_random_val.txt')

    # train_dataset_file = ops.join(dataset_dir, '9-1_train.txt')
    # val_dataset_file = ops.join(dataset_dir, '9-1_val.txt')

    assert ops.exists(train_dataset_file)

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    with tf.device('/gpu:0'):
    # with tf.device('/cpu:0'):
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                             CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                             shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                    CFG.TRAIN.IMG_WIDTH, 1],
                                             name='binary_input_label')
        instance_label_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                      CFG.TRAIN.IMG_WIDTH],
                                               name='instance_input_label')
        # binary_seg_img_tensor = tf.placeholder(dtype=tf.uint8,
        #                                      shape=[CFG.TRAIN.IMG_HEIGHT,
        #                                             CFG.TRAIN.IMG_WIDTH, 1])

        phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

        net = lanenet_merge_model.LaneNet(net_flag=net_flag, phase=phase)

        # calculate the loss
        compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label_tensor,
                                       instance_label=instance_label_tensor, name='lanenet_model')
        total_loss = compute_ret['total_loss']
        binary_seg_loss = compute_ret['binary_seg_loss']
        disc_loss = compute_ret['discriminative_loss']
        pix_embedding = compute_ret['instance_seg_logits']

        counts = compute_ret['counts']

        # calculate the accuracy
        out_logits = compute_ret['binary_seg_logits']
        out_logits = tf.nn.softmax(logits=out_logits)
        out_logits_out = tf.argmax(out_logits, axis=-1)  # transform a 2-channel feature map into a binary image
        out = tf.argmax(out_logits, axis=-1)
        out = tf.expand_dims(out, axis=-1)

        idx = tf.where(tf.equal(binary_label_tensor, 1))  # select the Positive Pixels in GT image
        pix_cls_ret = tf.gather_nd(out, idx)    # slice out the corresponding pixels in output image
        accuracy = tf.count_nonzero(pix_cls_ret)  # True Positive
        accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))
        # Accuracy = TP / (TP + FN), ie. Recall

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                                   CFG.TRAIN.LR_DECAY_STEPS, CFG.TRAIN.LR_DECAY_RATE, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(loss=total_loss,
                                                                    var_list=tf.trainable_variables(),
                                                                    global_step=global_step)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/tusimple_lanenet'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)
    img_output_dir = f'output/{net_flag}_{train_start_time}'

    # Set tf restorer
    mobile_pretrained_path = 'model/mobilenet/mobilenet_v2_1.0_224.ckpt'
    reader = tf.train.NewCheckpointReader(mobile_pretrained_path)
    restore_dict = dict()
    for v in tf.trainable_variables():
        s = v.name.split(':')[0]
        i = s.find('MobilenetV2')
        if i != -1:
            tensor_name = s[i:]
            # print(tensor_name)
            if reader.has_tensor(tensor_name):
                # print('has tensor ', tensor_name)
                restore_dict[tensor_name] = v

    pretrained_saver = tf.train.Saver(restore_dict, name="pretrained_saver")



    # Set tf summary
    tboard_save_path = f'tboard/tusimple_lanenet/{net_flag}/{train_start_time}'
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    # train_bin_seg_img = tf.summary.image('Train Binary Segmentation', tensor=binary_seg_img_tensor)
    # train_raw_img = tf.summary.image('Train Raw Image', gt_imgs[0] + VGG_MEAN)
    # val_bin_seg_img = tf.summary.image('Binary Segmentation', )
    # val_bin_seg_img = tf.summary.image('Binary Segmentation', )
    # val_bin_seg_img = tf.summary.image('Binary Segmentation', )
    # val_bin_seg_img = tf.summary.image('Binary Segmentation', )

    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_image.png', gt_imgs[0] + VGG_MEAN)
    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_binary_label.png', binary_gt_labels[0] * 255)
    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_instance_label.png', instance_gt_labels[0])
    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_binary_seg_img.png', binary_seg_img[0] * 255)
    #
    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_embedding.png', embedding_image)
    #
    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_image_VAL.png', gt_imgs_val[0] + VGG_MEAN)
    # cv2.imwrite(f'output/{train_start_time}_{net_flag}_binary_seg_img_VAL.png', val_binary_seg_img[0] * 255)


    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar, train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar])  # , train_bin_seg_img
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar,
                                             val_binary_seg_loss_scalar, val_instance_seg_loss_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # summary_writer = tf.summary.FileWriter(tboard_save_path)
    # summary_writer.add_graph(sess.graph)

    summary_writer = tf.summary.FileWriter(tboard_save_path, sess.graph)

    # Set the training parameters
    train_steps = CFG.TRAIN.STEPS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()
            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue
        elif net_flag == 'mobile' and weights_path is None:
            pass
            # pretrained_saver.restore(sess=sess, save_path=mobile_pretrained_path)

        train_cost_time_mean = []
        val_cost_time_mean = []
        for step in range(int(initial_step), train_steps):
            # training part
            t_start = time.time()

            with tf.device('/gpu:0'):
                raw_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(CFG.TRAIN.BATCH_SIZE)
                # gt_imgs = [cv2.resize(tmp,
                #                       dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                #                       dst=tmp,
                #                       interpolation=cv2.INTER_LINEAR)
                #            for tmp in gt_imgs]

                gt_imgs = [tmp - VGG_MEAN for tmp in raw_imgs]
                # binary_gt_labels = [cv2.resize(tmp,
                #                                dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                #                                dst=tmp,
                #                                interpolation=cv2.INTER_NEAREST)
                #                     for tmp in binary_gt_labels]
                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]
                # instance_gt_labels = [cv2.resize(tmp,
                #                                  dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                #                                  dst=tmp,
                #                                  interpolation=cv2.INTER_NEAREST)
                #                       for tmp in instance_gt_labels]
            phase_train = 'train'

            _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img, ct = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out, counts],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: phase_train})

            # binary_label_tensor = tf.assign(tf.multiply(binary_seg_img[0], 255))

            print(ct)
            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                log.error('cost is: {:.5f}'.format(c))
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                log.error('instance cost is: {:.5f}'.format(instance_loss))
                # cv2.imwrite(f'output/{train_start_time}_{net_flag}_nan_image.png', gt_imgs[0] + VGG_MEAN)
                # cv2.imwrite(f'output/{train_start_time}_{net_flag}_nan_instance_label.png', instance_gt_labels[0])
                # cv2.imwrite(f'output/{train_start_time}_{net_flag}_nan_binary_label.png', binary_gt_labels[0] * 255)
                return

            if step % 50 == 0:
                if not os.path.exists(img_output_dir):
                    os.mkdir(img_output_dir)
                print("Image Updated...")
                cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_TRAIN_raw.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_TRAIN_binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_TRAIN_instance_label.png', instance_gt_labels[0])
                cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_TRAIN_bin_seg.png', binary_seg_img[0] * 255)

                for i in range(4):
                    embedding[0][:, :, i] = minmax_scale(embedding[0][:, :, i])
                embedding_image = np.array(embedding[0], np.uint8)
                cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_TRAIN_embedding.png', embedding_image)

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=step)

            # validation part
            with tf.device('/gpu:0'):
                gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val \
                    = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE)
                # gt_imgs_val = [cv2.resize(tmp,
                #                           dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                #                           dst=tmp,
                #                           interpolation=cv2.INTER_LINEAR)
                #                for tmp in gt_imgs_val]
                gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
                # binary_gt_labels_val = [cv2.resize(tmp,
                #                                    dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                #                                    dst=tmp)
                #                         for tmp in binary_gt_labels_val]
                binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
                # instance_gt_labels_val = [cv2.resize(tmp,
                #                                      dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                #                                      dst=tmp,
                #                                      interpolation=cv2.INTER_NEAREST)
                #                           for tmp in instance_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            c_val, val_summary, val_accuracy, val_binary_seg_loss, val_instance_seg_loss, embedding, val_binary_seg_img, val_ct = \
                sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss, pix_embedding, out_logits_out, counts],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})

            if step % 50 == 0:
                if not os.path.exists(img_output_dir):
                    os.mkdir(img_output_dir)
                for i in range(CFG.TRAIN.VAL_BATCH_SIZE):
                    cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_VAL_{i}_raw.png',
                                gt_imgs_val[i] + VGG_MEAN)
                    cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_VAL_{i}_bin_seg.png',
                                val_binary_seg_img[i] * 255)
                    for j in range(4):
                        embedding[i][:, :, j] = minmax_scale(embedding[i][:, :, j])
                    embedding_image = np.array(embedding[i], np.uint8)
                    cv2.imwrite(img_output_dir + f'/{train_start_time}_{net_flag}_VAL_{i}_embedding.png',
                                embedding_image)

            summary_writer.add_summary(val_summary, global_step=step)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if step % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Step: {:d} total_loss= {:6f} binary_seg_loss= {:6f} instance_seg_loss= {:6f} accuracy= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(step + 1, c, binary_loss, instance_loss, train_accuracy,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if step % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                log.info('Step_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                         'instance_seg_loss= {:6f} accuracy= {:6f} '
                         'mean_cost_time= {:5f}s '.
                         format(step + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_accuracy,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if step % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=step)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net, initial_step=args.initial_step)
