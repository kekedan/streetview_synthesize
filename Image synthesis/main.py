#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import scipy.misc
from glob import glob
import os
import utils as ut
from model import Discriminator
import tensorflow as tf
import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("source_dir", "/home/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_images",
                    "Directory name to read the source images")
tf.flags.DEFINE_string("source_mask_dir", "/home/andy/dataset/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth/",
                    "Directory name to read the source mask images")

tf.flags.DEFINE_string("target_dir", "/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image",
                    "Directory name to read the target images")
tf.flags.DEFINE_string("target_heatmap_dir", "/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label2_big",
                    "Directory name to read the images")
tf.flags.DEFINE_string("target_semantic_dir", "/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label",
                    "Directory name to read the images")

tf.flags.DEFINE_string("result_dir", "./result/",
                    "Directory name to read the images")
tf.flags.DEFINE_string("logs_dir", "./logs/",
                    "Directory name to read the images")

tf.flags.DEFINE_string("model_dir", "/home/andy/Documents/Github/streetview_synthesize/FCN/",
                    "Directory name to read the images")

tf.flags.DEFINE_integer("sample_size", 80, "The size of sample images [1]")
tf.flags.DEFINE_integer("batch_size", 4, "The size of sample images [1]")
tf.flags.DEFINE_integer("sample_shape", 2, "The size of sample images [1]")

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE_h = 256
IMAGE_SIZE_w = 512


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    #model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
    model_data = scipy.io.loadmat(os.path.join(FLAGS.model_dir, 'imagenet-vgg-verydeep-19.mat'))

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main():

    data_source = sorted(glob(os.path.join(FLAGS.source_dir, "*.png")))[0:FLAGS.sample_size]
    # If mask is gray image, the is_gray flag need to be set True
    data_source_mask = sorted(glob(os.path.join(FLAGS.source_mask_dir, "*.png")))[0:FLAGS.sample_size]

    data_target = sorted(glob(os.path.join(FLAGS.target_dir, "*.png")))[0:FLAGS.sample_size]
    data_target_heatmap = sorted(glob(os.path.join(FLAGS.target_heatmap_dir, "*.png")))[0:FLAGS.sample_size]
    data_target_semantic = sorted(glob(os.path.join(FLAGS.target_semantic_dir, "*.png")))[0:FLAGS.sample_size]

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_h, IMAGE_SIZE_w, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE_h, IMAGE_SIZE_w], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    loss_soft_constrain = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits, annotation)))
    trainable_var = tf.trainable_variables()

    train_op = train(loss_soft_constrain, trainable_var)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    discriminator = Discriminator(sess, batch_size=FLAGS.batch_size, output_size_h=256,
                                  output_size_w=512, c_dim=4,
                                  dataset_name=FLAGS.dataset, checkpoint_dir=FLAGS.checkpoint_dir,
                                  dataset_dir=FLAGS.dataset_dir)

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=4)
    # summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...{}".format(ckpt.model_checkpoint_path))

    if FLAGS.mode == "train":
        train_step = 0
        for epoch in xrange(MAX_ITERATION):
            for batch_idx in range(FLAGS.sample_size / FLAGS.batch_size):
                print('[{:d}/{:d}] [{:d}/{:d}] step:{:d}'.format(
                    epoch, MAX_ITERATION, batch_idx, FLAGS.sample_size / FLAGS.batch_size, train_step))

                target_files = data_target[batch_idx * FLAGS.batch_size:(batch_idx + 1) * FLAGS.batch_size]
                targets = np.array([ut.get_image(target) for target in target_files])

                target_heatmap_files = data_target_heatmap[batch_idx * FLAGS.batch_size:(batch_idx + 1) * FLAGS.batch_size]
                target_heatmaps = np.array([ut.get_image(target_heatmap) for target_heatmap in target_heatmap_files])
                # TODO : better way
                target_heatmaps /= 255

                target_semantic_files = data_target_semantic[batch_idx * FLAGS.batch_size:(batch_idx + 1) * FLAGS.batch_size]
                target_semantics = np.array([ut.get_image(target_semantic) for target_semantic in target_semantic_files])

                feed_dict = {image: targets, annotation: target_heatmaps, keep_probability: 0.85}

                sess.run(train_op, feed_dict=feed_dict)

                if train_step % 5 == 0:
                    train_loss = sess.run(loss_soft_constrain, feed_dict=feed_dict)
                    print("Step: %d, Soft_loss:%g" % (train_step, train_loss))

                if train_step % 30 == 0:
                    scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{}_target.png'.format(train_step)),
                                      ut.merge(targets, (FLAGS.sample_shape, FLAGS.sample_shape)))
                    scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{}_target_heatmap.png'.format(train_step)),
                                      ut.merge(target_heatmaps, (FLAGS.sample_shape, FLAGS.sample_shape), is_gray=True))
                    scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{}_target_semantic.png'.format(train_step)),
                                      ut.label_visualize(
                                          ut.merge(target_semantics, (FLAGS.sample_shape, FLAGS.sample_shape),
                                                   is_gray=True)))

                    pred = sess.run(pred_annotation, feed_dict={image: targets, annotation: target_heatmaps,
                                                                keep_probability: 1.0})

                    pred = np.squeeze(pred, axis=3)
                    scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{}_target_pred.png'.format(train_step)),
                                      ut.merge(pred, (FLAGS.sample_shape, FLAGS.sample_shape), is_gray=True))
                    # summary_writer.add_summary(summary_str, step)
                if train_step % 100 == 1:
                    print('checkpoint')
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", train_step)


                #img_ret = blend(targets[i], sources[i], source_masks[i], offset=(0, 0))
                #if img_ret is not False:
                #    scipy.misc.imsave(os.path.join(output_dir, 'img_ret{}.png'.format(i)), img_ret)

                train_step += 1

if __name__ == '__main__':
    main()
