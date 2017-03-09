#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import mymodel
import myutils
import tensorflow as tf
import TensorflowUtils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("source_dir", "../../dataset/PedCut2013_SegmentationDataset/data/completeData/left_images",
                       "Directory name to read the source images")
tf.flags.DEFINE_string("source_mask_dir",
                       "../../dataset/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth/",
                       "Directory name to read the source mask images")

tf.flags.DEFINE_string("target_dir", "../../dataset/CITYSCAPES/CITY/instance_image_context",
                       "Directory name to read the target images")
tf.flags.DEFINE_string("target_heatmap_dir", "../../dataset/CITYSCAPES/CITY/instance_heatmap_context",
                       "Directory name to read the target heatmap")
tf.flags.DEFINE_string("target_semantic_dir", "../../dataset/CITYSCAPES/CITY/instance_heatmap_context",
                       "Directory name to read the target semantic")

tf.flags.DEFINE_string("result_dir", "./result/", "Directory name to the result")
tf.flags.DEFINE_string("logs_dir", "./logs/", "Directory name to the logs")
tf.flags.DEFINE_string("test_dir", "./test/", "Directory name to the test")

tf.flags.DEFINE_string("vgg_dir", "../../checkpoint/", "Path to vgg model mat")
tf.flags.DEFINE_string("model_dir", "../../checkpoint/image synthesis/relative", "path to model directory")

tf.flags.DEFINE_integer("batch_size", 9, "batch size for training")
tf.flags.DEFINE_integer("sample_shape", 3, "The size of sample images [1]")

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "Momentum term of adam [0.5]")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE_h = 256
IMAGE_SIZE_w = 512


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def initialize_and_saver():
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=4)
    # summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...{}".format(ckpt.model_checkpoint_path))
    else:
        print('No model founded')

    return sess, saver


def main():

    myutils.directory_check(FLAGS)
    input_fetcher = myutils.InputFetcher(flags=FLAGS, size=None)

    """
    # FCN
    # image should be in range 0-255
    # will be subtracted by mean(vgg) during inference
    """
    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_h, IMAGE_SIZE_w, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE_h, IMAGE_SIZE_w], name="annotation")
    with tf.variable_scope("fcn"):
        fcn = mymodel.FCN(flags=FLAGS, num_of_class=2)
        pred_annotation, logits = fcn.inference(image, keep_probability)
        loss_soft_constrain = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits, annotation)))
    fcn_variable = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='fcn')

    """
    # Discriminator
    # image should be regularize in range -1~1
    """
    # TODO: flexible batch size
    images_true = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_h, IMAGE_SIZE_w, 1],
                                 name='real_images')
    images_false = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_h, IMAGE_SIZE_w, 1],
                                  name='false_images')
    with tf.variable_scope("dis"):
        dis = mymodel.Discriminator(flags=FLAGS, df_dim=64, images_true=images_true, images_false=images_false)
        loss_dis_constrain, pred_true, pred_false = dis.d_loss, dis.D, dis.D_
    dis_variable = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

    """
    # Loss
    """
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)\
        .minimize(loss_dis_constrain, var_list=dis_variable)
    fcn_optim = train(loss_soft_constrain, fcn_variable)

    sess, saver = initialize_and_saver()

    if FLAGS.mode == "train":
        train_step = 0
        input_fetcher.shuffle()
        for epoch in xrange(MAX_ITERATION):
            for batch_idx in range(input_fetcher.batch_size):
                print('[{:d}/{:d}] [{:d}/{:d}] step:{:d}'.format(
                    epoch, MAX_ITERATION, batch_idx, input_fetcher.batch_size, train_step))

                target_images, target_heatmaps, target_semantics = input_fetcher.fetch(batch_idx)

                feed_dict_fcn = {image: target_images, annotation: target_heatmaps, keep_probability: 0.85}

                # TODO: normalize
                heatmaps_true = np.expand_dims(target_heatmaps, axis=3)
                pred = sess.run(pred_annotation, feed_dict={image: target_images, keep_probability: 1.0})
                pred = np.array(pred)
                feed_dict_dis = {images_true: heatmaps_true, images_false: pred}

                if train_step % 5 == 0:
                    train_loss = sess.run(loss_soft_constrain, feed_dict=feed_dict_fcn)
                    print("Step: %d, Soft_loss:%g" % (train_step, train_loss))
                    train_loss = sess.run(loss_dis_constrain, feed_dict=feed_dict_dis)
                    print("Step: %d, GAN_loss:%g" % (train_step, train_loss))

                if train_step % 100 == 0:
                    p_true, p_false = sess.run([pred_true, pred_false], feed_dict=feed_dict_dis)
                    myutils.logs_check2(flags=FLAGS, train_step=train_step, images_true=heatmaps_true,
                                        images_false=pred, p_true=p_true, p_false=p_false)

                    myutils.logs_check(flags=FLAGS, train_step=train_step, target_images=target_images,
                                       target_heatmaps=target_heatmaps, target_semantics=target_heatmaps, pred=np.squeeze(pred, axis=3))
                    # summary_writer.add_summary(summary_str, step)

                if train_step % 300 == 1:
                    print('checkpoint')
                    saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt"), train_step)

                """
                # FCN update
                """
                sess.run(fcn_optim, feed_dict=feed_dict_fcn)

                """
                # Discriminator update
                """
                sess.run(d_optim, feed_dict=feed_dict_dis)

                train_step += 1

if __name__ == '__main__':
    main()
