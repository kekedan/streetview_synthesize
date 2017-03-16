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

tf.flags.DEFINE_string("target_dir", "../../dataset/Relative/train/relative_combine_low_image",
                       "Directory name to read the target images")
tf.flags.DEFINE_string("target_heatmap_dir", "../../dataset/Relative/train/relative_combine_low_heatmap",
                       "Directory name to read the target heatmap")
tf.flags.DEFINE_string("target_semantic_dir", "../../dataset/CITYSCAPES/CITY/relative_heatmap",
                       "Directory name to read the target semantic")

tf.flags.DEFINE_string("result_dir", "./result_D_low/", "Directory name to the result")
tf.flags.DEFINE_string("logs_dir", "./logs_D_low/", "Directory name to the logs")
tf.flags.DEFINE_string("valid_dir", "./valid_D_low/", "Directory name to the logs")
tf.flags.DEFINE_string("test_dir", "./test_D_low/", "Directory name to the test")

tf.flags.DEFINE_string("vgg_dir", "../../checkpoint/", "Path to vgg model mat")
tf.flags.DEFINE_string("model_dir", "../../checkpoint/image synthesis/relative_combine_D_low", "path to model directory")

tf.flags.DEFINE_integer("batch_size", 9, "batch size for training")
tf.flags.DEFINE_integer("sample_shape", 3, "The size of sample images [1]")

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "Momentum term of adam [0.5]")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(2e4 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE_h = 128
IMAGE_SIZE_w = 256


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
        pred_annotation_float = tf.cast(pred_annotation, tf.float32)
    fcn_variable = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='fcn')

    """
    # Discriminator
    # image should be regularize in range -1~1
    """
    # TODO: flexible batch size
    images_true = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_h, IMAGE_SIZE_w, 1],
                                 name='real_images')
    #images_false = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_h, IMAGE_SIZE_w, 1],
    #                              name='false_images')
    with tf.variable_scope("dis"):
        dis = mymodel.Discriminator(flags=FLAGS, df_dim=64, images_true=images_true, images_false=pred_annotation_float)
        loss_dis_constrain, pred_true, pred_false = dis.d_loss, dis.D, dis.D_
    dis_variable = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

    """
    # Loss
    """
    loss_soft_constrain_with_D = loss_dis_constrain * 0.001 + loss_soft_constrain * 0.999
    loss_dis_constrain_with_D = loss_dis_constrain * 0.001
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)\
        .minimize(loss_dis_constrain_with_D, var_list=dis_variable)
    fcn_optim = train(loss_soft_constrain_with_D, fcn_variable)

    sess, saver = initialize_and_saver()

    if FLAGS.mode == "train":
        train_step = 0
        input_fetcher.shuffle()
        for epoch in xrange(MAX_ITERATION):
            for batch_idx in range(input_fetcher.batch_size):
                print('[{:d}/{:d}] [{:d}/{:d}] step:{:d}'.format(
                    epoch, MAX_ITERATION, batch_idx, input_fetcher.batch_size, train_step))

                target_images, target_heatmaps, target_semantics = input_fetcher.fetch(batch_idx)

                #feed_dict_fcn = {image: target_images, annotation: target_heatmaps, keep_probability: 0.85}
                # TODO: normalize
                heatmaps_true = np.expand_dims(target_heatmaps, axis=3)
                feed_dict_fcn = {image: target_images, annotation: target_heatmaps,
                                 images_true: heatmaps_true, keep_probability: 0.85}

                #pred = sess.run(pred_annotation, feed_dict={image: target_images, keep_probability: 1.0})
                #pred = np.array(pred)
                #feed_dict_dis = {images_true: heatmaps_true, images_false: pred}

                if train_step % 5 == 0:
                    SOFT_LOSS, DIS_LOSS, SOFT_LOSS_WD, DIS_LOSS_WD = \
                        sess.run([loss_soft_constrain, loss_dis_constrain,
                                  loss_soft_constrain_with_D, loss_dis_constrain_with_D], feed_dict=feed_dict_fcn)
                    print("Step: %d, Soft_loss:%g, Dis_loss:%g, Gen_loss:%g, Dis_loss:%g" %
                          (train_step, SOFT_LOSS, DIS_LOSS, SOFT_LOSS_WD, DIS_LOSS_WD))
                    #train_loss = sess.run(loss_dis_constrain, feed_dict=feed_dict_dis)
                    #print("Step: %d, GAN_loss:%g" % (train_step, train_loss))

                if train_step % 100 == 0:
                    pred, p_true, p_false = sess.run([pred_annotation, pred_true, pred_false],
                                                     feed_dict={image: target_images, annotation: target_heatmaps,
                                                                images_true: heatmaps_true, keep_probability: 1.0})

                    pred = np.array(pred)
                    myutils.logs_check2(flags=FLAGS, train_step=train_step, images_true=heatmaps_true,
                                        images_false=pred, p_true=p_true, p_false=p_false)

                    myutils.logs_check(flags=FLAGS, train_step=train_step, target_images=target_images,
                                       target_heatmaps=target_heatmaps, target_semantics=target_heatmaps, pred=np.squeeze(pred, axis=3))
                    # summary_writer.add_summary(summary_str, step)

                if train_step % 200 == 1:
                    print('checkpoint')
                    saver.save(sess, os.path.join(FLAGS.model_dir, "model.ckpt"), train_step)

                """
                # FCN update
                """
                sess.run(fcn_optim, feed_dict=feed_dict_fcn)

                """
                # Discriminator update
                """
                sess.run(d_optim, feed_dict=feed_dict_fcn)

                train_step += 1

    elif FLAGS.mode == "test":
        from glob import glob
        import scipy.misc
        data, gt = [], []
        dataset_dir = '../../../dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/val'
        for folder in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, folder, "*_leftImg8bit.png")
            data.extend(glob(path))

        data_size = len(data)
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images = [scipy.misc.imresize(
                scipy.misc.imread(val_image_name).astype(np.float32), 0.125, interp='bilinear', mode=None)
                            for val_image_name in val_images_name]

            pred, heatmap = sess.run([pred_annotation, logits], feed_dict={image: val_images, keep_probability: 1.0})

            heatmap_s = heatmap[:,:,:,1]
            heatmap_s = myutils.sigmoid(heatmap_s)
            myutils.test_check(flags=FLAGS, name=val_images_name[0], test_image=val_images,
                               pred=np.squeeze(pred, axis=3), heatmap=heatmap_s)

    elif FLAGS.mode == "valid":
        from glob import glob
        import scipy.misc
        data, gt = [], []
        dataset_dir = '../../dataset/Relative/valid/relative_combine_image'
        heatmap_dir = '../../dataset/Relative/valid/relative_combine_heatmap'

        data = sorted(sorted(glob(os.path.join(dataset_dir, "*.png"))))
        heatmap_name = sorted(sorted(glob(os.path.join(heatmap_dir, "*.png"))))
        data_size = len(data)
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]
            val_heatmaps_name = heatmap_name[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images = [scipy.misc.imread(val_image_name).astype(np.float32) for val_image_name in val_images_name]
            val_heatmaps = np.array([scipy.misc.imread(val_heatmap_name).astype(np.float32) for val_heatmap_name in val_heatmaps_name])

            pred, heatmap = sess.run([pred_annotation, logits], feed_dict={image: val_images, keep_probability: 1.0})

            heatmap_s = heatmap[:,:,:,1]
            heatmap_s = myutils.sigmoid(heatmap_s)
            myutils.valid_check(flags=FLAGS, name=val_images_name[0], valid_image=val_images,
                               pred=np.squeeze(pred, axis=3), heatmap=heatmap_s, gt=val_heatmaps)


if __name__ == '__main__':
    main()
