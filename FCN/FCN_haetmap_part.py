from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.misc
import os
from glob import glob

import TensorflowUtils as utils
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/data/vllab1/pose-hg-train/data/mpii/train/images", "path to dataset")
tf.flags.DEFINE_string("label_dir", "/data/vllab1/pose-hg-train/data/mpii/train/annot", "path to annotation")
tf.flags.DEFINE_string("model_dir", "../../../checkpoint/", "Path to vgg model mat")
tf.flags.DEFINE_string("logs_dir", "../../../checkpoint/FCN/heatmap_instance_pos/", "path to logs directory")

tf.flags.DEFINE_integer("batch_size", "9", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 16
SAMPLE_SHAPE = (3, 3)
IMAGE_SIZE_h = 256
IMAGE_SIZE_w = 256


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
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


def main(argv=None):
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_h, IMAGE_SIZE_w, 3], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_h, IMAGE_SIZE_w, 16], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)

    loss = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(labels=annotation, logits=logits)))

    trainable_var = tf.trainable_variables()

    # TODO new loss
    train_op = train(loss, trainable_var)

    # TODO next batch file name suffle
    data = sorted(glob(os.path.join(FLAGS.data_dir, "*.png")))
    train_size = len(data)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=2)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...{}".format(ckpt.model_checkpoint_path))
    else:
        print('No model found')

    if FLAGS.mode == "train":
        step = 0
        for epoch in xrange(MAX_ITERATION):
            np.random.shuffle(data)
            for batch_itr in xrange(train_size / FLAGS.batch_size):
                print('[{:d}/{:d}] [{:d}/{:d}]'.format(epoch, MAX_ITERATION, batch_itr, train_size / FLAGS.batch_size))
                train_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

                train_images, train_annotations = [], []
                for train_image_name in train_images_name[:FLAGS.batch_size/2]:
                    train_image = scipy.misc.imread(train_image_name).astype(np.float32)
                    train_annotation = scipy.misc.imread(os.path.join(
                        FLAGS.label_dir, train_image_name.split('/')[-1])).astype(np.float32) / 255.0

                    train_images.append(train_image)
                    train_annotations.append(train_annotation)

                # Flip
                for train_image_name in train_images_name[FLAGS.batch_size/2:]:
                    train_image = np.fliplr(scipy.misc.imread(train_image_name).astype(np.float32))
                    train_annotation = np.fliplr(scipy.misc.imread(os.path.join(
                        FLAGS.label_dir, train_image_name.split('/')[-1])).astype(np.float32) / 255.0)

                    train_images.append(train_image)
                    train_annotations.append(train_annotation)

                # train_images, train_annotations = traitn_dataset_reader.next_batch(FLAGS.batch_size)
                feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

                sess.run(train_op, feed_dict=feed_dict)

                if step % 5 == 0:
                    # train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    train_loss = sess.run(loss, feed_dict=feed_dict)
                    print("Step: %d, Train_loss:%g" % (step, train_loss))
                    # summary_writer.add_summary(summary_str, step)

                if step % 150 == 0:
                    # train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                    scipy.misc.imsave('logs/{:d}_image.png'.format(step), utils.merge(
                        np.array(train_images), SAMPLE_SHAPE).astype(np.uint8))
                    scipy.misc.imsave('logs/{:d}_gt.png'.format(step), utils.merge(
                        np.array(train_annotations) * 255.0, SAMPLE_SHAPE).astype(np.uint8))

                    pred_heatmap = sess.run(logits, feed_dict={image: train_images, annotation: train_annotations,
                                                                keep_probability: 1.0})

                    pred_heatmap = sigmoid(pred_heatmap)
                    scipy.misc.imsave('logs/{:d}_pred.png'.format(step), utils.merge(
                        pred_heatmap * 255.0, SAMPLE_SHAPE).astype(np.uint8))
                    # summary_writer.add_summary(summary_str, step)

                if step % 300 == 0:
                    # valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                    # valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                    #                                       keep_probability: 1.0})
                    # print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                    print('checkpoint')
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", step)

                step += 1

    elif FLAGS.mode == "visualize":
        data = sorted(glob(os.path.join(FLAGS.data_dir, "*.png")))
        train_size = len(data)

        for batch_itr in range(0, train_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, train_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            if FLAGS.batch_size == 1:
                val_name = val_images_name[0].split('/')[-1].split('.')[0]
            val_images = [scipy.misc.imread(val_image_name) for val_image_name in val_images_name]
            val_annotations = [scipy.misc.imread(
                os.path.join(FLAGS.label_dir, val_annotation_name.split('/')[-1])).astype(np.float32) / 255.0
                                 for val_annotation_name in val_images_name]


            pred, heatmap = sess.run([pred_annotation, logits], feed_dict={image: val_images, keep_probability: 1.0})
            heatmap_s = sigmoid(heatmap)
            scipy.misc.imsave('train_visual/{}_pred.png'.format(val_name), utils.merge(
                heatmap_s * 255.0, SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('train_visual/{}_image.png'.format(val_name), utils.merge(
                np.array(val_images), SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('train_visual/{}_gt.png'.format(val_name), utils.merge(
                np.array(val_annotations) * 255.0, SAMPLE_SHAPE).astype(np.uint8))
            #smin = heatmap_s.min()
            #heatmap_s -= smin
            #heatmap_s = heatmap_s * heatmap_s
            #valid_annotations = [np.squeeze(valid_annotations, axis=3)]
            #pred = np.squeeze(pred, axis=3)
            #break

    elif FLAGS.mode == "test":
        data, gt = [], []
        dataset_dir = '../../../dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/val'
        for folder in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, folder, "*_leftImg8bit.png")
            data.extend(glob(path))

        #np.random.shuffle(data)

        data_size = len(data)
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images = [scipy.misc.imresize(
                scipy.misc.imread(val_image_name).astype(np.uint8), 0.25, interp='bilinear', mode=None)
                            for val_image_name in val_images_name]

            pred, heatmap = sess.run([pred_annotation, logits], feed_dict={image: val_images, keep_probability: 1.0})
            heatmap_s = heatmap[:,:,:,1]
            smin = heatmap_s.min()
            heatmap_s -= smin
            heatmap_s = heatmap_s * heatmap_s
            #valid_annotations = [np.squeeze(valid_annotations, axis=3)]
            pred = np.squeeze(pred, axis=3)
            #heatmap = np.copy(logits[:,:,:,1])

            scipy.misc.imsave('test/{:d}_image.png'.format(batch_itr), utils.merge(
                np.array(val_images), SAMPLE_SHAPE))
            scipy.misc.imsave('test/{:d}_pred.png'.format(batch_itr), utils.heatmap_visualize(
                utils.merge(pred, SAMPLE_SHAPE, is_gray=True)))
            import math
            scipy.misc.imsave('test/{:d}_logit.png'.format(batch_itr), utils.heatmap_visualize(
                utils.merge(heatmap_s * heatmap_s, SAMPLE_SHAPE, is_gray=True)))
        #utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))

    elif FLAGS.mode == "valid_visual":

        data_size = train_size
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images, val_annotations = [], []
            for val_image_name in val_images_name:
                val_image = scipy.misc.imread(val_image_name).astype(np.float32)
                val_image = scipy.misc.imresize(val_image, 0.5)
                val_annotation = scipy.misc.imread(os.path.join(
                    FLAGS.label_dir, val_image_name.split('/')[-1])).astype(np.float32) / 255.0
                val_annotation = scipy.misc.imresize(val_annotation, 0.5)

                val_images.append(val_image)
                val_annotations.append(val_annotation)
            val_images = np.array(val_images)
            val_annotations = np.array(val_annotations)

            pred_heatmap = sess.run(logits, feed_dict={image: val_images, annotation: val_annotations, keep_probability: 1.0})
            pred_heatmap = sigmoid(pred_heatmap)

            out_name = val_image_name.split('/')[-1]
            scipy.misc.imsave('valid_visual/{}_image.png'.format(out_name), utils.merge(
                val_images, SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('valid_visual/{}_gt.png'.format(out_name), utils.merge(
                val_annotations, SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('valid_visual/{}_pred.png'.format(out_name), utils.merge(
                pred_heatmap * 255.0, SAMPLE_SHAPE).astype(np.uint8))
            visual = utils.merge(pred_heatmap * 200.0 + val_images * 0.8, SAMPLE_SHAPE)
            visual[np.nonzero(visual > 255)] = 255
            scipy.misc.imsave('valid_visual/{}_visual.png'.format(out_name), visual.astype(np.uint8))
        #utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))

    elif FLAGS.mode == "test_visual":

        data_size = train_size
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images = []
            for val_image_name in val_images_name:
                val_image = scipy.misc.imread(val_image_name).astype(np.float32)
                val_image = scipy.misc.imresize(val_image, 0.5)

                val_images.append(val_image)

            val_images = np.array(val_images)

            pred_heatmap = sess.run(logits, feed_dict={image: val_images, keep_probability: 1.0})
            pred_heatmap = sigmoid(pred_heatmap)

            out_name = val_image_name.split('/')[-1]
            scipy.misc.imsave('test_visual/{}_image.png'.format(out_name), utils.merge(
                val_images, SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('test_visual/{}_pred.png'.format(out_name), utils.merge(
                pred_heatmap * 255.0, SAMPLE_SHAPE).astype(np.uint8))
            visual = utils.merge(pred_heatmap * 10.0 + val_images * 0.8, SAMPLE_SHAPE)
            visual[np.nonzero(visual > 255)] = 255
            scipy.misc.imsave('test_visual/{}_visual.png'.format(out_name), visual.astype(np.uint8))
        #utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))

    elif FLAGS.mode == "test_visual_use_valid":

        data_size = train_size
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images = []
            for val_image_name in val_images_name:
                val_image = scipy.misc.imread(val_image_name).astype(np.float32)
                val_image_ori = np.copy(val_image)
                val_image = scipy.misc.imresize(val_image, 0.5)

                val_images.append(val_image)

            val_images = np.array(val_images)

            pred_heatmap = sess.run(logits, feed_dict={image: val_images, keep_probability: 1.0})
            pred_heatmap = sigmoid(pred_heatmap)

            out_name = val_image_name.split('/')[-1]
            scipy.misc.imsave('test_visual_use_valid/{}_image.png'.format(out_name), utils.merge(
                val_images, SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('test_visual_use_valid/{}_pred.png'.format(out_name), utils.merge(
                pred_heatmap * 255.0, SAMPLE_SHAPE).astype(np.uint8))
            visual = utils.merge(pred_heatmap * 200.0 * 255. + val_images , SAMPLE_SHAPE)
            visual[np.nonzero(visual > 255)] = 255
            scipy.misc.imsave('test_visual_use_valid/{}_visual.png'.format(out_name), visual.astype(np.uint8))
            visual = scipy.misc.imresize(pred_heatmap[0, :, :, :], 2.0) * 3.0 + val_image_ori
            visual[np.nonzero(visual > 255)] = 255
            scipy.misc.imsave('test_visual_use_valid_only/{}_visual.png'.format(out_name), visual.astype(np.uint8))

            #break
        #utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))

    elif FLAGS.mode == "test_out":

        data_size = train_size
        for batch_itr in range(0, data_size / FLAGS.batch_size):
            print('[{:d}/{:d}]'.format(batch_itr, data_size / FLAGS.batch_size))
            val_images_name = data[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            val_images = []
            for val_image_name in val_images_name:
                val_image = scipy.misc.imread(val_image_name).astype(np.float32)

                val_images.append(val_image)

            val_images = np.array(val_images)

            pred_heatmap = sess.run(logits, feed_dict={image: val_images, keep_probability: 1.0})
            pred_heatmap = sigmoid(pred_heatmap)

            out_name = val_image_name.split('/')[-1]
            scipy.misc.imsave('test_out/{}_image.png'.format(out_name), utils.merge(
                val_images, SAMPLE_SHAPE).astype(np.uint8))
            scipy.misc.imsave('test_out/{}_pred.png'.format(out_name), utils.merge(
                pred_heatmap * 255.0, SAMPLE_SHAPE).astype(np.uint8))

            pred_heatmap = pred_heatmap[0]
            val_images = val_images[0]
            visual = pred_heatmap * 400.0 + val_images
            visual[np.nonzero(visual > 255)] = 255
            scipy.misc.imsave('test_out/{}_visual.png'.format(out_name), visual.astype(np.uint8))
            break
        #utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))

if __name__ == "__main__":
    tf.app.run()
