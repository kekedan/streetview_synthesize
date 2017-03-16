import os
from glob import glob
import numpy as np
import scipy.misc
import h5py
labels = [
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (  0,  0,  0),
    (111, 74,  0),
    ( 81,  0, 81),
    (128, 64,128),
    (244, 35,232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32),
    (  0,  0,142)
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def label_visualize(img):
    """
    transfer label image to color image
    :param img_dir: dir of the image
    :return: visualization
    example code:
        img_dir = '/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label/aachen_000051_000019_gtFine_labelIds.png'
        label_visualize(img_dir)
    """
    visual = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(0, 34):
        index = np.nonzero(img == i)
        visual[index + (0,)] = labels[i][0]
        visual[index + (1,)] = labels[i][1]
        visual[index + (2,)] = labels[i][2]

    return visual


def merge(images, size, is_gray=False):
    h, w = images.shape[1], images.shape[2]
    if is_gray:
        img = np.zeros((int(h * size[0]), int(w * size[1])))
        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx // size[1])
            img[j*h:j*h+h, i*w:i*w+w] = image
    else:
        img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx // size[1])
            img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def directory_check(flags):
    if not os.path.exists(flags.result_dir):
        os.makedirs(flags.result_dir)
    if not os.path.exists(flags.logs_dir):
        os.makedirs(flags.logs_dir)
    if not os.path.exists(flags.valid_dir):
        os.makedirs(flags.valid_dir)
    if not os.path.exists(flags.test_dir):
        os.makedirs(flags.test_dir)
    if not os.path.exists(flags.model_dir):
        os.makedirs(flags.model_dir)


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


def test_check(flags, name, test_image, pred, heatmap):
    merge_shape = (flags.sample_shape, flags.sample_shape)
    name = name.split('/')[-1]
    name = name.split('.')[0]

    scipy.misc.imsave(os.path.join(flags.test_dir, '{}_target_image.png'.format(name)),
                      merge(np.array(test_image), merge_shape, is_gray=False))
    scipy.misc.imsave(os.path.join(flags.test_dir, '{}_target_pred.png'.format(name)),
                      merge(pred, merge_shape, is_gray=True))
    scipy.misc.imsave(os.path.join(flags.test_dir, '{}_target_heatmap.png'.format(name)),
                      merge(heatmap, merge_shape, is_gray=True))


def valid_check(flags, name, valid_image, pred, heatmap, gt):
    merge_shape = (flags.sample_shape, flags.sample_shape)
    name = name.split('/')[-1]
    name = name.split('.')[0]

    scipy.misc.imsave(os.path.join(flags.valid_dir, '{}_valid_image.png'.format(name)),
                      merge(np.array(valid_image), merge_shape, is_gray=False))
    scipy.misc.imsave(os.path.join(flags.valid_dir, '{}_valid_gt.png'.format(name)),
                      merge(gt, merge_shape, is_gray=True))
    #scipy.misc.imsave(os.path.join(flags.valid_dir, '{}_valid_pred.png'.format(name)),
    #                  merge(pred, merge_shape, is_gray=True))
    scipy.misc.imsave(os.path.join(flags.valid_dir, '{}_valid_heatmap.png'.format(name)),
                      merge(heatmap, merge_shape, is_gray=True))


def logs_check(flags, train_step, target_images, target_heatmaps, target_semantics, pred):
    merge_shape = (flags.sample_shape, flags.sample_shape)
    scipy.misc.imsave(os.path.join(flags.logs_dir, '{}_target.png'.format(train_step)),
                      merge(target_images, merge_shape))
    scipy.misc.imsave(os.path.join(flags.logs_dir, '{}_target_heatmap.png'.format(train_step)),
                      merge(target_heatmaps, merge_shape, is_gray=True))
    scipy.misc.imsave(os.path.join(flags.logs_dir, '{}_target_semantic.png'.format(train_step)),
                      label_visualize(merge(target_semantics, merge_shape, is_gray=True)))
    scipy.misc.imsave(os.path.join(flags.logs_dir, '{}_target_pred.png'.format(train_step)),
                      merge(pred, merge_shape, is_gray=True))


def logs_check2(flags, train_step, images_true, images_false, p_true, p_false):
    merge_shape = (flags.sample_shape, flags.sample_shape)
    images_true = np.concatenate((images_true, images_true, images_true), axis=3)
    images_false = np.concatenate((images_false, images_false, images_false), axis=3)
    threshold = 0.5
    true_ids = np.nonzero((np.array(p_true) > threshold))
    false_ids = np.nonzero((np.array(p_false) < threshold))
    images_true[true_ids, :, :, 2] = 1
    images_false[false_ids, :, :, 0] = 1
    scipy.misc.imsave(os.path.join(flags.result_dir, '{}_true.png'.format(train_step)),
                      merge(images_true, merge_shape))
    scipy.misc.imsave(os.path.join(flags.result_dir, '{}_false.png'.format(train_step)),
                      merge(images_false, merge_shape))


class InputFetcher(object):
    def __init__(self, flags, size):
        self.FLAGS = flags
        self.data_source, self.data_source_mask, self.data_target, self.train_size, self.batch_size = \
            [], [], [], 0, 0

        self.get_input_files_name()

    def get_input_files_name(self):
        self.data_source = sorted(glob(os.path.join(self.FLAGS.source_dir, "*.png")))
        self.data_source_mask = sorted(glob(os.path.join(self.FLAGS.source_mask_dir, "*.png")))

        self.data_target = sorted(glob(os.path.join(self.FLAGS.target_dir, "*.png")))

        self.train_size = len(self.data_target)
        self.batch_size = self.train_size / self.FLAGS.batch_size

    def shuffle(self):
        np.random.shuffle(self.data_target)

    def fetch(self, batch_idx):
        target_images_name = self.data_target[batch_idx * self.FLAGS.batch_size:(batch_idx + 1) * self.FLAGS.batch_size]

        target_images = np.array([scipy.misc.imread(target_image_name).astype(np.float32)
                                  for target_image_name in target_images_name])

        target_heatmaps = [
            scipy.misc.imread(os.path.join(self.FLAGS.target_heatmap_dir, target_image_name.split('/')[-1])).astype(
                np.float32)
            for target_image_name in target_images_name]

        # TODO : better way
        target_heatmaps = np.array(target_heatmaps) / 255
        yo = np.expand_dims(target_heatmaps, axis=3)
        target_semantics = np.concatenate((yo, yo, yo), axis=3)
        return target_images, target_heatmaps, target_semantics
