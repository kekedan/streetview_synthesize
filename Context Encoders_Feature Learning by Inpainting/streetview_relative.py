import os
from glob import glob
import scipy
from model import *
from util import *

n_epochs = 10000
weight_decay_rate = 0.00001
momentum = 0.9
lambda_recon = 0.9
lambda_adv = 0.1

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "../../../dataset/CITYSCAPES/CITY/", "path to dataset")
tf.flags.DEFINE_string("name_dir", "../../../dataset/CITYSCAPES/CITY/extended_human_wo.pkl", "path to name")
tf.flags.DEFINE_string("human_dir", "../../../dataset/CITYSCAPES/CITY/human_w.pkl", "path to test name")
tf.flags.DEFINE_string("model_dir", "../../../checkpoint/context inpainting/inpainting_D/", "path to model directory")
tf.flags.DEFINE_string("result_dir", "./logs_inpainting_D/", "path to result directory")
tf.flags.DEFINE_string("test_dir", "./test_inpainting_D/", "path to result directory")

tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_integer("sample_shape", "4", "for sample merge")
tf.flags.DEFINE_integer("image_size_h", "256", "height of the image")
tf.flags.DEFINE_integer("image_size_w", "512", "width of the image")

tf.flags.DEFINE_float("learning_rate", "3e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")


is_train = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, [])

images_tf = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, 3], name="images")
images_hole = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, 3], name="images_hole")
images_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, 3], name='images_mask')

model = Model()
bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction \
    = model.build_reconstruction(images_hole, is_train)

# TODO VAE ration
mask_recon = images_mask
mask_overlap = tf.ones_like(mask_recon) - mask_recon
loss_recon_ori = tf.square(tf.subtract(images_tf, reconstruction))
#loss_recon = tf.reduce_mean(tf.reduce_sum(loss_recon_ori, [1, 2, 3]))
loss_recon_hole = tf.reduce_mean(tf.reduce_sum(tf.multiply(mask_recon, loss_recon_ori), [1, 2, 3]))
loss_recon_overlap = tf.reduce_mean(tf.reduce_sum(tf.multiply(mask_overlap, loss_recon_ori), [1, 2, 3]))
loss_recon = tf.add(loss_recon_hole, loss_recon_overlap * 10)

# TODO check adv loss
adversarial_pos = model.build_adversarial(images_tf, is_train)
adversarial_neg = model.build_adversarial(reconstruction, is_train, reuse=True)
#adversarial_all = tf.concat(0, [adversarial_pos, adversarial_neg])
#labels_D = tf.concat(0, [tf.ones([FLAGS.batch_size]), tf.zeros([FLAGS.batch_size])])
#labels_G = tf.ones([FLAGS.batch_size])
#loss_adv_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_all, labels_D))
#loss_adv_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, labels_G))
loss_adv_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_pos, tf.ones([FLAGS.batch_size])))
loss_adv_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, tf.zeros([FLAGS.batch_size])))
loss_adv_D = loss_adv_D_real + loss_adv_D_fake
loss_adv_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, tf.ones([FLAGS.batch_size])))

# TODO vae and adv ration
loss_G = loss_adv_G * lambda_adv + loss_recon * lambda_recon
loss_D = loss_adv_D * lambda_adv

var_G = filter(lambda x: x.name.startswith('GEN'), tf.trainable_variables())
var_D = filter(lambda x: x.name.startswith('DIS'), tf.trainable_variables())
W_G = filter(lambda x: x.name.endswith('W:0'), var_G)
W_D = filter(lambda x: x.name.endswith('W:0'), var_D)
loss_G += weight_decay_rate * tf.reduce_mean(tf.pack(map(lambda x: tf.nn.l2_loss(x), W_G)))
loss_D += weight_decay_rate * tf.reduce_mean(tf.pack(map(lambda x: tf.nn.l2_loss(x), W_D)))


# TODO check gradients
optimizer_G = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_vars_G = optimizer_G.compute_gradients(loss_G, var_list=var_G )
#grads_vars_G = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G)
train_op_G = optimizer_G.apply_gradients(grads_vars_G)
optimizer_D = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_vars_D = optimizer_D.compute_gradients(loss_D, var_list=var_D)
#grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
train_op_D = optimizer_D.apply_gradients(grads_vars_D)

saver = tf.train.Saver(max_to_keep=4)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...{}".format(ckpt.model_checkpoint_path))
else:
    print('No model found')

if FLAGS.mode == 'train':
    iters = 0

    # streetview without human
    file_obj = open(FLAGS.name_dir, 'r')
    file_name = pickle.load(file_obj)
    train_size = len(file_name)

    # random human mask
    file_obj = open(FLAGS.human_dir, 'r')
    mask_name = pickle.load(file_obj)
    mask_size = len(mask_name)
    mask_name = np.repeat(mask_name, (train_size/mask_size) + 1)

    for epoch in range(n_epochs):
        np.random.shuffle(file_name)
        np.random.shuffle(mask_name)

        for batch_itr in xrange(train_size / FLAGS.batch_size):
            print('[{:d}/{:d}] [{:d}/{:d}]'.format(epoch, n_epochs, batch_itr, train_size / FLAGS.batch_size))
            batch_images_name = file_name[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]
            batch_masks_name = mask_name[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            train_images = np.array([scipy.misc.imread(
                os.path.join(FLAGS.data_dir, 'coarse_image', '{}_leftImg8bit.png'.format(name))).astype(np.float32)
                                    for name in batch_images_name])
            train_images = np.array(train_images) / 255 * 2 - 1

            train_masks = np.array([read_mask(
                os.path.join(FLAGS.data_dir, 'fine_mask', '{}_gtFine_labelIds.png'.format(name))).astype(np.float32)
                                    for name in batch_masks_name])

            train_images_hole = (1 - train_masks) * train_images

            # Generative Part is updated every iteration
            _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val = sess.run(
                    [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1],
                    feed_dict={
                        images_tf: train_images,
                        images_mask: train_masks,
                        images_hole: train_images_hole,
                        learning_rate: FLAGS.learning_rate,
                        is_train: True
                        })

            # TODO check adv ration
            if iters % 1 == 0:
                _, loss_D_val, adv_pos_val, adv_neg_val, loss_adv_D_val = sess.run(
                        [train_op_D, loss_D, adversarial_pos, adversarial_neg, loss_adv_D],
                        feed_dict={
                            images_tf: train_images,
                            images_mask: train_masks,
                            images_hole: train_images_hole,
                            learning_rate: FLAGS.learning_rate,
                            is_train: True
                                })
                # Printing activations every 10 iterations
                print "Iter:", iters, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_G_val, \
                    "Dis ADV Loss:", loss_adv_D_val, "|||", "Gen Loss:", loss_G_val, "Dis Loss:", loss_D_val, "||||", \
                    adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max()

            if iters % 100 == 0:
                reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
                        [reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, loss_G, loss_D],
                        feed_dict={
                            images_tf: train_images,
                            images_mask: train_masks,
                            images_hole: train_images_hole,
                            is_train: False
                            })

                samples = (255. * (train_images + 1) / 2.).astype(int)
                samples_hole = (255. * (train_images_hole + 1) / 2.).astype(int)
                reconstruction_vals_ori = (255. * (reconstruction_vals + 1) / 2.).astype(int)
                reconstruction_vals = train_masks * reconstruction_vals_ori + (1 - train_masks) * samples

                scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_ori.png'.format(iters/10))
                                  , merge(samples, (FLAGS.sample_shape, FLAGS.sample_shape)))
                scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_ori_hole.png'.format(iters/10))
                                  , merge(samples_hole, (FLAGS.sample_shape, FLAGS.sample_shape)))
                scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_rec_ori.png'.format(iters/10))
                                  , merge(reconstruction_vals_ori, (FLAGS.sample_shape, FLAGS.sample_shape)))
                scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_rec.png'.format(iters/10))
                                  , merge(reconstruction_vals, (FLAGS.sample_shape, FLAGS.sample_shape)))

                if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                    print "NaN detected!!"
                    #ipdb.set_trace()

            iters += 1

        # Every epoch
        FLAGS.learning_rate *= 0.99
        print(FLAGS.learning_rate)
        if epoch % 1 == 0:
            print('save')
            saver.save(sess, FLAGS.model_dir + "model.ckpt", global_step=epoch)

elif FLAGS.mode == 'test':
    file_obj = open(FLAGS.human_dir, 'r')
    file_name = pickle.load(file_obj)
    train_size = len(file_name)

    np.random.shuffle(file_name)

    for batch_itr in range(0, train_size / FLAGS.batch_size):
        print('[{:d}/{:d}]'.format(batch_itr, train_size / FLAGS.batch_size))

        sample_images_name = file_name[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]
        sample_images = np.array([scipy.misc.imread(
            os.path.join(FLAGS.data_dir, 'fine_image', '{}_leftImg8bit.png'.format(name))).astype(np.float32)
                                  for name in sample_images_name])
        sample_images = np.array(sample_images) / 255 * 2 - 1

        sample_masks = np.array([read_mask(
            os.path.join(FLAGS.data_dir, 'fine_mask', '{}_gtFine_labelIds.png'.format(name))).astype(np.float32)
                                 for name in sample_images_name])

        sample_images_hole = (1 - sample_masks) * sample_images

        # TODO figure out what's going on hereQQ
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...{}".format(ckpt.model_checkpoint_path))
        reconstruction_vals = sess.run(
            reconstruction,
            feed_dict={
                images_hole: sample_images_hole,
                is_train: False
            })

        samples = (255. * (sample_images + 1) / 2.).astype(int)
        samples_hole = (255. * (sample_images_hole + 1) / 2.).astype(int)
        reconstruction_vals_ori = (255. * (reconstruction_vals + 1) / 2.).astype(int)
        reconstruction_vals = sample_masks * reconstruction_vals_ori + (1 - sample_masks) * samples

        scipy.misc.imsave(os.path.join(FLAGS.test_dir, '{:d}_ori.png'.format(batch_itr))
                          , merge(samples, (FLAGS.sample_shape, FLAGS.sample_shape)))
        scipy.misc.imsave(os.path.join(FLAGS.test_dir, '{:d}_rec_ori.png'.format(batch_itr))
                          , merge(reconstruction_vals_ori, (FLAGS.sample_shape, FLAGS.sample_shape)))
        scipy.misc.imsave(os.path.join(FLAGS.test_dir, '{:d}_rec.png'.format(batch_itr))
                          , merge(reconstruction_vals, (FLAGS.sample_shape, FLAGS.sample_shape)))
        scipy.misc.imsave(os.path.join('./out', '{}_leftImg8bit.png'.format(name))
                          , merge(reconstruction_vals, (FLAGS.sample_shape, FLAGS.sample_shape)))

elif FLAGS.mode == 'out':
    file_obj = open(FLAGS.human_dir, 'r')
    file_name = pickle.load(file_obj)
    data_size = len(file_name)

    for idx, name in enumerate(file_name):
        print('[{:d}/{:d}]'.format(idx, data_size))

        sample_images = np.array(scipy.misc.imread(
            os.path.join(FLAGS.data_dir, 'fine_image', '{}_leftImg8bit.png'.format(name))).astype(np.float32))
        sample_images = np.array(sample_images) / 255 * 2 - 1

        sample_masks = np.array(read_mask(
            os.path.join(FLAGS.data_dir, 'fine_mask', '{}_gtFine_labelIds.png'.format(name))).astype(np.float32))

        sample_images_hole = (1 - sample_masks) * sample_images

        reconstruction_vals = sess.run(
            reconstruction,
            feed_dict={
                images_hole: [sample_images_hole],
                is_train: False
            })

        samples = (255. * (sample_images + 1) / 2.).astype(int)
        samples_hole = (255. * (sample_images_hole + 1) / 2.).astype(int)
        reconstruction_vals_ori = (255. * (np.copy(reconstruction_vals) + 1) / 2.).astype(int)
        #reconstruction = sample_masks * reconstruction_vals_ori + (1 - sample_masks) * samples

        scipy.misc.imsave(os.path.join('./out', '{}_leftImg8bit.png'.format(name)), reconstruction_vals_ori[0, :, :, :])

