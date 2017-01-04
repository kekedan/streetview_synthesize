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
sample_itr = 5

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "/data/vllab1/dataset/CITYSCAPES/CITY", "path to dataset")
tf.flags.DEFINE_string("name_dir", "/data/vllab1/dataset/CITYSCAPES/CITY/human_wo.pkl", "path to name")
tf.flags.DEFINE_string("test_name_dir", "/data/vllab1/dataset/CITYSCAPES/CITY/human_w.pkl", "path to test name")
tf.flags.DEFINE_string("model_dir", "/data/vllab1/checkpoint/context inpainting/inpainting/", "path to model directory")
tf.flags.DEFINE_string("result_dir", "./test_wo/", "path to result directory")

tf.flags.DEFINE_integer("batch_size", "16", "batch size for training")
tf.flags.DEFINE_integer("sample_shape", "4", "for sample merge")
tf.flags.DEFINE_integer("image_size_h", "256", "height of the image")
tf.flags.DEFINE_integer("image_size_w", "512", "width of the image")

tf.flags.DEFINE_float("learning_rate", "3e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")


is_train = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, [])

images_tf = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, 3], name="images")
images_mask = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size_h, FLAGS.image_size_w, 3], name='images_mask')

labels_D = tf.concat(0, [tf.ones([FLAGS.batch_size]), tf.zeros([FLAGS.batch_size])])
labels_G = tf.ones([FLAGS.batch_size])

model = Model()

bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction \
    = model.build_reconstruction(images_tf, is_train)
adversarial_pos = model.build_adversarial(images_tf, is_train)
adversarial_neg = model.build_adversarial(reconstruction, is_train, reuse=True)
adversarial_all = tf.concat(0, [adversarial_pos, adversarial_neg])

# Applying bigger loss for overlapping region
#mask_recon = tf.pad(tf.ones([hiding_size_h - 2*overlap_size, hiding_size_w - 2*overlap_size]), [[overlap_size,overlap_size], [overlap_size,overlap_size]])
#mask_recon = tf.reshape(mask_recon, [hiding_size_h, hiding_size_w, 1])
#mask_recon = tf.concat(2, [mask_recon]*3)
#mask_overlap = 1 - mask_recon

mask_recon = images_mask
mask_overlap = 1 - mask_recon

# TODO check loss
loss_recon_ori = tf.square(images_tf - reconstruction)
loss_recon_center = tf.reduce_mean(tf.sqrt(1e-5 + tf.reduce_sum(loss_recon_ori * mask_recon, [1, 2, 3]))) * 2  # Loss for non-overlapping region
loss_recon_overlap = tf.reduce_mean(tf.sqrt(1e-5 + tf.reduce_sum(loss_recon_ori * mask_overlap, [1, 2, 3]))) # Loss for overlapping region
loss_recon = loss_recon_center + loss_recon_overlap

loss_adv_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_all, labels_D))
loss_adv_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, labels_G))

loss_G = loss_adv_G * lambda_adv + loss_recon * lambda_recon
loss_D = loss_adv_D * lambda_adv

var_G = filter(lambda x: x.name.startswith('GEN'), tf.trainable_variables())
var_D = filter(lambda x: x.name.startswith('DIS'), tf.trainable_variables())

W_G = filter(lambda x: x.name.endswith('W:0'), var_G)
W_D = filter(lambda x: x.name.endswith('W:0'), var_D)

loss_G += weight_decay_rate * tf.reduce_mean(tf.pack(map(lambda x: tf.nn.l2_loss(x), W_G)))
loss_D += weight_decay_rate * tf.reduce_mean(tf.pack(map(lambda x: tf.nn.l2_loss(x), W_D)))

sess = tf.InteractiveSession()

# TODO check gradients
optimizer_G = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_vars_G = optimizer_G.compute_gradients(loss_G, var_list=var_G )
#grads_vars_G = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G)
train_op_G = optimizer_G.apply_gradients(grads_vars_G)

optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate )
grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
#grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
train_op_D = optimizer_D.apply_gradients( grads_vars_D )

saver = tf.train.Saver(max_to_keep=5)

tf.initialize_all_variables().run()

ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...{}".format(ckpt.model_checkpoint_path))

if FLAGS.mode == 'train':
    iters = 0

    loss_D_val = 0.
    loss_G_val = 0.

    file_obj = open(FLAGS.name_dir, 'r')
    file_name = pickle.load(file_obj)
    train_size = len(file_name)

    sample_images_name = file_name[sample_itr * FLAGS.batch_size:(sample_itr + 1) * FLAGS.batch_size]
    sample_images = np.array([scipy.misc.imread(
        os.path.join(FLAGS.data_dir, 'fine_image', '{}_leftImg8bit.png'.format(name))).astype(np.float32)
                             for name in sample_images_name])
    sample_images = np.array(sample_images) / 255 * 2 - 1

    sample_masks = np.array([read_mask(
        os.path.join(FLAGS.data_dir, 'fine_mask', '{}_gtFine_labelIds.png'.format(name))).astype(np.float32)
                            for name in sample_images_name])

    for epoch in range(n_epochs):

        if epoch % 5 == 0:
            reconstruction_vals, recon_ori_vals, bn1_val, bn2_val, bn3_val, bn4_val, bn5_val, bn6_val, debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
                [reconstruction, reconstruction_ori, bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, loss_G,
                 loss_D],
                feed_dict={
                    images_tf: sample_images,
                    images_mask: sample_masks,
                    is_train: False
                })
            samples = (255. * (sample_images + 1) / 2.).astype(int)
            samples_hole = (1-sample_masks) * samples
            reconstruction_vals_ori = (255. * (reconstruction_vals + 1) / 2.).astype(int)
            reconstruction_vals = sample_masks * reconstruction_vals_ori + samples_hole
            if epoch == 0:
                scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_ori.png'.format(epoch))
                                  , merge(samples, (FLAGS.sample_shape, FLAGS.sample_shape)))
                scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_ori_mask.png'.format(epoch))
                                  , merge(samples_hole, (FLAGS.sample_shape, FLAGS.sample_shape)))

            scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_rec_ori.png'.format(epoch))
                              , merge(reconstruction_vals_ori, (FLAGS.sample_shape, FLAGS.sample_shape)))
            scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_rec.png'.format(epoch))
                              , merge(reconstruction_vals, (FLAGS.sample_shape, FLAGS.sample_shape)))

        np.random.shuffle(file_name)
        for batch_itr in xrange(train_size / FLAGS.batch_size):
            print('[{:d}/{:d}] [{:d}/{:d}]'.format(epoch, n_epochs, batch_itr, train_size / FLAGS.batch_size))
            batch_images_name = file_name[batch_itr * FLAGS.batch_size:(batch_itr + 1) * FLAGS.batch_size]

            train_images = np.array([scipy.misc.imread(
                os.path.join(FLAGS.data_dir, 'fine_image', '{}_leftImg8bit.png'.format(name))).astype(np.float32)
                                    for name in batch_images_name])
            train_images = np.array(train_images) / 255 * 2 - 1

            train_masks = np.array([read_mask(
                os.path.join(FLAGS.data_dir, 'fine_mask', '{}_gtFine_labelIds.png'.format(name))).astype(np.float32)
                                    for name in batch_images_name])

            # Generative Part is updated every iteration
            _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val = sess.run(
                    [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1],
                    feed_dict={
                        images_tf: train_images,
                        images_mask: train_masks,
                        learning_rate: FLAGS.learning_rate,
                        is_train: True
                        })

            # TODO 2:1?
            if iters % 2 == 0:
                _, loss_D_val, adv_pos_val, adv_neg_val = sess.run(
                        [train_op_D, loss_D, adversarial_pos, adversarial_neg],
                        feed_dict={
                            images_tf: train_images,
                            images_mask: train_masks,
                            learning_rate: FLAGS.learning_rate,
                            is_train: True
                                })
                # Printing activations every 10 iterations
                print "Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_G_val, "Dis Loss:", loss_D_val, "||||", adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max()

            if iters % 100 == 0:
                # TODO batch check
                reconstruction_vals, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
                        [reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, loss_G, loss_D],
                        feed_dict={
                            images_tf: train_images,
                            images_mask: train_masks,
                            is_train: False
                            })

                # Generate result every 1000 iterations
                if iters % 500 == 0:
                    ii = 0
                    '''
                    for rec_val, img,x,y in zip(reconstruction_vals, test_images, xs, ys):
                        rec_hid = (255. * (rec_val+1)/2.).astype(int)
                        rec_con = (255. * (img+1)/2.).astype(int)

                        rec_con[y:y+64, x:x+64] = rec_hid
                        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.'+str(int(iters/100))+'.jpg'), rec_con)
                        ii += 1
                        if ii > 50: break
                    if iters == 0:
                        sample
                        for train_image in train_images:
                            test_image = (255. * (train_image+1)/2.).astype(int)
                            test_image[32:32+64,32:32+64] = 0
                            cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), test_image)
                            ii += 1
                            if ii > 50: break
                    '''


                print "========================================================================"
                print bn1_val.max(), bn1_val.min()
                print bn2_val.max(), bn2_val.min()
                print bn3_val.max(), bn3_val.min()
                print bn4_val.max(), bn4_val.min()
                print bn5_val.max(), bn5_val.min()
                print bn6_val.max(), bn6_val.min()
                print debn4_val.max(), debn4_val.min()
                print debn3_val.max(), debn3_val.min()
                print debn2_val.max(), debn2_val.min()
                print debn1_val.max(), debn1_val.min()
                print recon_ori_vals.max(), recon_ori_vals.min()
                print reconstruction_vals.max(), reconstruction_vals.min()
                print loss_G_val, loss_D_val
                print "========================================================================="

                if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                    print "NaN detected!!"
                    #ipdb.set_trace()

            iters += 1

        # Every epoch
        FLAGS.learning_rate *= 0.99
        #saver.save(sess, model_path + 'model', global_step=epoch)
        if epoch % 10 == 0:
            print('save')
            saver.save(sess, FLAGS.model_dir + "model.ckpt", global_step=epoch)

elif FLAGS.mode == 'test':
    file_obj = open(FLAGS.test_name_dir, 'r')
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

        for index in range(0, FLAGS.batch_size):
            hole_idx = np.nonzero(sample_masks[index, :, :])
            sample_images[index, hole_idx[0], hole_idx[1], 0] = 0
            sample_images[index, hole_idx[0], hole_idx[1], 1] = 0
            sample_images[index, hole_idx[0], hole_idx[1], 2] = 0

        reconstruction_vals, recon_ori_vals, bn1_val, bn2_val, bn3_val, bn4_val, bn5_val, bn6_val, debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
            [reconstruction, reconstruction_ori, bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, loss_G,
             loss_D],
            feed_dict={
                images_tf: sample_images,
                images_mask: sample_masks,
                is_train: False
            })

        samples = (255. * (sample_images + 1) / 2.).astype(int)
        samples_hole = (1-sample_masks) * samples
        reconstruction_vals_ori = (255. * (reconstruction_vals + 1) / 2.).astype(int)
        reconstruction_vals = sample_masks * reconstruction_vals_ori + samples_hole

        scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_ori.png'.format(batch_itr))
                          , merge(samples, (FLAGS.sample_shape, FLAGS.sample_shape)))
        scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_rec_ori.png'.format(batch_itr))
                          , merge(reconstruction_vals_ori, (FLAGS.sample_shape, FLAGS.sample_shape)))
        scipy.misc.imsave(os.path.join(FLAGS.result_dir, '{:d}_rec.png'.format(batch_itr))
                          , merge(reconstruction_vals, (FLAGS.sample_shape, FLAGS.sample_shape)))