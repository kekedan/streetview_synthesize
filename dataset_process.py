from glob import glob
import json
import scipy.ndimage
from PIL import Image, ImageDraw
import pickle
import os
import itertools
from scipy.ndimage.filters import gaussian_filter
import skimage.measure
import poissonblending
import h5py

from utils import *
import h5py

CITYSCAPES_dir = '/mnt/data/andy/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train'
CITYSCAPES_label_dir = '/mnt/data/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train'
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


def store_single(filename, height , width, col_num, sel):
    """
    Given an merge-image [rol_num, col_num]
    Select one of it and store it
    example code:
        filename = '5800.png'
        height , width, col_num = 256, 512, 4
        sel = 5
        store_single(filename, height , width, col_num, sel)
    """
    img = scipy.misc.imread(filename).astype(np.float)
    name, extension = filename.split('.')
    col, row = sel % col_num, sel / col_num
    sel_image = img[row*height:(row+1)*height, col*width:(col+1)*width, :]
    scipy.misc.imsave('single.' + extension, sel_image)


def crop_images(dataset_dir):
    """
    Read all images under the different folders
    Crop, resize and store them
    example code:
        crop_images(CITYSCAPES_dir)
    """
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*.png")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.uint8)
        img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY_test/fine_image/' + filePath.split('/')[-1], img)


def crop_images_label(dataset_dir, is_mask=True):
    """
    Read all labels under the different folders
    Crop, resize and store them
    example code:
        ddir = CITYSCAPES_label_dir
        crop_images_label(ddir, is_mask=False)
    """
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*_labelIds.png")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.uint8)
        img = scipy.misc.imresize(img, 0.25, interp='nearest', mode=None)
        if is_mask:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool_)

            mask[np.nonzero(img == 24)] = True
            img = mask

        scipy.misc.imsave('/mnt/data/andy/dataset/CITYSCAPES/label/' + filePath.split('/')[-1], img)
        #break


def crop_images_label_big(dataset_dir, is_mask=True):
    """
    Read all labels under the different folders
    Crop, resize and store them
    The mask range should be larger
    example code:
        ddir = '/home/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train'
        crop_images_label_big(ddir, is_mask=True)
    """
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*_labelIds.png")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.float32)
        if is_mask:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

            mask[np.nonzero(img == 24)] = 255
            img = mask

        img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        img[np.nonzero(img > 0)] = 255
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY_valid/fine_mask/' + filePath.split('/')[-1], img)


def crop_images_color(dataset_dir, is_mask=True):
    """
    Read all labels under the different folders
    Crop, resize and store them
    example code:
        crop_images_color(CITYSCAPES_dir, is_mask=True)
    """
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "*_labelIds.png")
        data.extend(glob(path))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.uint8)
        img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        if is_mask:
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

            idx_person = np.where(np.all(img == [220, 20, 60, 255], axis=-1))
            #idx_rider = np.where(np.all(img == [255, 0, 0, 255], axis=-1))
            #idx_void = np.where(np.all(img == [0, 0, 0, 255], axis=-1))

            #indices = np.concatenate((idx_person, idx_rider, idx_void), axis=1)
            indices = idx_person
            # mask[indices[0], indices[1], :] = (0, 0, 0, 255)
            mask[indices[0], indices[1]] = 0
            mask = np.reshape(mask, (256, 512))

        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
        #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image/' + filePath.split('/')[-1],
                          img[0:192, :])
        #break


def crop_images_same_dir(data_set_dir):
    """
    Read all images under the same folder
    Crop, resize and store them
    example code:
    crop_images_same_dir('/mnt/data/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom')
    """
    data = glob(os.path.join(data_set_dir, "*.png"))

    # data_length = len(data)
    # high < 256 because want to cut the bottom
    # offs_h = np.random.randint(low=0, high=200, size=data_length)
    # offs_h_end = offs_h + 256
    # offs_w = np.random.randint(low=0, high=512, size=data_length)
    # offs_w_end = offs_w + 512
    # print offs_h, offs_h_end

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))

        img = scipy.misc.imread(filePath).astype(np.float)
        #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_random/' + filePath.split('/')[-1],
        #                  img[offs_h[index]:offs_h_end[index], offs_w[index]:offs_w_end[index] :])
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/CITYSCAPES_crop_bottom_192/' + filePath.split('/')[-1],
                          img[0:192, :, :])
        #break


def label_visualize(img_dir):
    """
    transfer label image to color image
    :param img_dir: dir of the image
    :return: visualization
    example code:
        img_dir = '/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label/aachen_000051_000019_gtFine_labelIds.png'
        label_visualize(img_dir)
    """
    img = scipy.misc.imread(img_dir).astype(np.uint8)
    yo = np.nonzero(img == 1)
    visual = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(0, 34):
        index = np.nonzero(img == i)
        visual[index + (0,)] = labels[i][0]
        visual[index + (1,)] = labels[i][1]
        visual[index + (2,)] = labels[i][2]

    scipy.misc.imsave('./' + img_dir.split('/')[-1], visual)


def semantic_vidualize(img):
    yo = np.nonzero(img == 1)
    visual = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(0, 34):
        index = np.nonzero(img == i)
        visual[index + (0,)] = labels[i][0]
        visual[index + (1,)] = labels[i][1]
        visual[index + (2,)] = labels[i][2]

    return visual


def create_mask_img():
    data = sorted(glob(os.path.join('../../dataset/CITYSCAPES/CITY/human_image', "*.png")))
    label = sorted(glob(os.path.join('../../dataset/CITYSCAPES/CITY/human_mask', "*.png")))

    dataset_dir = '../../dataset/CITYSCAPES/CITY_valid'
    file_name = os.path.join(dataset_dir, 'human_w.pkl')
    file_obj = open(file_name, 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)
    for index in range(0, length):
        print ('%d/%d' % (index, len(human_file_name)))
        name = human_file_name[index]
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)

        #fileName = filePath.split('/')[-1].split('.')[0]
        img = scipy.misc.imread(os.path.join('../../dataset/CITYSCAPES/CITY_valid/human_image', image_name)).astype(np.uint8)
        label2 = scipy.misc.imread(os.path.join('../../dataset/CITYSCAPES/CITY_valid/fine_mask', mask_name)).astype(np.uint8)

        indices = np.nonzero(label2 == 255)
        img[indices + (0,)] = 0
        img[indices + (1,)] = 255
        img[indices + (2,)] = 0
        scipy.misc.imsave(os.path.join('../../dataset/CITYSCAPES/CITY_valid/human_mask_inpainting', image_name), img.astype(np.uint8))
        #scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image_mask/' + label2[i].split('/')[-1], img)


def create_merged_image(img_dir):
    """
    example code:
        ddir = '/home/andy/dataset/CITYSCAPES/for_wonderful_chou/sel_mask_demo/exemplar_based'
        create_merged_image(ddir)
    :param img_dir:
    :return:
    """
    data = sorted(glob(os.path.join(img_dir, "*.png")))
    imgs = np.array([scipy.misc.imread(img)[0:192, :, :] for img in data]).astype(np.float32)
    merged_img = merge(imgs, (2, 2))
    scipy.misc.imsave(os.path.join(img_dir, 'merged.png'), merged_img)


def create_instance():
    alpha = 0.01
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)

    for index in range(0, length):
        name = human_file_name[index]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]

        # astype(np.float32) somehow do the normalize
        # should use(np.uint8) if the inpainting method is context
        image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name)).astype(np.float32)
        inpainting = scipy.misc.imread(os.path.join(dataset_dir, 'inpainting_high', image_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))

        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        for obj in objects:
            if obj['label'] == 'person':
                #print(object['polygon'])
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                mask[np.nonzero(mask > 0)] = 255
                mask_instance.append(mask)

                human_pixel = np.nonzero(mask == 255)
                human_pixel_set.append(len(human_pixel[0]))
                # human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                # if human_ratio > alpha:
                #     mask_instance_sel.append(instance_num)
                instance_num += 1

        human_pixel_index = np.argsort(human_pixel_set)
        mask_instance_sorted = []
        for sorted_index in human_pixel_index:
            mask_instance_sorted.append(mask_instance[sorted_index])
        print(instance_num)
        if instance_num > 20:
            # Select the up to 10 biggest instance
            mask_instance_sorted = mask_instance_sorted[-21:-1]
            instance_num = 20

        instance_array = np.arange(instance_num)
        possible_c = np.arange(instance_num)
        np.random.shuffle(possible_c)
        if len(possible_c) > 10:
            # Cn0 + Cn1 + Cn2 + ... + Cnn-1
            # Random select up to 10 possible combinations
            possible_c = possible_c[0:10]
        for c_idx in possible_c:
            instance_combine = list(itertools.combinations(instance_array, c_idx))
            np.random.shuffle(instance_combine)
            if len(instance_combine) > 4:
                # Random select up to 4 combinations under one possible combinations. ex:
                # Cn3: (1, 2, 8), (2, 4, 6), (5, 18, 19), (3, 4, 5)
                instance_combine = instance_combine[0:4]
            for i_idx, combine_list in enumerate(instance_combine):
                img = np.copy(image)
                mask = np.zeros((256, 512), dtype=np.int8)
                heatmap = np.zeros((256, 512), dtype=np.int8)
                for m_idx, m in enumerate(mask_instance_sorted):
                    if m_idx in combine_list:
                        mask[np.nonzero(m == 255)] = 1
                    else:
                        heatmap[np.nonzero(m == 255)] = 1

                #mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=5).astype(mask.dtype)
                # TODO mask too big and context inpainting color weired
                # Fixed colore weried, np.float32 will aotuomatically normalized inf 0~1
                mask = np.dstack((mask, mask, mask))
                img_instace = img * mask + inpainting * (1 - mask)

                scipy.misc.imsave('../../dataset/CITYSCAPES/CITY/relative_high_image/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), img_instace)
                scipy.misc.imsave('../../dataset/CITYSCAPES/CITY/relative_high_heatmap/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), heatmap)


def create_instance_new_standard():
    alpha = 0.01
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)

    instance_multi_view_pose = {}
    for index in range(0, length):
        name = human_file_name[index]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]

        # astype(np.float32) somehow do the normalize
        # should use if the inpainting method is context
        image = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train', city_name, image_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train', city_name, image_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))

        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        instance_single_view_pose = []
        for obj in objects:
            if obj['label'] == 'person':
                #print(object['polygon'])
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                human_pixel = np.nonzero(mask > 0)
                miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(
                    human_pixel[1])
                ori_heigh, ori_width = maxy - miny, maxx - minx

                mask[human_pixel] = 1
                mask_image = image * np.dstack((mask, mask, mask))
                #mask_instance.append(mask)

                scale =  float(256)/max(ori_heigh, ori_width)
                resize_image = scipy.misc.imresize(mask_image[miny:maxy, minx:maxx, :], scale)
                backgorund = np.zeros((256, 256, 3), dtype=np.float32)
                heigh, width = np.shape(resize_image)[0], np.shape(resize_image)[1]
                margin_V = (256 - heigh) / 2
                margin_H = (256 - width) / 2
                backgorund[margin_V:margin_V+heigh, margin_H:margin_H+width, :] = resize_image

                #human_pixel = np.nonzero(mask == 255)
                #human_pixel_set.append(len(human_pixel[0]))
                # human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                # if human_ratio > alpha:
                #     mask_instance_sel.append(instance_num)
                scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/pose/{}_{:d}.png'.format(name, instance_num), backgorund)
                instance_pos = {'num': instance_num, 'minx': minx, 'miny': miny, 'heigh': heigh, 'width':width,
                                'scale': scale, 'margin_V': margin_V, 'margin_H': margin_H}
                instance_single_view_pose.append(instance_pos)
                instance_num += 1

        instance_multi_view_pose[name] = instance_single_view_pose

    file_obj = open('instance_multi_view_pose.pkl', 'wb')
    pickle.dump(instance_multi_view_pose, file_obj)
    file_obj.close()


def place_back_instance():

    file_obj = open(os.path.join('', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)

    for single_view_name in instance_multi_view_pose:
        single_view = instance_multi_view_pose[single_view_name]
        single_view_image = np.zeros((1024, 2048, 3), dtype=np.uint8)
        for instance_idx, instance in enumerate(single_view):
            instance_num = instance['num']
            scale = instance['scale']
            minx, miny = instance['minx'], instance['miny']
            heigh, width = instance['heigh'], instance['width']
            margin_V, margin_H = instance['margin_V'], instance['margin_H']

            instance_image = scipy.misc.imread(os.path.join('/data/vllab1/prediction',
                                                            '{}_{:d}.png.jpg'.format(single_view_name, instance_num)))

            instance_image_ori = instance_image[margin_V:margin_V+heigh, margin_H:margin_H+width]
            instance_image_resize = scipy.misc.imresize(instance_image_ori, 1.0 / scale)
            heigh, width = np.shape(instance_image_resize)[0], np.shape(instance_image_resize)[1]

            single_view_image[miny:miny+heigh, minx:minx+width, :] = instance_image_resize

        scipy.misc.imsave('{}.png'.format(single_view_name), single_view_image)

        break


def place_back_instance_h5():

    data = sorted(glob(os.path.join('/data/vllab1/prediction', "*.jpg")))
    data = []

    file = open('imgname.txt', 'r')
    for line in file:
        data.append(line)
        #print line,

    file_obj = open(os.path.join('', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)
    cur = 0

    filename = 'final_preds.h5'
    f = h5py.File(filename, 'r')
    preds = f['heatmaps']

    for single_view_name_idx, single_view_name in enumerate(sorted(instance_multi_view_pose)):
        print('{:d}/{:d} [{:d}]'.format(single_view_name_idx, length, cur))
        single_view = instance_multi_view_pose[single_view_name]
        image_name = '{}_leftImg8bit.png'.format(single_view_name)
        city_name = single_view_name.split('_', 1)[0]
        single_view_image = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train', city_name, image_name)).astype(np.float32)
        single_view_image_v = np.copy(single_view_image)
        single_view_heatmp = np.zeros((1024, 2048), dtype=np.float32)
        '''
        for instance_idx, instance in enumerate(single_view):
            instance_num, scale = instance['num'], instance['scale']
            minx, miny = instance['minx'], instance['miny']
            heigh, width = instance['heigh'], instance['width']
            margin_V, margin_H = instance['margin_V'], instance['margin_H']
            instance_image = scipy.misc.imread(os.path.join('/data/vllab1/prediction',
                                                            '{}_{:d}.png.jpg'.format(single_view_name, instance_num)))

            instance_image_ori = instance_image[margin_V:margin_V+heigh, margin_H:margin_H+width]
            instance_image_resize = scipy.misc.imresize(instance_image_ori, 1.0 / scale)
            heigh_resize, width_resize = np.shape(instance_image_resize)[0], np.shape(instance_image_resize)[1]
            single_view_image_v[miny:miny+heigh_resize, minx:minx+width_resize, :] = instance_image_resize

            head = np.round(((preds[cur, 9, 0:2] - [margin_H, margin_V]) / scale) + [minx, miny])

            single_view_heatmp[int(head[1]), int(head[0])] = 1
            cur += 1

        if(len(single_view) > 9):
            continue
        '''

        cur_size = len(single_view)
        for cur_idx in range(0, cur_size):
            instance_idx = cur + cur_idx
            instance_name = data[instance_idx]
            instance_name = instance_name.split('.')[0]
            instance_name_idx = int(instance_name.split('_')[-1])
            #instance_name_idx = cur_idx
            for instance in single_view:
                if instance['num'] == instance_name_idx:
                    break

            instance_num, scale = instance['num'], instance['scale']
            print('{} {}'.format(instance_idx, instance_num))
            minx, miny = instance['minx'], instance['miny']
            heigh, width = instance['heigh'], instance['width']
            margin_V, margin_H = instance['margin_V'], instance['margin_H']
            instance_image = scipy.misc.imread(os.path.join('/data/vllab1/prediction',
                                                            '{}_{:d}.png.jpg'.format(single_view_name, instance_num)))

            instance_image_ori = instance_image[margin_V:margin_V+heigh, margin_H:margin_H+width]
            instance_image_resize = scipy.misc.imresize(instance_image_ori, 1.0 / scale)
            heigh_resize, width_resize = np.shape(instance_image_resize)[0], np.shape(instance_image_resize)[1]
            single_view_image_v[miny:miny+heigh_resize, minx:minx+width_resize, :] = instance_image_resize

            head = np.round(((preds[instance_idx, 9, 0:2] - [margin_H, margin_V]) / scale) + [minx, miny])

            single_view_heatmp[int(head[1]), int(head[0])] = 1

        cur += cur_size
        #if(cur_size > 9):
        #    continue

        single_view_heatmp = scipy.ndimage.morphology.binary_dilation(single_view_heatmp, iterations=5).astype(
            single_view_heatmp.dtype)
        single_view_heatmp = np.dstack((single_view_heatmp, single_view_heatmp, single_view_heatmp))
        single_view_image[np.nonzero(single_view_heatmp == 1)] = 200

        scipy.misc.imsave('{}.png'.format(single_view_name), single_view_image_v)
        scipy.misc.imsave('{}_heatmap.png'.format(single_view_name), single_view_image)
        break


def create_mask_img_instance():
    data = sorted(glob(os.path.join('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image', "*.png")))
    label = sorted(glob(os.path.join('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label2_big', "*.png")))

    data = ['/mnt/data/andy/dataset/CITYSCAPES/sel_mask_demo/ori/strasbourg_000000_015602_leftImg8bit.png']
    label = sorted(glob(os.path.join('/mnt/data/andy/dataset/CITYSCAPES/instance', "*.png")))

    length = len(data)
    for i in range(0, length):
        print ('%d/%d' % (i, length))
        #fileName = filePath.split('/')[-1].split('.')[0]
        image = scipy.misc.imread(data[i]).astype(np.float)
        numInstance = 0
        for instance in label:
            label2 = scipy.misc.imread(instance).astype(np.int)

            img = np.copy(image)
            indices = np.nonzero(label2 == 255)
            img[indices + (0,)] = 0
            img[indices + (1,)] = 255
            img[indices + (2,)] = 0
            scipy.misc.imsave('/mnt/data/andy/dataset/CITYSCAPES/image_mask_instance/mask_{}_'.format(numInstance)
                              + data[i].split('/')[-1], img)
            numInstance += 1


def select_human_img():
    alpha = 0.05 # [0.001]
    beta = 0.00
    data_set_dir = '../../dataset/CITYSCAPES/CITY/fine_mask'
    data_set_image_dir = '../../dataset/CITYSCAPES/CITY/fine_image'
    data = sorted(glob(os.path.join(data_set_dir, "*.png")))
    data_image = sorted(glob(os.path.join(data_set_image_dir, "*.png")))
    w_human, h_human, wo_human = [], [], []
    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(filePath)
        #img_data = scipy.misc.imread(data_image[index])
        human_pixel = np.nonzero(img == 255)
        human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
        full_name = filePath.split('/')[-1].split('_')
        name = '{}_{}_{}'.format(full_name[0], full_name[1], full_name[2])
        if human_ratio > alpha:
            w_human.append(name)
            #scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/human_mask_new/{}'.format(filePath.split('/')[-1]), img)
            #scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/human_image_new/{}'.format(data_image[index].split('/')[-1]), img_data)
        elif human_ratio == beta:
            wo_human.append(name)
        else:
            h_human.append(name)

    file_obj = open('human_w.pkl', 'wb')
    pickle.dump(w_human, file_obj)
    file_obj.close()
    file_obj = open('human_h.pkl', 'wb')
    pickle.dump(h_human, file_obj)
    file_obj.close()
    file_obj = open('human_wo.pkl', 'wb')
    pickle.dump(wo_human, file_obj)
    file_obj.close()

    print len(w_human), len(h_human), len(wo_human)


def select_human_img_2():
    alpha = 0.05
    human_w = []

    data_set_dir = '../../dataset/CITYSCAPES/CITY_valid/fine_mask'
    data = sorted(glob(os.path.join(data_set_dir, "*.png")))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(filePath)
        human_pixel = np.nonzero(img == 255)
        human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
        full_name = filePath.split('/')[-1].split('_')
        name = '{}_{}_{}'.format(full_name[0], full_name[1], full_name[2])
        if human_ratio > alpha:
            human_w.append(name)
            image = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY_valid/fine_image', '{}_leftImg8bit.png'.format(name)))
            scipy.misc.imsave(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY_valid/human_image', '{}_leftImg8bit.png'.format(name)), image)

    '''
    data_set_dir = '../../dataset/CITYSCAPES/CITY/coarse_mask'
    data = sorted(glob(os.path.join(data_set_dir, "*.png")))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(filePath)
        human_pixel = np.nonzero(img == 255)
        human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
        full_name = filePath.split('/')[-1].split('_')
        name = '{}_{}_{}'.format(full_name[0], full_name[1], full_name[2])
        if human_ratio > alpha:
            human_no_extra.append(name)
    '''

    file_obj = open('human_w.pkl', 'wb')
    pickle.dump(human_w, file_obj)
    file_obj.close()

    print len(human_w)


def select_no_human_img():
    alpha = 0.
    human_no_extra = []

    data_set_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/fine_mask'
    data = sorted(glob(os.path.join(data_set_dir, "*.png")))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(filePath)
        human_pixel = np.nonzero(img == 255)
        human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
        full_name = filePath.split('/')[-1].split('_')
        name = '{}_{}_{}'.format(full_name[0], full_name[1], full_name[2])
        if human_ratio == alpha:
            human_no_extra.append(name)

    data_set_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/coarse_mask'
    data = sorted(glob(os.path.join(data_set_dir, "*.png")))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(filePath)
        human_pixel = np.nonzero(img == 255)
        human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
        full_name = filePath.split('/')[-1].split('_')
        name = '{}_{}_{}'.format(full_name[0], full_name[1], full_name[2])
        if human_ratio == alpha:
            human_no_extra.append(name)

    file_obj = open('human_no_extra.pkl', 'wb')
    pickle.dump(human_no_extra, file_obj)
    file_obj.close()

    print len(human_no_extra)


def load_image_with_name():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_name = os.path.join(dataset_dir, 'extended_human.pkl')
    file_obj = open(file_name, 'r')
    human_file_name = pickle.load(file_obj)
    for index in range(0, len(human_file_name)):
        print ('%d/%d' % (index, len(human_file_name)))
        name = human_file_name[index]
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        mask_name2 = '{}_gtCoarse_labelIds.png'.format(name)

        image = scipy.misc.imread(os.path.join(dataset_dir, 'coarse_image', image_name))
        scipy.misc.imsave(os.path.join('../../dataset/CITYSCAPES/CITY/human_image', image_name), image)

        try:
            mask = scipy.misc.imread(os.path.join(dataset_dir, 'coarse_mask', mask_name))
            scipy.misc.imsave(os.path.join('../../dataset/CITYSCAPES/CITY/human_mask', mask_name), mask)
        except IOError:
            mask = scipy.misc.imread(os.path.join(dataset_dir, 'coarse_mask', mask_name2))
            scipy.misc.imsave(os.path.join('../../dataset/CITYSCAPES/CITY/human_mask', mask_name2), mask)


def load_batch_with_name():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    fileName = os.path.join(dataset_dir, 'extended_human.pkl')
    file_obj = open(fileName, 'r')
    human_file_name = pickle.load(file_obj)
    np.random.shuffle(human_file_name)
    mask = np.zeros((4, 256, 512))
    index, num = 0, 0
    while num < 4:
        name = human_file_name[index]
        print name
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        mask_name2 = '{}_gtCoarse_labelIds.png'.format(name)

        try:
            print os.path.join(dataset_dir, 'coarse_mask', mask_name)
            mask[num, :, :] = scipy.misc.imread(os.path.join(dataset_dir, 'coarse_mask', mask_name))
        except IOError:
            mask[num, :, :] = scipy.misc.imread(os.path.join(dataset_dir, 'coarse_mask', mask_name2))
            num += 1
            print num

        index += 1

    scipy.misc.imsave('extended_human.png', merge(mask, (2,2), is_gray=True))


def instance_visual():

    dataset_dir = '../../dataset/CITYSCAPES/CITY/relative_context_image'
    inpainting_name = sorted(glob(os.path.join(dataset_dir, "*.png")))
    dataset_dir = '../../dataset/CITYSCAPES/CITY/relative_high_image'
    inpainting_high_name = sorted(glob(os.path.join(dataset_dir, "*.png")))

    dataset_len = min(len(inpainting_name), len(inpainting_high_name))
    print(dataset_len)

    for i in range(0, dataset_len):
        print('{:2d}/{:2d}'.format(i, dataset_len))

        name = inpainting_name[i].split('/')[-1]
        heatmap_name = os.path.join('../../dataset/CITYSCAPES/CITY/relative_context_heatmap', name)
        inpainting_exem_name = os.path.join('../../dataset/CITYSCAPES/CITY/relative_exem_context', name)
        ori_name_split = name.split('_')
        ori_name = ori_name_split[0] + '_' + ori_name_split[1] + '_' +\
                   ori_name_split[2] + '_' + ori_name_split[3] +'.png'
        image_name = os.path.join('../../dataset/CITYSCAPES/CITY/human_image', ori_name)

        inpainting = scipy.misc.imread(inpainting_name[i]).astype(np.float32)
        image = scipy.misc.imread(image_name).astype(np.float32)
        #inpainting_exem = scipy.misc.imread(inpainting_exem_name).astype(np.float32)
        heatmap = scipy.misc.imread(heatmap_name).astype(np.float32)
        yo = np.nonzero(heatmap == 255)
        inpainting_v = np.copy(inpainting)
        inpainting_v[yo + (0,)] += 120
        inpainting_v[yo + (1,)] += 120
        inpainting_v[yo + (2,)] += 120
        inpainting[np.nonzero(inpainting>255)] = 255

        heatmap_v = np.copy(inpainting)
        heatmap_v[yo + (0,)] = 255
        heatmap_v[yo + (1,)] = 255
        heatmap_v[yo + (2,)] = 255
        heatmap = np.dstack((heatmap, heatmap, heatmap))

        name = inpainting_name[i].split('/')[-1]

        v1 = np.hstack((image, inpainting))
        v2 = np.hstack((heatmap_v, inpainting_v))

        v = np.vstack((v1, v2))
        scipy.misc.imsave('visual/{}'.format(name), v.astype(np.uint8))


def instance_visual_new():
    dataset_dir = '../../dataset/CITYSCAPES/CITY/relative_context_image'
    inpainting_name = sorted(glob(os.path.join(dataset_dir, "*.png")))
    dataset_dir = '../../dataset/CITYSCAPES/CITY/relative_high_image'
    inpainting_high_name = sorted(glob(os.path.join(dataset_dir, "*.png")))

    dataset_len = min(len(inpainting_name), len(inpainting_high_name))
    print(dataset_len)

    for i in range(0, dataset_len):
        print('{:2d}/{:2d}'.format(i, dataset_len))

        name = inpainting_name[i].split('/')[-1]
        heatmap_name = os.path.join('../../dataset/CITYSCAPES/CITY/relative_context_heatmap', name)
        heatmap_high_name = os.path.join('../../dataset/CITYSCAPES/CITY/relative_high_heatmap', name)

        inpainting = scipy.misc.imread(inpainting_name[i]).astype(np.float32)
        heatmap = scipy.misc.imread(heatmap_name).astype(np.float32)
        yo = np.nonzero(heatmap == 255)
        inpainting_v = np.copy(inpainting)
        inpainting_v[yo + (0,)] += 120
        inpainting_v[yo + (1,)] += 120
        inpainting_v[yo + (2,)] += 120
        inpainting_v[np.nonzero(inpainting_v>255)] = 255

        v1 = np.hstack((inpainting, inpainting_v))

        inpainting_high = scipy.misc.imread(inpainting_high_name[i]).astype(np.float32)
        name = inpainting_high_name[i].split('/')[-1]
        heatmap_high_name = os.path.join('../../dataset/CITYSCAPES/CITY/relative_high_heatmap', name)
        heatmap_high = scipy.misc.imread(heatmap_high_name).astype(np.float32)
        yo = np.nonzero(heatmap_high == 255)
        inpainting_high_v = np.copy(inpainting_high)
        inpainting_high_v[yo + (0,)] += 120
        inpainting_high_v[yo + (1,)] += 120
        inpainting_high_v[yo + (2,)] += 120
        inpainting_high_v[np.nonzero(inpainting_high_v>255)] = 255

        v2 = np.hstack((inpainting_high, inpainting_high_v))

        v = np.vstack((v1, v2))
        scipy.misc.imsave('instance_visual/{}'.format(name), v.astype(np.uint8))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def heatmap_alpha():
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test/frankfurt_000000_010763_leftImg8bit_target_heatmap.png'
    img = scipy.misc.imread(dir)

    img_25 = np.copy(img)
    img_25[np.nonzero(img > 63)] = 255
    img_25[np.nonzero(img < 63)] = 0

    img_50 = np.copy(img)
    img_50[np.nonzero(img > 127)] = 255
    img_50[np.nonzero(img < 127)] = 0

    img_75 = np.copy(img).astype(np.float32)
    img_75 /= 255
    img_75 = sigmoid(img_75)
    img_75 *= 255
    #img_75[np.nonzero(img > 191)] = 255
    #img_75[np.nonzero(img < 191)] = 0

    v1 = np.hstack((img, img_25))
    v2 = np.hstack((img_50, img_75))
    v = np.vstack((v1, v2))

    scipy.misc.imsave('yo.png', v)


def synthesis_visual():
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test_low'
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    dir = '/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val'
    semantic_name = []
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder, "*_labelIds.png")
        semantic_name.extend(glob(path))
    semantic_name = sorted(semantic_name)

    ped_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES/image'
    mask_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES/mask'
    pedestrian_name = sorted(glob(os.path.join(ped_dir, "*.png")))
    mask_name = sorted(glob(os.path.join(mask_dir, "*.png")))
    data_len = len(heatmap_name)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]
        city = name[i].split('_')[0]

        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        semantic = scipy.misc.imread(semantic_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        heatamp = scipy.misc.imresize(heatamp, 2., interp='bilinear', mode=None)
        pedestrian = scipy.misc.imread(pedestrian_name[500]).astype(np.float32)
        mask = scipy.misc.imread(mask_name[500]).astype(np.float32) / 255
        mask = np.dstack((mask, mask, mask))
        pedestrian *= mask
        ped_height, ped_width = 114, 59


        sig, threshold = 3, 10
        print(sig, threshold)
        blurred = gaussian_filter(heatamp, sigma=sig)
        blurred[np.nonzero(blurred >= threshold)] = 255
        blurred[np.nonzero(blurred < threshold)] = 0

        blurred_index = np.nonzero(blurred == 255)
        image_v = np.copy(image)
        image_v[blurred_index + (0,)] += 80
        image_v[blurred_index + (1,)] += 80
        image_v[blurred_index + (2,)] += 80
        image_v[np.nonzero(image_v > 255)] = 255

        all_labels = skimage.measure.label(blurred, background=0)
        for label in range(1, all_labels.max()+1):
            label_map = np.zeros((256, 512), dtype=np.float32)
            human_pixel = np.nonzero(all_labels == label)
            miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
            heigh, width = maxy - miny, maxx - minx
            ceny, cenx = np.round(np.mean(human_pixel[0])), np.round(np.mean(human_pixel[1]))
            human_pixel_size = len(human_pixel[0])
            if human_pixel_size < 1000:
                continue

            label_map[human_pixel] = 255
            cut_upy, cut_dny, cut_lx, cut_rx = 0, 0, 0, 0
            lx = int(np.round(cenx - ped_width / 2))
            if lx < 0:
                cut_lx = -lx
                lx = 0

            upy = int(np.round(ceny - ped_height/2))
            if upy < 0:
                cut_upy = -upy
                upy = 0

            rx = lx + ped_width - cut_lx
            if rx > 512-1:
                cut_rx = rx - (512-1)
                rx = 512-1

            dny = upy + ped_height - cut_upy
            if dny > 256-1:
                cut_dny = dny - (256-1)
                dny = 256-1

            hole = (1-mask)[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :] * image[upy:dny, lx:rx, :]
            fill = hole + pedestrian[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :]
            image_synthesis = np.copy(image)
            image_synthesis[upy:dny, lx:rx, :] = fill
            #scipy.misc.imshow(label_map)
            #for ped_name in  pedestrian_name:
            #    ped_name = ped_name.split('/')[-1]
            #    ped_name_split = ped_name.split('_')
            #    height, width = int(ped_name_split[0]), int(ped_name_split[1])



        scipy.misc.imsave('ICCV/synthesis_visual/{}_image.png'.format(name), image.astype(np.uint8))
        scipy.misc.imsave('ICCV/synthesis_visual/{}_semantic.png'.format(name), semantic_vidualize(semantic).astype(np.uint8))
        scipy.misc.imsave('ICCV/synthesis_visual/{}_heatmap.png'.format(name), image_v.astype(np.uint8))
        scipy.misc.imsave('ICCV/synthesis_visual/{}_synthesis.png'.format(name), image_synthesis.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_seg.png'.format(name), blurred.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_ped.png'.format(name), pedestrian.astype(np.uint8))

        break


def valid_visual():
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/valid'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    gt_name = sorted(glob(os.path.join(dir, "*_gt.png")))
    data_len = len(heatmap_name)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]

        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        gt = scipy.misc.imread(gt_name[i]).astype(np.float32)

        gt_index = np.nonzero(gt == 255)
        image_gt = np.copy(image)
        image_gt[gt_index + (0,)] += 120
        image_gt[gt_index + (1,)] += 120
        image_gt[gt_index + (2,)] += 120
        image_gt[np.nonzero(image_gt>255)] = 255

        sig, threshold = 7, 10
        print(sig, threshold)
        blurred = gaussian_filter(heatamp, sigma=sig)
        blurred = np.dstack((blurred, blurred, blurred))
        image_heatmap = np.copy(image)
        image_heatmap += blurred
        image_heatmap[np.nonzero(image_heatmap > 255)] = 255
        #blurred[np.nonzero(blurred >= threshold)] = 255
        #blurred[np.nonzero(blurred < threshold)] = 0

        #blurred_index = np.nonzero(blurred == 255)
        #image_v = np.copy(image)
        #image_v[blurred_index + (0,)] += 80
        #image_v[blurred_index + (1,)] += 80
        #image_v[blurred_index + (2,)] += 80
        #image_v[np.nonzero(image_v > 255)] = 255


        scipy.misc.imsave('ICCV/heatmap_visual/valid/{}_gt.png'.format(name), image_gt.astype(np.uint8))
        scipy.misc.imsave('ICCV/heatmap_visual/valid/{}_heatmap.png'.format(name), image_heatmap.astype(np.uint8))


def test_visual():
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test_low'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    data_len = len(heatmap_name)

    sig, threshold = 3, 10
    print(sig, threshold)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]

        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        heatamp = np.dstack((heatamp, heatamp, heatamp)) * 3


        #blurred = gaussian_filter(heatamp, sigma=sig)
        #blurred = np.dstack((blurred, blurred, blurred))
        #blurred[np.nonzero(blurred >= threshold)] = 255
        #blurred[np.nonzero(blurred < threshold)] = 0

        blurred_index = np.nonzero(heatamp> 1)
        image_heatmap = np.copy(image)
        #heatamp = gaussian_filter(heatamp, sigma=1)
        image_heatmap += heatamp
        image_heatmap[np.nonzero(image_heatmap>255)] = 255


        scipy.misc.imsave('ICCV/heatmap_visual/test_low/{}.png'.format(name), image.astype(np.uint8))
        scipy.misc.imsave('ICCV/heatmap_visual/test_low/{}_heatmap.png'.format(name), image_heatmap.astype(np.uint8))


def heatmap_test_visual():
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test_low'
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    data_len = len(heatmap_name)

    sig, threshold = 3, 15
    print(sig, threshold)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]

        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        heatamp = scipy.misc.imresize(heatamp, 2., interp='bilinear', mode=None)


        blurred = gaussian_filter(heatamp, sigma=sig)
        blurred[np.nonzero(blurred >= threshold)] = 255
        blurred = np.dstack((blurred, blurred, blurred))

        # blurred = gaussian_filter(heatamp, sigma=sig)
        # blurred = np.dstack((blurred, blurred, blurred))
        # blurred[np.nonzero(blurred >= threshold)] = 255
        # blurred[np.nonzero(blurred < threshold)] = 0

        blurred_index = np.nonzero(blurred > threshold)
        image_heatmap = np.copy(image)
        image_heatmap[blurred_index] += 120
        image_heatmap[np.nonzero(image_heatmap > 255)] = 255

        scipy.misc.imsave('ICCV/heatmap_test_visual/test_low/{}.png'.format(name), image.astype(np.uint8))
        scipy.misc.imsave('ICCV/heatmap_test_visual/test_low/{}_heatmap.png'.format(name),
                          image_heatmap.astype(np.uint8))


def valid():
    data_set_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/relative_high_image'
    data_name = glob(os.path.join(data_set_dir, "*.png"))
    np.random.shuffle(data_name)
    pure_name = []
    for name in data_name:
        p = name.split('/')[-1]
        pure_name.append(p)
    p_valid = pure_name[0:500]
    p_train = pure_name[500:-1]
    with h5py.File('relative_high.h5', 'w') as f:
        f['valid'] = p_valid
        f['train'] = p_train


def read():
    a = h5py.File('relative_context.h5')
    image_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/relative_context_image'
    heatmap_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/relative_context_heatmap'
    valid = a['valid']
    train = a['train']
    i, vl, tl = 0, len(valid), len(train)
    for name in valid:
        print("{:d}/{:d}".format(i, vl))
        image = scipy.misc.imread(os.path.join(image_dir, name))
        heatmap = scipy.misc.imread(os.path.join(heatmap_dir, name))
        scipy.misc.imsave(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/valid/relative_context_image', name), image)
        scipy.misc.imsave(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/valid/relative_context_heatmap', name), heatmap)
        i += 1

    i = 0
    for name in train:
        print("{:d}/{:d}".format(i, tl))
        image = scipy.misc.imread(os.path.join(image_dir, name))
        heatmap = scipy.misc.imread(os.path.join(heatmap_dir, name))
        scipy.misc.imsave(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/train/relative_context_image', name), image)
        scipy.misc.imsave(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/train/relative_context_heatmap', name), heatmap)
        i +=1


class PedestrianImage:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.aspect = float(width) / float(height)


def pedestrian_from_city():
    out = '../../dataset/pedestrian/CITYSCAPES'
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)
    pedestrian_list = []
    total = 0
    for index in range(0, length):
        name = human_file_name[index]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        semantic_name = '{}_gtFine_labelIds.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]


        image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name)).astype(np.float32)
        semantic = scipy.misc.imread(os.path.join(dataset_dir, 'fine_label', semantic_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))

        '''
        semantic[np.nonzero(semantic != 24)] = 0
        semantic[np.nonzero(semantic != 0)] = 1
        all_labels = skimage.measure.label(semantic, background=0)
        for label in range(1, all_labels.max()+1):
            label_map = np.zeros((256, 512), dtype=np.float32)
            human_pixel = np.nonzero(all_labels == label)
            miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
            height, width = maxy - miny, maxx - minx
            if  height < 15 or width < 15:
                continue
            aspect = float(width) / float(height)
            if aspect < 0.7 :
                continue
            #    continue
            if minx < 0:
                minx = 0
            if miny < 0:
                miny = 0
            if maxx > 511:
                maxx = 511
            if maxy > 255:
                maxy = 255
            label_map[human_pixel] = 1
            instance_name = '{:d}_{}.png'.format(label, name)
            scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES_multiple/mask/{}'.format(instance_name), label_map[miny:maxy, minx:maxx].astype(np.uint8))
            scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES_multiple/image/{}'.format(instance_name), image[miny:maxy, minx:maxx].astype(np.uint8))

            crop_mask = np.dstack((label_map[miny:maxy, minx:maxx], label_map[miny:maxy, minx:maxx], label_map[miny:maxy, minx:maxx]))
            crop_image = image[miny:maxy, minx:maxx] * crop_mask
            scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES_multiple/crop/{}'.format(instance_name), crop_image.astype(np.uint8))

            pedestrian_image = PedestrianImage(instance_name, width=width, height=height)
            pedestrian_list.append(pedestrian_image)

        '''

        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        objects = label_instance['objects']
        instance_num = 0
        for obj in objects:
            if obj['label'] == 'person':
                #print(object['polygon'])
                polygon = [tuple(poly) for poly in obj['polygon']]
                poly = np.copy(polygon)
                minx, miny, maxx, maxy = min(poly[:, 0]), min(poly[:, 1]), max(poly[:, 0]), max(poly[:, 1])
                minx, miny, maxx, maxy = minx/4, miny/4, maxx/4, maxy/4.0
                if minx < 0:
                    minx = 0
                if miny < 0:
                    miny = 0
                if maxx > 511:
                    maxx = 511
                if maxy > 255:
                    maxy = 255
                height, width = maxy - miny, maxx - minx

                if height < 20 or width < 15:
                    continue
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                yo = np.array(img).astype(np.float32)
                img = scipy.misc.imresize(yo, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                #mask[np.nonzero(mask > 0)] = 255
                pedestrian_image = image[miny:maxy, minx:maxx]

                stroe_name = '{}_{:d}.png'.format(name, instance_num)
                scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES_new/image/{}'.format(
                    stroe_name), pedestrian_image.astype(np.uint8))
                scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES_new/mask/{}'.format(
                    stroe_name), mask[miny:maxy, minx:maxx].astype(np.uint8))

                #mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)

                mask_index = np.nonzero(mask==0)
                mask /= 255
                image_mask = np.copy(image)
                image_mask *= np.dstack((mask, mask, mask))
                pedestrian_image = image_mask[miny:maxy, minx:maxx]
                scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES_new/crop/{}'.format(
                    stroe_name), pedestrian_image.astype(np.uint8))
                pedestrian_image = PedestrianImage(stroe_name, width=width, height=height)
                pedestrian_list.append(pedestrian_image)

                instance_num += 1
                total +=1

        #break

    print(total)
    file_obj = open('/data/vllab1/dataset/pedestrian/CITYSCAPES_new/pedestrian_list.pkl', 'wb')
    pickle.dump(pedestrian_list, file_obj)
    file_obj.close()


def synthesis_visual_fit():
    # Image, heatmap, semantic
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test_low'
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    dir = '/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val'
    semantic_name = []
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder, "*_labelIds.png")
        semantic_name.extend(glob(path))
    semantic_name = sorted(semantic_name)
    data_len = len(heatmap_name)

    # pedestrian image, mask, list
    ped_single_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_new/image'
    mask_single_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_new/mask'
    dataset_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_new'
    file_name = os.path.join(dataset_dir, 'pedestrian_list.pkl')
    file_obj = open(file_name, 'r')
    pedestrian_list = pickle.load(file_obj)
    # pedestrian_multi image, mask, list
    ped_mul_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_multiple/image'
    mask_mul_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_multiple/mask'
    dataset_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_multiple'
    file_name = os.path.join(dataset_dir, 'pedestrian_list.pkl')
    file_obj = open(file_name, 'r')
    pedestrian_multi_list = pickle.load(file_obj)

    # parameter
    sig, threshold = 1, 1
    print(sig, threshold)
    out_dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/{}_{}'.format(sig, threshold)
    #out_dir = '/data/vllab1/Github/streetview_synthesize/ICCV/wrong_random'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        # Name split
        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]
        city = name.split('_')[0]
        # get image, semantic, heatmap_low
        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        semantic = scipy.misc.imread(semantic_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        heatamp = scipy.misc.imresize(heatamp, 2., interp='bilinear', mode=None)
        # Segment heatmap
        blurred = gaussian_filter(heatamp, sigma=sig)
        blurred[np.nonzero(blurred >= threshold)] = 255
        blurred[np.nonzero(blurred < threshold)] = 0
        # Visual segment heatmap
        blurred_index = np.nonzero(blurred == 255)
        image_v = np.copy(image)
        image_v[blurred_index + (0,)] += 80
        image_v[blurred_index + (1,)] += 80
        image_v[blurred_index + (2,)] += 80
        image_v[np.nonzero(image_v > 255)] = 255

        # Use segment heatmap seythesis image
        image_synthesis = np.copy(image)
        all_labels = skimage.measure.label(blurred, background=0)
        for label in range(1, all_labels.max()+1):
            human_pixel = np.nonzero(all_labels == label)
            miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
            height, width = maxy - miny, maxx - minx
            # invalid region
            if height < 20 or width < 15:
                continue
            semantic_check_region = semantic[maxy*4-5:maxy*4+5, minx*4:maxx*4]
            road_pixel = len(np.nonzero(semantic_check_region == 7)[0])
            sidewalk_pixel = len(np.nonzero(semantic_check_region == 8)[0])
            parking_pixel = len(np.nonzero(semantic_check_region == 9)[0])
            rail_pixel = len(np.nonzero(semantic_check_region == 10)[0])
            if road_pixel + sidewalk_pixel + parking_pixel + rail_pixel < 10:
                image_v[human_pixel + (0,)] = 255
                continue
            aspect = float(width) / float(height)
            if aspect > 0.7:
                image_v[human_pixel + (1,)] = 255
                pedestrian_fit_all = pedestrian_multi_list
                ped_dir = ped_mul_dir
                mask_dir = mask_mul_dir
                heat_alpha = 1
            else:
                image_v[human_pixel + (2,)] = 255
                pedestrian_fit_all = pedestrian_list
                ped_dir = ped_single_dir
                mask_dir = mask_single_dir
                heat_alpha = 255

            ceny, cenx = np.round(np.mean(human_pixel[0])), np.round(np.mean(human_pixel[1]))
            pedestrian_fit_all.sort(key=lambda x: abs(x.width-width) + abs(x.height-height))
            #random_sel = np.random.randint(0, len(pedestrian_fit_all))

            sel_ped_name, ped_width, ped_height = pedestrian_fit_all[0].name, pedestrian_fit_all[0].width, pedestrian_fit_all[0].height
            sel_ped_image = scipy.misc.imread(os.path.join(ped_dir, sel_ped_name)).astype(np.float32)
            sel_ped_mask = scipy.misc.imread(os.path.join(mask_dir, sel_ped_name)).astype(np.float32)/heat_alpha
            sel_ped_mask = np.dstack((sel_ped_mask, sel_ped_mask, sel_ped_mask))
            sel_ped_image *= sel_ped_mask


            #road_pixel = np.nonzero(semantic == 7)
            #sidewalk_pixel = np.nonzero(semantic == 8)
            #parking_pixel = np.nonzero(semantic == 9)
            #rail_pixel = np.nonzero(semantic == 10)
            #possible_random = np.hstack((road_pixel, sidewalk_pixel, parking_pixel, rail_pixel))
            #ramdom_sel = np.random.randint(0, len(possible_random[0]))
            #ceny, cenx = possible_random[0][ramdom_sel] / 4, possible_random[1][ramdom_sel] / 4
            #ceny -= ped_height/2


            ped_width, ped_height =  int(ped_width), int(ped_height)
            cut_upy, cut_dny, cut_lx, cut_rx = 0, 0, 0, 0
            lx = int(np.round(cenx - ped_width / 2))
            if lx < 0:
                cut_lx = -lx
                lx = 0

            upy = int(np.round(ceny - ped_height/2))
            if upy < 0:
                cut_upy = -upy
                upy = 0

            rx = lx + ped_width - cut_lx
            if rx > 512-1:
                cut_rx = rx - (512-1)
                rx = 512-1

            dny = upy + ped_height - cut_upy
            if dny > 256-1:
                cut_dny = dny - (256-1)
                dny = 256-1

            hole = (1-sel_ped_mask)[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :] * image_synthesis[upy:dny, lx:rx, :]
            fill = hole + sel_ped_image[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :]
            image_synthesis[upy:dny, lx:rx, :] = fill

            #image_synthesis[miny:maxy, minx:maxx, :] *= 1 - sel_ped_mask
            #image_synthesis[miny:maxy, minx:maxx, :] += sel_ped_image * sel_ped_mask
            #scipy.misc.imshow(label_map)
            #for ped_name in  pedestrian_name:
            #    ped_name = ped_name.split('/')[-1]
            #    ped_name_split = ped_name.split('_')
            #    height, width = int(ped_name_split[0]), int(ped_name_split[1])



        #scipy.misc.imsave('ICCV/synthesis_visual/{}_image.png'.format(name), image.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_semantic.png'.format(name), semantic_vidualize(semantic).astype(np.uint8))
        scipy.misc.imsave(os.path.join(out_dir, '{}_image_{}_{}.png'.format(name, sig, threshold)), image_v.astype(np.uint8))
        scipy.misc.imsave(os.path.join(out_dir, '{}_heatmap_{}_{}.png'.format(name, sig, threshold)), image_synthesis.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_seg.png'.format(name), blurred.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_ped.png'.format(name), pedestrian.astype(np.uint8))
        #break


def pedestrian_from_Ped():
    dir = '/data/vllab1/dataset/pedestrian/completeData/image'
    image_name = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/dataset/pedestrian/completeData/mask'
    mask_name = sorted(glob(os.path.join(dir, "*.png")))
    data_len = len(mask_name)

    pedestrian_list = []
    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        mask = scipy.misc.imread(mask_name[i]).astype(np.float32)
        name = image_name[i].split('/')[-1]
        human_pixel = np.nonzero(mask == 255)
        miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
        height, width = maxy - miny, maxx - minx
        pedestrian_image = PedestrianImage(name, width=width, height=height)
        pedestrian_list.append(pedestrian_image)

        scipy.misc.imsave('/data/vllab1/dataset/pedestrian/Ped/image/{}'.format(name), image[miny:maxy, minx:maxx, :])
        scipy.misc.imsave('/data/vllab1/dataset/pedestrian/Ped/mask/{}'.format(name), mask[miny:maxy, minx:maxx])


    file_obj = open('pedestrian_list.pkl', 'wb')
    pickle.dump(pedestrian_list, file_obj)
    file_obj.close()


def wrong_similiar():
    dataset_dir = '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/test'
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "munich_000094_000019_leftImg8bit.png")
        data.extend(glob(path))
    image_name = sorted(data)
    data_len = len(image_name)

    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    np.random.shuffle(human_file_name)
    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        name = image_name[i].split('/')[-1]
        human_name = human_file_name[i%192]
        city = human_name.split('_')[0]
        image = scipy.misc.imread(image_name[i]).astype(np.float32)

        human_image_name = '{}_leftImg8bit.png'.format(human_name)
        human_mask_name = '{}_gtFine_labelIds.png'.format(human_name)
        human_image = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train',
                                                     city, human_image_name)).astype(np.float32)
        human_mask = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                                    city, human_mask_name)).astype(np.float32)

        human_mask[np.nonzero(human_mask != 24)] = 0
        human_mask[np.nonzero(human_mask != 0)] = 1

        human_mask = np.dstack((human_mask, human_mask, human_mask))
        image = image * (1-human_mask) + human_image * human_mask
        image = scipy.misc.imresize(image, 0.25, interp='bilinear', mode=None)

        #img_ret = poissonblending.blend(image, human_image, human_mask, offset=(0, 0))


        scipy.misc.imsave('/data/vllab1/CVPR/demo/baseline_similiar/{}'.format(name), image)
        #break


def wrong_similiar_random():
    dataset_dir = '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/test'
    data = []
    for folder in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, folder, "berlin_000116_000019_leftImg8bit.png")
        data.extend(glob(path))
    image_name = sorted(data)
    data_len = len(image_name)

    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    np.random.shuffle(human_file_name)
    for i in range(0, 150):
        print('{:d}/{:d}'.format(i, data_len))
        name = image_name[0].split('/')[-1]
        human_name = human_file_name[i%192]
        city = human_name.split('_')[0]
        image = scipy.misc.imread(image_name[0]).astype(np.float32)

        human_image_name = '{}_leftImg8bit.png'.format(human_name)
        human_mask_name = '{}_gtFine_labelIds.png'.format(human_name)
        human_image = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train',
                                                     city, human_image_name)).astype(np.float32)
        human_mask = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                                    city, human_mask_name)).astype(np.float32)

        human_mask[np.nonzero(human_mask != 24)] = 0
        human_mask[np.nonzero(human_mask != 0)] = 1

        human_mask = np.dstack((human_mask, human_mask, human_mask))
        image = image * (1-human_mask) + human_image * human_mask
        image = scipy.misc.imresize(image, 0.25, interp='bilinear', mode=None)

        #img_ret = poissonblending.blend(image, human_image, human_mask, offset=(0, 0))


        scipy.misc.imsave('/data/vllab1/CVPR/demo/baseline_similiar/{:d}_{}'.format(i, name), image)
        #break


def pedestrian_from_city_big():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    np.random.shuffle(human_file_name)
    data_len = len(human_file_name)
    pedestrian_list = []
    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        human_name = human_file_name[i]
        city = human_name.split('_')[0]

        human_image_name = '{}_leftImg8bit.png'.format(human_name)
        human_mask_name = '{}_gtFine_labelIds.png'.format(human_name)
        human_image = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train',
                                                     city, human_image_name)).astype(np.float32)
        human_mask = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                                    city, human_mask_name)).astype(np.float32)

        human_mask[np.nonzero(human_mask != 24)] = 0
        human_mask[np.nonzero(human_mask != 0)] = 1

        all_labels = skimage.measure.label(human_mask, background=0)
        human_mask = np.dstack((human_mask, human_mask, human_mask))
        image = human_image * (1 - human_mask) + human_image * human_mask
        for label in range(1, all_labels.max() + 1):
            human_pixel = np.nonzero(all_labels == label)
            miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
            height, width = maxy - miny, maxx - minx
            if height < 100 or width < 100:
                continue
            aspect = float(width) / float(height)
            if aspect < 0.5:
                continue
            sel_image = image[miny:maxy, minx:maxx, :]
            sel_mask = human_mask[miny:maxy, minx:maxx, 0]
            pedestrian_image = PedestrianImage(human_image_name, width=width, height=height)
            pedestrian_list.append(pedestrian_image)

            scipy.misc.imsave('/data/vllab1/dataset/pedestrian/CITYSCAPES_big/image/{}_{}'.format(label, human_image_name), sel_image)
            scipy.misc.imsave('/data/vllab1/dataset/pedestrian/CITYSCAPES_big/mask/{}_{}'.format(label, human_image_name), sel_mask)


    file_obj = open('city_big_list.pkl', 'wb')
    pickle.dump(pedestrian_list, file_obj)
    file_obj.close()


def wrong_random():
    # Image, heatmap, semantic
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test_low'
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    dir = '/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val'
    semantic_name = []
    for folder in os.listdir(dir):
        path = os.path.join(dir, folder, "*_labelIds.png")
        semantic_name.extend(glob(path))
    semantic_name = sorted(semantic_name)
    data_len = len(heatmap_name)

    # pedestrian image, mask, list
    ped_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_single/image'
    mask_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_single/mask'
    dataset_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_single'
    file_name = os.path.join(dataset_dir, 'pedestrian_list_single.pkl')
    file_obj = open(file_name, 'r')
    pedestrian_list = pickle.load(file_obj)

    # parameter
    sig, threshold = 2, 5
    print(sig, threshold)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        # Name split
        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]
        city = name.split('_')[0]
        # get image, semantic, heatmap_low
        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        semantic = scipy.misc.imread(semantic_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        heatamp = scipy.misc.imresize(heatamp, 2., interp='bilinear', mode=None)
        # Segment heatmap
        blurred = gaussian_filter(heatamp, sigma=sig)
        blurred[np.nonzero(blurred >= threshold)] = 255
        blurred[np.nonzero(blurred < threshold)] = 0
        # Visual segment heatmap
        blurred_index = np.nonzero(blurred == 255)
        image_v = np.copy(image)
        image_v[blurred_index + (0,)] += 80
        image_v[blurred_index + (1,)] += 80
        image_v[blurred_index + (2,)] += 80
        image_v[np.nonzero(image_v > 255)] = 255

        # Use segment heatmap seythesis image
        image_synthesis = np.copy(image)
        all_labels = skimage.measure.label(blurred, background=0)
        for label in range(1, all_labels.max() + 1):
            human_pixel = np.nonzero(all_labels == label)
            miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
            height, width = maxy - miny, maxx - minx
            # valid region check
            if height < 5 or width < 9:
                continue
            semantic_chekch_region = semantic[(maxy-3)*4:(maxy+3)*4, minx*4:maxx*4]
            road_pixel = len(np.nonzero(semantic_chekch_region == 7)[0])
            sidewalk_pixel = len(np.nonzero(semantic_chekch_region == 8)[0])
            parking_pixel = len(np.nonzero(semantic_chekch_region == 9)[0])
            rail_track_pixel = len(np.nonzero(semantic_chekch_region == 10)[0])
            if road_pixel + sidewalk_pixel + parking_pixel + rail_track_pixel < 5:
                image_v[human_pixel + (0,)] = 255
                continue
            # valid aspect check
            aspect = float(width) / float(height)
            # TODO
            if aspect > 0.7:
                image_v[human_pixel + (1,)] = 255
                continue
            else:
                image_v[human_pixel + (2,)] = 255

            pedestrian_list.sort(key=lambda x: abs(x.aspect - aspect))
            pedestrian_list_top10 = pedestrian_list
            pedestrian_list_top10.sort(key=lambda x: abs(x.width - width) + abs(x.height - height))
            #sel_index = np.random.randint(0, pedestrian_list_len)
            sel_index = 0
            #cenx, ceny = np.random.randint(0, 512), np.random.randint(0, 256)
            ceny, cenx = np.round(np.mean(human_pixel[0])), np.round(np.mean(human_pixel[1]))

            sel_ped_name, ped_width, ped_height = \
                pedestrian_list_top10[sel_index].name, pedestrian_list_top10[sel_index].width, pedestrian_list_top10[sel_index].height
            sel_ped_image = scipy.misc.imread(os.path.join(ped_dir, sel_ped_name)).astype(np.float32)
            sel_ped_mask = scipy.misc.imread(os.path.join(mask_dir, sel_ped_name)).astype(np.float32)/255

            if False:

                #sel_ped_mask = np.dstack((sel_ped_mask, sel_ped_mask, sel_ped_mask))
                #sel_ped_image *= sel_ped_mask
                ped_width, ped_height = int(ped_width) + 20, int(ped_height) + 20

                cut_upy, cut_dny, cut_lx, cut_rx = 0, 0, 0, 0
                lx = int(np.round(cenx - ped_width / 2))
                if lx < 0:
                    cut_lx = -lx
                    lx = 0

                upy = int(np.round(ceny - ped_height/2))
                if upy < 0:
                    cut_upy = -upy
                    upy = 0

                rx = lx + ped_width - cut_lx
                if rx > 512-1:
                    cut_rx = rx - (512-1)
                    rx = 512-1

                dny = upy + ped_height - cut_upy
                if dny > 256-1:
                    cut_dny = dny - (256-1)
                    dny = 256-1
                #hole = (1-sel_ped_mask)[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :] * image_synthesis[upy:dny, lx:rx, :]
                #fill = hole + sel_ped_image[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :]
                #image_synthesis[upy:dny, lx:rx, :] = fill

                for_poisson_image = np.ones((ped_height, ped_width, 3)) * 255
                for_poisson_image[10:-10, 10:-10, :] = sel_ped_image
                for_poisson_mask = np.zeros((ped_height, ped_width))
                for_poisson_mask[10:-10, 10:-10] = sel_ped_mask
                for_poisson_mask = scipy.ndimage.morphology.binary_dilation(for_poisson_mask, iterations=1).astype(for_poisson_mask.dtype)
                image_synthesis = poissonblending.blend(image_synthesis, for_poisson_image, for_poisson_mask, offset=(upy-10, lx-10))
            else:
                sel_ped_mask = np.dstack((sel_ped_mask, sel_ped_mask, sel_ped_mask))
                sel_ped_image *= sel_ped_mask
                ped_width, ped_height = int(ped_width), int(ped_height)

                cut_upy, cut_dny, cut_lx, cut_rx = 0, 0, 0, 0
                lx = int(np.round(cenx - ped_width / 2))
                if lx < 0:
                    cut_lx = -lx
                    lx = 0

                upy = int(np.round(ceny - ped_height / 2))
                if upy < 0:
                    cut_upy = -upy
                    upy = 0

                rx = lx + ped_width - cut_lx
                if rx > 512 - 1:
                    cut_rx = rx - (512 - 1)
                    rx = 512 - 1

                dny = upy + ped_height - cut_upy
                if dny > 256 - 1:
                    cut_dny = dny - (256 - 1)
                    dny = 256 - 1
                hole = (1-sel_ped_mask)[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :] * image_synthesis[upy:dny, lx:rx, :]
                fill = hole + sel_ped_image[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :]
                image_synthesis[upy:dny, lx:rx, :] = fill

        # scipy.misc.imsave('ICCV/synthesis_visual/{}_image.png'.format(name), image.astype(np.uint8))
        # scipy.misc.imsave('ICCV/synthesis_visual/{}_semantic.png'.format(name), semantic_vidualize(semantic).astype(np.uint8))
        scipy.misc.imsave('ICCV/wrong_random/{}_heatmap.png'.format(name), image_v.astype(np.uint8))
        scipy.misc.imsave('ICCV/wrong_random/{}_synthesis.png'.format(name), image_synthesis.astype(np.uint8))
        #scipy.misc.imsave('ICCV/wrong_random/{}_semantic.png'.format(name), semantic_vidualize(semantic).astype(np.uint8))
        #break


def creat_mask_image_for_high():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_name = os.path.join(dataset_dir, 'human_w.pkl')
    style_name = os.path.join(dataset_dir, 'human_wo.pkl')
    file_obj = open(file_name, 'r')
    style_obj = open(style_name, 'r')
    human_file_name = pickle.load(file_obj)
    style_file_name = pickle.load(style_obj)
    np.random.shuffle(style_file_name)
    for index in range(0, len(human_file_name)):
        print ('%d/%d' % (index, len(human_file_name)))
        name = human_file_name[index]
        style = style_file_name[index]

        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        style_name = '{}_leftImg8bit.png'.format(style)

        image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name))
        mask = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name))
        style_image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', style_name))

        mask_index = np.nonzero(mask==255)
        image[mask_index + (0,)] = 0
        image[mask_index + (1,)] = 0
        image[mask_index + (2,)] = 0

        style_image = np.copy(image)
        style_image[0:128, :, :] = image[128:256, :, :]

        scipy.misc.imsave(os.path.join('style_2', image_name), style_image)
        break


def split_inpaint_valid_train():
    data_set_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/inpainting_context'
    data = glob(os.path.join(data_set_dir, "*.png"))
    data_len = len(data)

    np.random.shuffle(data)

    for i in range(0, 10):
        print('{:d}/{:d}'.format(i, data_len))
        name = data[i].split('/')[-1]
        inpaint_context = scipy.misc.imread(
            os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/inpainting_context', name))
        inpaint_context_low = scipy.misc.imresize(inpaint_context, 0.5, interp='bilinear', mode=None)
        inpaint_high = scipy.misc.imread(
            os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/inpainting_high', name))
        inpaint_high_low = scipy.misc.imresize(inpaint_high, 0.5, interp='bilinear', mode=None)

        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/valid/inpainting_context', name), inpaint_context)
        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/valid/inpainting_context_low', name), inpaint_context_low)
        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/valid/inpainting_high', name), inpaint_high)
        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/valid/inpainting_high_low', name), inpaint_high_low)

        print(name)

    for i in range(10, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        name = data[i].split('/')[-1]
        inpaint_context = scipy.misc.imread(
            os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/inpainting_context', name))
        inpaint_context_low = scipy.misc.imresize(inpaint_context, 0.5, interp='bilinear', mode=None)
        inpaint_high = scipy.misc.imread(
            os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY/inpainting_high', name))
        inpaint_high_low = scipy.misc.imresize(inpaint_high, 0.5, interp='bilinear', mode=None)

        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/train/inpainting_context', name), inpaint_context)
        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/train/inpainting_context_low', name), inpaint_context_low)
        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/train/inpainting_high', name), inpaint_high)
        scipy.misc.imsave(
            os.path.join('/data/vllab1/dataset/Relative/train/inpainting_high_low', name), inpaint_high_low)

        print(name)


def create_instance_train_valid():
    data_set_dir = '../../dataset/Relative/valid/inpainting_high'
    data = glob(os.path.join(data_set_dir, "*.png"))
    length = len(data)

    for index in range(0, length):
        name = data[index].split('/')[-1].split('.')[0]
        name_split = name.split('_')
        name = name_split[0] + '_' + name_split[1] + '_' + name_split[2]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]

        # astype(np.float32) somehow do the normalize
        # should use if the inpainting method is context
        image = scipy.misc.imread(os.path.join('../../dataset/CITYSCAPES/CITY', 'fine_image', image_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join('../../dataset/CITYSCAPES/CITY', 'fine_mask', mask_name)).astype(np.float32)
        inpainting = scipy.misc.imread(os.path.join('../../dataset/Relative/valid/', 'inpainting_high', image_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))

        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        for obj in objects:
            if obj['label'] == 'person':
                #print(object['polygon'])
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                mask[np.nonzero(mask > 0)] = 255
                mask_instance.append(mask)

                human_pixel = np.nonzero(mask == 255)
                human_pixel_set.append(len(human_pixel[0]))
                # human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                # if human_ratio > alpha:
                #     mask_instance_sel.append(instance_num)
                instance_num += 1

        human_pixel_index = np.argsort(human_pixel_set)
        mask_instance_sorted = []
        for sorted_index in human_pixel_index:
            mask_instance_sorted.append(mask_instance[sorted_index])
        print(instance_num)
        if instance_num > 20:
            # Select the up to 10 biggest instance
            mask_instance_sorted = mask_instance_sorted[-21:-1]
            instance_num = 20

        instance_array = np.arange(instance_num)
        possible_c = np.arange(instance_num)
        np.random.shuffle(possible_c)
        if len(possible_c) > 10:
            # Cn0 + Cn1 + Cn2 + ... + Cnn-1
            # Random select up to 10 possible combinations
            possible_c = possible_c[0:10]
        for c_idx in possible_c:
            instance_combine = list(itertools.combinations(instance_array, c_idx))
            np.random.shuffle(instance_combine)
            if len(instance_combine) > 4:
                # Random select up to 4 combinations under one possible combinations. ex:
                # Cn3: (1, 2, 8), (2, 4, 6), (5, 18, 19), (3, 4, 5)
                instance_combine = instance_combine[0:4]
            for i_idx, combine_list in enumerate(instance_combine):
                img = np.copy(image)
                mask = np.zeros((256, 512), dtype=np.int8)
                heatmap = np.zeros((256, 512), dtype=np.int8)
                for m_idx, m in enumerate(mask_instance_sorted):
                    if m_idx in combine_list:
                        mask[np.nonzero(m == 255)] = 1
                    else:
                        heatmap[np.nonzero(m == 255)] = 1

                #mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=5).astype(mask.dtype)
                # TODO mask too big and context inpainting color weired
                # Fixed colore weried, np.float32 will aotuomatically normalized inf 0~1
                mask = np.dstack((mask, mask, mask))
                img_instace = img * mask + inpainting * (1 - mask)

                scipy.misc.imsave('../../dataset/Relative/valid/relative_high_image/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), img_instace)
                scipy.misc.imsave('../../dataset/Relative/valid/relative_high_heatmap/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), heatmap)


def combine_two_inpaint():
    data_set_dir = '/data/vllab1/dataset/Relative/train/relative_high_image'
    data = glob(os.path.join(data_set_dir, "*.png"))
    data_len = len(data)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = data[i].split('/')[-1]
        name = name.split('.')[0]

        print(name)
        img = scipy.misc.imread(data[i])
        scipy.misc.imsave(os.path.join(
            '/data/vllab1/dataset/Relative/train/relative_combine_image', name + '_high.png'
        ), img)

        #break


def low_resolution():
    data_set_dir = '/data/vllab1/dataset/Relative/valid/relative_combine_image'
    data = glob(os.path.join(data_set_dir, "*.png"))
    data_len = len(data)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = data[i].split('/')[-1]
        img = scipy.misc.imread(data[i])
        img = scipy.misc.imresize(img, 0.5, interp='bilinear', mode=None)
        scipy.misc.imsave(os.path.join(
            '/data/vllab1/dataset/Relative/valid/relative_combine_low_image', name), img)


def valid_sigmoid():
    dir = '/data/vllab1/combine'
    image_name = sorted(glob(os.path.join(dir, "*.png")))
    deta_len = len(image_name)

    for i in range(0, deta_len):
        name = image_name[i].split('/')[-1]
        image = scipy.misc.imread(image_name[i])
        image = scipy.misc.imresize(image, 2., interp='bilinear', mode=None)
        image[np.nonzero(image < 10)] = 0
        image[np.nonzero(image != 0)] = 255
        scipy.misc.imsave(os.path.join('/data/vllab1/combine_treshold', name), image)


def hyper_visual():

    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/1_1'
    h1_1 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/1_5'
    h1_5 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/1_10'
    h1_10 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/3_1'
    h3_1 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/3_5'
    h3_5 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/3_10'
    h3_10 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/5_1'
    h5_1 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/5_5'
    h5_5 = sorted(glob(os.path.join(dir, "*.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/final/5_10'
    h5_10 = sorted(glob(os.path.join(dir, "*.png")))
    data_len = len(h1_1)
    for i in range(0, data_len):
        name = h1_1[i].split('/')[-1]
        img1_1 = scipy.misc.imread(h1_1[i])
        img1_5 = scipy.misc.imread(h1_5[i])
        img1_10 = scipy.misc.imread(h1_10[i])
        img3_1 = scipy.misc.imread(h3_1[i])
        img3_5 = scipy.misc.imread(h3_5[i])
        img3_10 = scipy.misc.imread(h3_10[i])
        img5_1 = scipy.misc.imread(h5_1[i])
        img5_5 = scipy.misc.imread(h5_5[i])
        img5_10 = scipy.misc.imread(h5_10[i])
        yo = np.array([img1_1, img1_5, img1_10, img3_1, img3_5, img3_10, img5_1, img5_5, img5_10])
        scipy.misc.imsave('/data/vllab1/Github/streetview_synthesize/ICCV/hyper_final_visual/{}'.format(name),
                          merge(yo, (3, 3)))
        #break


def real_test():
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/selected'
    sel = sorted(glob(os.path.join(dir, "*.png")))
    np.random.shuffle(sel)
    data_len = len(sel)
    ans_correct_num = 0
    sim_correct_num = 0
    ran_correct_num = 0
    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))
        name_split = sel[i].split('/')[-1].split('_')
        name = name_split[0] + '_' + name_split[1] + '_' + name_split[2]
        similiar = sorted(glob(os.path.join('/data/vllab1/Github/streetview_synthesize/ICCV/wrong_similiar', "{}*.png".format(name))))
        random = sorted(
            glob(os.path.join('/data/vllab1/Github/streetview_synthesize/ICCV/wrong_random', "{}*.png".format(name))))

        syn = scipy.misc.imread(sel[i])
        sim = scipy.misc.imread(similiar[0])
        ran = scipy.misc.imread(random[0])
        TEST = [syn, sim, ran]
        TEST_QUES = [0, 1, 2]
        np.random.shuffle(TEST_QUES)
        for j in range(1,4):
            if TEST_QUES[j-1] == 0:
                TEST_ANS = j
            elif TEST_QUES[j-1] == 1:
                TEST_SIM = j
            elif TEST_QUES[j-1] == 2:
                TEST_RAN = j
        visual = np.hstack((TEST[TEST_QUES[0]], TEST[TEST_QUES[1]], TEST[TEST_QUES[2]]))
        scipy.misc.imshow(visual)
        number = input('Your ANS: ')
        if number == TEST_ANS:
            ans_correct_num +=1
        elif number == TEST_SIM:
            sim_correct_num +=1
        elif number == TEST_RAN:
            ran_correct_num +=1


    print('syn score: {:d}/{:d}'.format(ans_correct_num, data_len))
    print('sim score: {:d}/{:d}'.format(sim_correct_num, data_len))
    print('ran score: {:d}/{:d}'.format(ran_correct_num, data_len))


def move_inpainting_visual():
    data = glob(os.path.join('/data/vllab1/Github/streetview_synthesize/Context Encoders_Feature Learning by Inpainting/PoissonGaussSeidel', "*.png"))

    # data_length = len(data)
    # high < 256 because want to cut the bottom
    # offs_h = np.random.randint(low=0, high=200, size=data_length)
    # offs_h_end = offs_h + 256
    # offs_w = np.random.randint(low=0, high=512, size=data_length)
    # offs_w_end = offs_w + 512
    # print offs_h, offs_h_end

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(data[index])
        name = data[index].split('/')[-1].split('_')
        out_name = '{}_{}_{}_poisson_{}'.format(name[0], name[1], name[2], name[3])

        scipy.misc.imsave(os.path.join('/data/vllab1/Github/streetview_synthesize/Context Encoders_Feature Learning by Inpainting/PoissonGaussSeidel_visual', out_name), img)
        #break


def move_train_valid_test():
    # Ref select_human_img_2

    data_set_dir = '../../dataset/CITYSCAPES/CITY_valid/human_image'
    data = sorted(glob(os.path.join(data_set_dir, "*.png")))

    for index, filePath in enumerate(data):
        print ('{}/{}'.format(index, len(data)))
        name = filePath.split('/')[-1].split('.')[0]
        name_test = name.split('_')
        mask_name = '{}_{}_{}_gtFine_labelIds.png'.format(name_test[0], name_test[1], name_test[2])
        img = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/CITY_valid/fine_mask', mask_name)).astype(np.uint8)
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY_valid/human_mask/' + mask_name, img.astype(np.uint8))
        #break


def inpainting_style():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)

    for index in range(0, length):
        name = human_file_name[index]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]

        # astype(np.float32) somehow do the normalize
        # should use if the inpainting method is context
        image = scipy.misc.imread(os.path.join(
            dataset_dir, 'fine_image', image_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join(
            dataset_dir, 'fine_mask', mask_name)).astype(np.float32) / 255
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))

        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        objects = label_instance['objects']
        image_style = np.copy(image)
        instance_obj = []
        for obj in objects:
            if obj['label'] == 'person':
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                human_pixel = np.nonzero(mask > 0)
                miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(
                    human_pixel[1])
                ori_heigh, ori_width = maxy - miny, maxx - minx
                size = ori_heigh * ori_width

                instance_pos = {'size': size, 'minx': minx, 'miny': miny, 'ori_heigh': ori_heigh, 'ori_width': ori_width}
                instance_obj.append(instance_pos)
                #mask[human_pixel] = 1
                #mask_image = image * np.dstack((mask, mask, mask))

        newlist = sorted(instance_obj, key=lambda k: k['size'])
        for instance in newlist:
            minx, miny = instance['minx'], instance['miny']
            ori_heigh, ori_width = instance['ori_heigh'], instance['ori_width']
            range_y = (np.arange(256) + miny) % 256
            range_x = (np.arange(512) + minx - ori_width) % 512
            yooo = True
            for minyy in range_y:
                for minxx in range_x:
                    if minyy + ori_heigh > 256 or minxx + ori_width > 512:
                        continue
                    if np.sum(mask_ori[minyy:minyy+ori_heigh, minxx:minxx+ ori_width]) == 0:
                        image_style[miny:miny+ori_heigh, minx:minx+ori_width, :] = image[minyy:minyy+ori_heigh, minxx:minxx+ ori_width]
                        yooo = False
                        break
            if(yooo):
                print('noooooo!!')

        scipy.misc.imsave('style/{}'.format(image_name), image_style)
        #break


def create_instance_for_pos():
    alpha = 0.01
    dataset_dir = '../../dataset/CITYSCAPES/CITY_valid'
    file_obj = open(os.path.join(dataset_dir, 'human_w.pkl'), 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)

    instance_multi_view_pose = {}
    for index in range(0, length):
        name = human_file_name[index]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]

        # astype(np.float32) somehow do the normalize
        # should use if the inpainting method is context
        image = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/val', city_name, image_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/val', city_name, image_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val',
                                 city_name, instance_name))

        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        instance_single_view_pose = []
        for obj in objects:
            if obj['label'] == 'person':
                #print(object['polygon'])
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                human_pixel = np.nonzero(mask > 0)
                miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(
                    human_pixel[1])
                ori_heigh, ori_width = maxy - miny, maxx - minx

                mask[human_pixel] = 1
                mask_image = image * np.dstack((mask, mask, mask))
                #mask_instance.append(mask)

                scale =  float(256)/max(ori_heigh, ori_width)
                resize_image = scipy.misc.imresize(mask_image[miny:maxy, minx:maxx, :], scale)
                backgorund = np.zeros((256, 256, 3), dtype=np.float32)
                heigh, width = np.shape(resize_image)[0], np.shape(resize_image)[1]
                margin_V = (256 - heigh) / 2
                margin_H = (256 - width) / 2
                backgorund[margin_V:margin_V+heigh, margin_H:margin_H+width, :] = resize_image

                #human_pixel = np.nonzero(mask == 255)
                #human_pixel_set.append(len(human_pixel[0]))
                # human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                # if human_ratio > alpha:
                #     mask_instance_sel.append(instance_num)
                scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY_valid/pose/{}_{:d}.png'.format(name, instance_num), backgorund)
                instance_pos = {'num': instance_num, 'minx': minx, 'miny': miny, 'heigh': heigh, 'width':width,
                                'scale': scale, 'margin_V': margin_V, 'margin_H': margin_H}
                instance_single_view_pose.append(instance_pos)
                instance_num += 1

        instance_multi_view_pose[name] = instance_single_view_pose
        #break

    file_obj = open('instance_multi_view_pose.pkl', 'wb')
    pickle.dump(instance_multi_view_pose, file_obj)
    file_obj.close()


def re_permuate_h5():
    data = []
    file = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'imgname.txt'), 'r')
    for line in file:
        data.append(line)

    # Get pos prediction
    filename = 'final_preds.h5'
    f = h5py.File(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', filename), 'r')
    preds = f['heatmaps']

    # Get instance_multi_view_pose
    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)

    order = np.zeros(811)
    cur = 0
    for single_view_name_idx, single_view_name in enumerate(sorted(instance_multi_view_pose)):
        print('{:d}/{:d} [{:d}]'.format(single_view_name_idx, length, cur))
        # Parse each pedestrian
        single_view = instance_multi_view_pose[single_view_name]
        cur_size = len(single_view)
        for cur_idx in range(0, cur_size):
            # Find the correct order match
            instance_idx = cur + cur_idx
            instance_name = data[instance_idx]
            instance_name = instance_name.split('.')[0]
            instance_name_idx = int(instance_name.split('_')[-1])
            order[cur + instance_name_idx] = instance_idx
        # Next view
        cur += cur_size

    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'order.pkl'), 'wb')
    pickle.dump(order, file_obj)
    file_obj.close()


def create_datasetPed_cityscape():
    dataset_dir = '../../dataset/CITYSCAPES/CITY_valid'

    # Get image name order
    # Get image name order
    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'order.pkl'), 'r')
    order = pickle.load(file_obj)

    # Get instance_multi_view_pose
    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)

    # Get pos prediction
    filename = 'final_preds.h5'
    f = h5py.File(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', filename), 'r')
    preds = f['heatmaps']
    preds_coordinate = f['preds']

    # Panomata
    PedMeta_all = []
    # Parse each view
    cur = 0
    for single_view_name_idx, single_view_name in enumerate(sorted(instance_multi_view_pose)):
        print('{:d}/{:d} [{:d}]'.format(single_view_name_idx, length, cur))
        # Get all needed files name
        image_name = '{}_leftImg8bit.png'.format(single_view_name)
        inpainting_name = '{}_poisson_leftImg8bit.png'.format(single_view_name)
        mask_name = '{}_gtFine_labelIds.png'.format(single_view_name)
        instance_name = '{}_gtFine_polygons.json'.format(single_view_name)
        city_name = single_view_name.split('_', 1)[0]
        single_view_image = scipy.misc.imread(os.path.join(
            '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/val', city_name, image_name)).astype(np.float32)

        # Get all needed files
        #single_view_image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name)).astype(np.float32)
        #mask = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name)).astype(np.float32)
        #inpainting = scipy.misc.imread(os.path.join(
        #    '/data/vllab1/Github/streetview_synthesize/Context Encoders_Feature Learning by Inpainting/python_blend', inpainting_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val',
                                 city_name, instance_name))
        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        # Pre calculate mask
        objects = label_instance['objects']
        mask_set = []
        size_set = []
        for obj in objects:
            if obj['label'] == 'person':
                polygon = [tuple(poly) for poly in obj['polygon']]
                # Get polygon mask
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                #img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                human_pixel = np.nonzero(mask > 0)
                miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(
                    human_pixel[1])
                ori_heigh, ori_width = maxy - miny, maxx - minx

                mask[human_pixel] = 1
                mask_set.append(mask[miny:maxy, minx:maxx])
                size_set.append([ori_heigh, ori_width])
                #mask_image = single_view_image * np.dstack((mask, mask, mask))

        # Parse each pedestrian
        single_view = instance_multi_view_pose[single_view_name]
        cur_size = len(single_view)
        for cur_idx in range(0, cur_size):
            # Find the correct order match
            order_idx = cur + cur_idx
            pred_idx = order[order_idx]

            #instance_name = data[instance_idx]
            #instance_name = instance_name.split('.')[0]
            #instance_name_idx = int(instance_name.split('_')[-1])
            # instance_name_idx = cur_idx
            #for instance in single_view:
            #    if instance['num'] == instance_name_idx:
            #        break

            # Get correct mask-image
            instance = single_view[cur_idx]
            instance_num, scale = instance['num'], instance['scale']
            print('{} {}'.format(pred_idx, instance_num))
            minx, miny = instance['minx'], instance['miny']
            heigh, width = instance['heigh'], instance['width']
            margin_V, margin_H = instance['margin_V'], instance['margin_H']

            ori_heigh, ori_width = size_set[instance_num]
            instance_image = single_view_image[miny:miny + ori_heigh, minx:minx + ori_width]
            mask = mask_set[instance_num]
            mask_image = instance_image * np.dstack((mask, mask, mask))
            mask *= 255

            # Coordinate visual
            #instance_image_visual = scipy.misc.imread(
            #    os.path.join('/data/vllab1/prediction', '{}_{:d}.png.jpg'.format(single_view_name, instance_num)))
            #instance_image_ori_visual = instance_image_visual[margin_V:margin_V + heigh, margin_H:margin_H + width]
            #instance_image_resize_visual = scipy.misc.imresize(instance_image_ori_visual, (ori_heigh, ori_width))

            # Heatmap visual
            instance_image_resize_visual_pose = np.copy(mask_image)

            heatmap_head = preds[pred_idx, 9]
            heatmap_head = scipy.misc.imresize(heatmap_head, 4.0, interp='bilinear', mode=None)
            heatmap_head_ori = heatmap_head[margin_V:margin_V + heigh, margin_H:margin_H + width]
            heatmap_head_ori_resize = scipy.misc.imresize(heatmap_head_ori, (ori_heigh, ori_width))
            instance_image_resize_visual_pose[:, :, 0] += heatmap_head_ori_resize

            heatmap_feet = preds[pred_idx, 0] + preds[pred_idx, 5]
            heatmap_feet = scipy.misc.imresize(heatmap_feet, 4.0, interp='bilinear', mode=None)
            heatmap_feet_ori = heatmap_feet[margin_V:margin_V + heigh, margin_H:margin_H + width]
            heatmap_feet_ori_resize = scipy.misc.imresize(heatmap_feet_ori, (ori_heigh, ori_width))
            instance_image_resize_visual_pose[:, :, 2] += heatmap_feet_ori_resize

            instance_image_resize_visual_pose[np.nonzero(instance_image_resize_visual_pose >= 255)] = 255

            out_name = '{}_{:d}.png'.format(single_view_name, instance_num)
            PedMeta = {'name': out_name, 'preds': preds[pred_idx], 'preds_coordinate': preds_coordinate[pred_idx]}
            PedMeta_all.append(PedMeta)
            scipy.misc.imsave(
                os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid/image',
                             out_name), mask_image.astype(np.uint8))
            scipy.misc.imsave(
                os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid/mask',
                             out_name), mask.astype(np.uint8))
            #scipy.misc.imsave(
            #    os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid/visual',
            #                 out_name), instance_image_resize_visual.astype(np.uint8))
            scipy.misc.imsave(
                os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid/visual',
                             '{}_{:d}_heatmap.png'.format(single_view_name, instance_num)), instance_image_resize_visual_pose.astype(np.uint8))
        # Next view
        cur += cur_size
        #break

    file_obj = open('/data/vllab1/CVPR/datasetPed/cityscape_valid/PedMeta.pkl', 'wb')
    pickle.dump(PedMeta_all, file_obj)
    file_obj.close()


def create_dataset_heatmap_pos_coordinate():
    dataset_dir = '../../dataset/CITYSCAPES/CITY_valid'

    # Get image name order
    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'order.pkl'), 'r')
    order = pickle.load(file_obj)

    # Get instance_multi_view_pose
    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)

    # Get pos prediction
    filename = 'final_preds.h5'
    f = h5py.File(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_valid', filename), 'r')
    preds = f['preds']

    # Parse each view
    cur = 0
    for single_view_name_idx, single_view_name in enumerate(sorted(instance_multi_view_pose)):
        # Count
        single_view = instance_multi_view_pose[single_view_name]
        cur_size = len(single_view)
        print('{:d}/{:d} [{:d}]'.format(single_view_name_idx, length, cur))

        # Get all needed files name
        image_name = '{}_leftImg8bit.png'.format(single_view_name)
        inpainting_name = '{}_leftImg8bit_poissopn.png'.format(single_view_name)
        mask_name = '{}_gtFine_labelIds.png'.format(single_view_name)
        instance_name = '{}_gtFine_polygons.json'.format(single_view_name)
        city_name = single_view_name.split('_', 1)[0]

        # Get all needed files
        single_view_image = scipy.misc.imread(os.path.join(dataset_dir, 'human_image', image_name)).astype(np.float32)
        mask = scipy.misc.imread(os.path.join(dataset_dir, 'human_mask', mask_name)).astype(np.float32)
        inpainting = scipy.misc.imread(os.path.join(
            '/data/vllab1/Github/streetview_synthesize/Context Encoders_Feature Learning by Inpainting/out_valid/out_blend', inpainting_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/val',
                                 city_name, instance_name))
        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        # Cacualte all instance
        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        for obj in objects:
            if obj['label'] == 'person':
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                mask[np.nonzero(mask > 0)] = 255
                mask_instance.append(mask)

                human_pixel = np.nonzero(mask == 255)
                human_pixel_set.append(len(human_pixel[0]))
                instance_num += 1

        # Sorting (meanwhile maintain original order)
        human_pixel_index = np.argsort(-np.array(human_pixel_set))
        mask_instance_sorted = []
        for sorted_index in human_pixel_index:
            mask_instance_sorted.append(mask_instance[sorted_index])
        print(instance_num)
        if instance_num > 20:
            # Select the up to 10 biggest instance
            instance_num = 20

        # Produce dataset pair [image, heatmap]
        instance_array = np.arange(instance_num)
        possible_c = np.arange(instance_num)
        np.random.shuffle(possible_c)
        if len(possible_c) > 10:
            # Cn0 + Cn1 + Cn2 + ... + Cnn-1
            # Random select up to 10 possible combinations
            possible_c = possible_c[0:10]
        for c_idx in possible_c:
            instance_combine = list(itertools.combinations(instance_array, c_idx))
            np.random.shuffle(instance_combine)
            if len(instance_combine) > 4:
                # Random select up to 4 combinations under one possible combinations. ex:
                # Cn3: (1, 2, 8), (2, 4, 6), (5, 18, 19), (3, 4, 5)
                instance_combine = instance_combine[0:4]
            for i_idx, combine_list in enumerate(instance_combine):
                img = np.copy(single_view_image)
                mask = np.zeros((256, 512), dtype=np.float32)
                heatmap = np.zeros((256, 512, 3), dtype=np.float32)
                for m_idx, m in enumerate(mask_instance_sorted):
                    if m_idx in combine_list:
                        mask[np.nonzero(m == 255)] = 1
                    else:
                        order_idx = cur + human_pixel_index[m_idx]
                        pred_idx = order[order_idx]

                        instance = single_view[human_pixel_index[m_idx]]
                        instance_num, scale = instance['num'], instance['scale']
                        minx, miny = int(np.round(instance['minx'] / 4.0)), int(np.round(instance['miny'] / 4.0))
                        heigh, width = instance['heigh'], instance['width']
                        margin_V, margin_H = instance['margin_V'], instance['margin_H']

                        # Get corrdinate
                        head = np.round(((preds[pred_idx, 9, 0:2] - [margin_H, margin_V]) / (scale * 4.0)) + [minx, miny])
                        foot_l = np.round(((preds[pred_idx, 0, 0:2] - [margin_H, margin_V]) / (scale * 4.0)) + [minx, miny])
                        foot_r = np.round(((preds[pred_idx, 5, 0:2] - [margin_H, margin_V]) / (scale * 4.0)) + [minx, miny])

                        if head[1] >= 256:
                            head[1] = 255
                        if head[0] >= 512:
                            head[0] = 511
                        if foot_l[1] >= 256:
                            foot_l[1] = 255
                        if foot_l[0] >= 512:
                            foot_l[0] = 511
                        if foot_r[1] >= 256:
                            foot_r[1] = 255
                        if foot_r[0] >= 512:
                            foot_r[0] = 511

                        pred_height = - head[1] + ((foot_l[1] + foot_r[1]) / 2)
                        sigma = 1 + (pred_height / 8)

                        # Gaussian blur
                        heatmap_head = np.zeros((256, 512), dtype=np.float32)
                        heatmap_head[head[1], head[0]] = 1
                        heatmap_head = gaussian_filter(heatmap_head, sigma=sigma)
                        heatmap_head /= np.max(heatmap_head)

                        heatmap_foot = np.zeros((256, 512), dtype=np.float32)
                        heatmap_foot[foot_l[1], foot_l[0]] = 1
                        heatmap_foot[foot_r[1], foot_r[0]] = 1
                        heatmap_foot = gaussian_filter(heatmap_foot, sigma=sigma)
                        heatmap_foot /= np.max(heatmap_foot)

                        # Generate heatmap
                        heatmap[np.nonzero(m == 255) + (1,)] = 1
                        heatmap[:, :, 0] += heatmap_head
                        heatmap[:, :, 2] += heatmap_foot

                # Fixed colore weried, np.float32 will aotuomatically normalized inf 0~1
                mask = np.dstack((mask, mask, mask))
                img_instace = img * mask + inpainting * (1 - mask)

                heatmap *= 255
                heatmap[np.nonzero(heatmap>255)] = 255
                scipy.misc.imsave('/data/vllab1/CVPR/dataset/valid/image/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), img_instace)
                scipy.misc.imsave('/data/vllab1/CVPR/dataset/valid/heatmap/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), heatmap)

        cur += cur_size
        #break


def create_dataset_heatmap_pos():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'

    # Get image name order
    file_obj = open(os.path.join('', 'order.pkl'), 'r')
    order = pickle.load(file_obj)

    # Get instance_multi_view_pose
    file_obj = open(os.path.join('', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)

    # Get pos prediction
    filename = 'final_preds.h5'
    f = h5py.File(filename, 'r')
    preds = f['preds']

    # Parse each view
    cur = 0
    for single_view_name_idx, single_view_name in enumerate(sorted(instance_multi_view_pose)):
        # Count
        single_view = instance_multi_view_pose[single_view_name]
        cur_size = len(single_view)
        print('{:d}/{:d} [{:d}]'.format(single_view_name_idx, length, cur))

        # Get all needed files name
        image_name = '{}_leftImg8bit.png'.format(single_view_name)
        inpainting_name = '{}_poisson_leftImg8bit.png'.format(single_view_name)
        mask_name = '{}_gtFine_labelIds.png'.format(single_view_name)
        instance_name = '{}_gtFine_polygons.json'.format(single_view_name)
        city_name = single_view_name.split('_', 1)[0]
        #single_view_image = scipy.misc.imread(os.path.join(
        #    '/data/vllab1/dataset/CITYSCAPES/leftImg8bit_trainvaltest/leftImg8bit/train', city_name, image_name)).astype(np.float32)
        #single_view_image_v = np.copy(single_view_image)
        #single_view_heatmp = np.zeros((1024, 2048), dtype=np.float32)

        # Get all needed files
        single_view_image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name)).astype(np.float32)
        mask = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name)).astype(np.float32)
        inpainting = scipy.misc.imread(os.path.join(
            '/data/vllab1/Github/streetview_synthesize/Context Encoders_Feature Learning by Inpainting/python_blend', inpainting_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))
        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        # Cacualte all instance
        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        for obj in objects:
            if obj['label'] == 'person':
                #print(object['polygon'])
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                mask[np.nonzero(mask > 0)] = 255
                mask_instance.append(mask)

                human_pixel = np.nonzero(mask == 255)
                human_pixel_set.append(len(human_pixel[0]))
                # human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                # if human_ratio > alpha:
                #     mask_instance_sel.append(instance_num)
                instance_num += 1

        # Sorting (meanwhile maintain original order)
        human_pixel_index = np.argsort(human_pixel_set)
        mask_instance_sorted = []
        for sorted_index in human_pixel_index:
            mask_instance_sorted.append(mask_instance[sorted_index])
        print(instance_num)
        if instance_num > 20:
            # Select the up to 10 biggest instance
            instance_num = 20

        # Produce dataset pair [image, heatmap]
        instance_array = np.arange(instance_num)
        possible_c = np.arange(instance_num)
        np.random.shuffle(possible_c)
        if len(possible_c) > 10:
            # Cn0 + Cn1 + Cn2 + ... + Cnn-1
            # Random select up to 10 possible combinations
            possible_c = possible_c[0:10]
        for c_idx in possible_c:
            instance_combine = list(itertools.combinations(instance_array, c_idx))
            np.random.shuffle(instance_combine)
            if len(instance_combine) > 4:
                # Random select up to 4 combinations under one possible combinations. ex:
                # Cn3: (1, 2, 8), (2, 4, 6), (5, 18, 19), (3, 4, 5)
                instance_combine = instance_combine[0:4]
            for i_idx, combine_list in enumerate(instance_combine):
                img = np.copy(single_view_image)
                mask = np.zeros((256, 512), dtype=np.uint8)
                heatmap = np.zeros((256, 512, 3), dtype=np.uint8)
                heatmap_head = np.zeros((256, 512), dtype=np.float32)
                for m_idx, m in enumerate(mask_instance_sorted):
                    if m_idx in combine_list:
                        mask[np.nonzero(m == 255)] = 1
                    else:
                        order_idx = cur + human_pixel_index[m_idx]
                        pred_idx = order[order_idx]

                        instance = single_view[human_pixel_index[m_idx]]
                        instance_num, scale = instance['num'], instance['scale']
                        minx, miny = int(instance['minx'] / 4.0), int(instance['miny'] / 4.0)
                        heigh, width = instance['heigh'], instance['width']
                        margin_V, margin_H = instance['margin_V'], instance['margin_H']


                        head = np.round(((preds[pred_idx, 9, 0:2] - [margin_H, margin_V]) / (scale * 4.0)) + [minx, miny])
                        foot_l = np.round(((preds[pred_idx, 0, 0:2] - [margin_H, margin_V]) / (scale * 4.0)) + [minx, miny])
                        foot_r = np.round(((preds[pred_idx, 5, 0:2] - [margin_H, margin_V]) / (scale * 4.0)) + [minx, miny])

                        pred_height = - head[1] + ((foot_l[1] + foot_r[1]) / 2)
                        iterations = 1 + (pred_height / 10)

                        #heatmap_head_pred = preds[pred_idx, 9]
                        #heatmap_head = scipy.misc.imresize(heatmap_head_pred, 4.0, interp='bilinear', mode=None)
                        #heatmap_head_ori = heatmap_head[margin_V:margin_V + heigh, margin_H:margin_H + width]
                        #heatmap_head_ori_resize = scipy.misc.imresize(heatmap_head_ori, (1.0 / (scale * 4.0)))
                        #heigh, width = np.shape(heatmap_head_ori_resize)[0], np.shape(heatmap_head_ori_resize)[1]
                        #heatmap[miny:miny+heigh, minx:minx+width, 1] += heatmap_head_ori_resize

                        #m = np.dstack((m, m, m))
                        #yo = np.nonzero(m == 255)
                        heatmap_head[int(head[1]), (head[0])] = 1
                        heatmap[np.nonzero(m == 255) + (0,)] = 180

                heatmap_head = scipy.ndimage.morphology.binary_dilation(heatmap_head, iterations=5).astype(heatmap_head.dtype)
                heatmap_head = gaussian_filter(heatmap_head * 255, sigma=3)
                heatmap[:, :, 0] = heatmap_head

                #mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=5).astype(mask.dtype)
                # TODO mask too big and context inpainting color weired
                # Fixed colore weried, np.float32 will aotuomatically normalized inf 0~1
                mask = np.dstack((mask, mask, mask))
                img_instace = img * mask + inpainting * (1 - mask)

                scipy.misc.imsave('/data/vllab1/CVPR/dataset/train/image/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), img_instace)
                scipy.misc.imsave('/data/vllab1/CVPR/dataset/train/heatmap/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), heatmap)

        cur += cur_size
        break


def create_dataset_heatmap_pos_mask():
    dataset_dir = '../../dataset/CITYSCAPES/CITY'

    # Get image name order
    file_obj = open(os.path.join('', 'order.pkl'), 'r')
    order = pickle.load(file_obj)

    # Get instance_multi_view_pose
    file_obj = open(os.path.join('', 'instance_multi_view_pose.pkl'), 'r')
    instance_multi_view_pose = pickle.load(file_obj)
    length = len(instance_multi_view_pose)

    # Get pos prediction
    filename = 'final_preds.h5'
    f = h5py.File(filename, 'r')
    preds = f['heatmaps']

    # Parse each view
    cur = 0
    for single_view_name_idx, single_view_name in enumerate(sorted(instance_multi_view_pose)):
        # Count
        single_view = instance_multi_view_pose[single_view_name]
        cur_size = len(single_view)
        print('{:d}/{:d} [{:d}]'.format(single_view_name_idx, length, cur))

        # Get all needed files name
        image_name = '{}_leftImg8bit.png'.format(single_view_name)
        inpainting_name = '{}_poisson_leftImg8bit.png'.format(single_view_name)
        mask_name = '{}_gtFine_labelIds.png'.format(single_view_name)
        instance_name = '{}_gtFine_polygons.json'.format(single_view_name)
        city_name = single_view_name.split('_', 1)[0]

        # Get all needed files
        single_view_image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name)).astype(np.float32)
        mask = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name)).astype(np.float32)
        inpainting = scipy.misc.imread(os.path.join(
            '/data/vllab1/Github/streetview_synthesize/Context Encoders_Feature Learning by Inpainting/python_blend', inpainting_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))
        with open(instance) as data_file:
            label_instance = json.load(data_file)
            data_file.close()

        # Cacualte all instance
        objects = label_instance['objects']
        instance_num = 0
        mask_instance = []
        human_pixel_set = []
        for obj in objects:
            if obj['label'] == 'person':
                polygon = [tuple(poly) for poly in obj['polygon']]
                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                mask[np.nonzero(mask > 0)] = 255
                mask_instance.append(mask)

                human_pixel = np.nonzero(mask == 255)
                human_pixel_set.append(len(human_pixel[0]))
                instance_num += 1

        # Sorting (meanwhile maintain original order)
        human_pixel_index = np.argsort(human_pixel_set)
        mask_instance_sorted = []
        for sorted_index in human_pixel_index:
            mask_instance_sorted.append(mask_instance[sorted_index])
        print(instance_num)
        if instance_num > 20:
            # Select the up to 10 biggest instance
            instance_num = 20

        # Produce dataset pair [image, heatmap]
        instance_array = np.arange(instance_num)
        possible_c = np.arange(instance_num)
        np.random.shuffle(possible_c)
        if len(possible_c) > 10:
            # Cn0 + Cn1 + Cn2 + ... + Cnn-1
            # Random select up to 10 possible combinations
            possible_c = possible_c[0:10]
        for c_idx in possible_c:
            instance_combine = list(itertools.combinations(instance_array, c_idx))
            np.random.shuffle(instance_combine)
            if len(instance_combine) > 4:
                # Random select up to 4 combinations under one possible combinations. ex:
                # Cn3: (1, 2, 8), (2, 4, 6), (5, 18, 19), (3, 4, 5)
                instance_combine = instance_combine[0:4]
            for i_idx, combine_list in enumerate(instance_combine):
                img = np.copy(single_view_image)
                mask = np.zeros((256, 512), dtype=np.float32)
                heatmap = np.zeros((256, 512, 3), dtype=np.float32)
                heatmap_head = np.zeros((256, 512), dtype=np.float32)
                for m_idx, m in enumerate(mask_instance_sorted):
                    if m_idx in combine_list:
                        mask[np.nonzero(m == 255)] = 1
                    else:
                        # Get index
                        order_idx = cur + human_pixel_index[m_idx]
                        pred_idx = order[order_idx]
                        # Get instance
                        instance = single_view[human_pixel_index[m_idx]]
                        instance_num, scale = instance['num'], instance['scale']
                        minx, miny = int(instance['minx'] / 4.0), int(instance['miny'] / 4.0)
                        heigh, width = instance['heigh'], instance['width']
                        margin_V, margin_H = instance['margin_V'], instance['margin_H']
                        instance_mask = scipy.misc.imread(
                            os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_train/mask', '{}_{}.png'.format(single_view_name, instance_num)))
                        instance_mask = scipy.misc.imresize(instance_mask, 0.25)
                        heigh_ori, width_ori = np.shape(instance_mask)[0], np.shape(instance_mask)[1]

                        # Get prediction heatmap
                        heatmap_head_pred = preds[pred_idx, 9]
                        heatmap_max = np.max(heatmap_head_pred)
                        heatmap_head_pred[np.nonzero(heatmap_head_pred) < 0] = 0
                        heatmap_head_pred = heatmap_head_pred / 255.0 * heatmap_max

                        heatmap_head = scipy.misc.imresize(heatmap_head_pred, 4.0, interp='bilinear', mode=None)
                        heatmap_bilinear_max = float(np.max(heatmap_head))
                        heatmap_head_ori = heatmap_head[margin_V:margin_V + heigh, margin_H:margin_H + width]
                        heatmap_head_ori_resize = scipy.misc.imresize(heatmap_head_ori, (heigh_ori, width_ori))
                        # Crop and restore mask
                        heatmap_head_ori_resize *= (instance_mask / 255)
                        heatmap_head_ori_resize = (heatmap_head_ori_resize / 255.0) * 1

                        #heigh, width = np.shape(heatmap_head_ori_resize)[0], np.shape(heatmap_head_ori_resize)[1]

                        heatmap[miny:miny + heigh_ori, minx:minx + width_ori, 1] += heatmap_head_ori_resize
                        heatmap[np.nonzero(m == 255) + (0,)] = 1

                heatmap *= 255
                heatmap[np.nonzero(heatmap > 255)] = 255

                mask = np.dstack((mask, mask, mask))
                img_instace = img * mask + inpainting * (1 - mask)

                scipy.misc.imsave('/data/vllab1/CVPR/dataset/train/image/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), img_instace)
                scipy.misc.imsave('/data/vllab1/CVPR/dataset/train/heatmap/{}_{:d}_{:d}.png'.format(
                    image_name.split('.')[0], c_idx, i_idx), heatmap)

        cur += cur_size
        break


def move_and_resize_dataset():
    data = glob(os.path.join('/data/vllab1/CVPR/dataset/copy/test/image', "*.png"))

    for index, filePath in enumerate(data):
        print ('%d/%d' % (index, len(data)))
        img = scipy.misc.imread(filePath).astype(np.uint8)
        img = scipy.misc.imresize(img, 0.5, interp='bilinear', mode=None)
        scipy.misc.imsave('/data/vllab1/CVPR/dataset/test/image/' + filePath.split('/')[-1], img.astype(np.uint8))
        #break


def search_datasetPed():

    file_obj = open(os.path.join('/data/vllab1/CVPR/datasetPed/cityscape_train', 'PedMeta.pkl'), 'r')
    PedMeta = pickle.load(file_obj)


def heatmap_suppression():
    dir = '/data/vllab1/Github/streetview_synthesize/FCN/test_visual'
    data = glob(os.path.join(dir, "*_pred.png"))
    data_len = len(data)

    for img_idx, img_name in enumerate(data):
        name = img_name.split('/')[-1].split('.')[0]
        name = name +'.png'
        print('{:d}/{:d} {}'.format(img_idx, data_len, name))
        pred = scipy.misc.imresize(scipy.misc.imread(img_name), 2.0).astype(np.float32)
        image = scipy.misc.imread(os.path.join('/data/vllab1/CVPR/dataset/copy/test/image', name)).astype(np.float32)

        thre = 1
        head = np.copy(pred[:, :, 0])
        #head = gaussian_filter(head, sigma=1)
        head[np.nonzero[head] < thre] = 0
        map_aug = np.zeros((256 + 2, 512 + 2))
        map_aug1 = np.zeros((256 + 2, 512 + 2))
        map_aug2 = np.zeros((256 + 2, 512 + 2))
        map_aug3 = np.zeros((256 + 2, 512 + 2))
        map_aug4 = np.zeros((256 + 2, 512 + 2))

        map_aug[1:-1, 1:-1] = head
        map_aug1[1:-1, 0:-2] = head
        map_aug2[1:-1, 2:] = head
        map_aug3[0:-2, 1:-1] = head
        map_aug4[2:, 1:-1] = head

        peakMap = (map_aug < map_aug1) & (map_aug > map_aug2) & (map_aug > map_aug3) & (map_aug > map_aug4)
        peakMap = peakMap[1:-1, 1:-1]



        peakMap = scipy.ndimage.morphology.binary_dilation(peakMap, iterations=1).astype(peakMap.dtype)
        head_idx = np.nonzero(peakMap)
        #[X, Y] = find(peakMap);

        #feet = np.copy(pred[:, :, 2])
        #head_idx = np.nonzero(head>1)

        visual = np.copy(image)
        visual[head_idx + (0,)] = 255
        visual[head_idx + (1,)] = 255
        visual[head_idx + (2,)] = 255

        visual[:, :, 0] += pred[:, :, 0] * 100
        visual[np.nonzero(visual>255)] = 255


        scipy.misc.imsave(os.path.join('/data/vllab1/CVPR/demo/heatmap_suppression', name), visual.astype(np.uint8))

        if img_idx > 10:
            break


def find_peak(heatmap, thre=1):
    heatmap[np.nonzero(heatmap < thre)] = 0
    height, width = np.shape(heatmap)[0], np.shape(heatmap)[1]
    map_aug = np.zeros((height + 2, width + 2))
    map_aug1 = np.zeros((height + 2, width + 2))
    map_aug2 = np.zeros((height + 2, width + 2))
    map_aug3 = np.zeros((height + 2, width + 2))
    map_aug4 = np.zeros((height + 2, width + 2))

    map_aug[1:-1, 1:-1] = heatmap
    map_aug1[1:-1, 0:-2] = heatmap
    map_aug2[1:-1, 2:] = heatmap
    map_aug3[0:-2, 1:-1] = heatmap
    map_aug4[2:, 1:-1] = heatmap

    peakMap = (map_aug < map_aug1) & (map_aug > map_aug2) & (map_aug > map_aug3) & (map_aug > map_aug4)
    peakMap = peakMap[1:-1, 1:-1]
    peakMap_idx = np.nonzero(peakMap)
    peakMap_idx = np.array(zip(*peakMap_idx))
    if len(peakMap_idx) == 0:
        return peakMap_idx

    deleIdx = []
    flag = np.ones((len(peakMap_idx)))
    for i in range(0, len(peakMap_idx)):
        if flag[i] > 0:
            for j in range(i+1, len(peakMap_idx)):
                v = peakMap_idx[i] - peakMap_idx[j]
                if np.linalg.norm(v) <= 10:
                    flag[j] = 0
                    deleIdx.append(j)

    peakMap_idx_y = np.delete(peakMap_idx[:, 0], deleIdx)
    peakMap_idx_x = np.delete(peakMap_idx[:, 1], deleIdx)
    peakMap_idx = zip(peakMap_idx_y, peakMap_idx_x)
    return peakMap_idx


def simpleblend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    img_mask = gaussian_filter(img_mask, sigma=1)
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask3 = np.dstack((img_mask, img_mask, img_mask))

    img_source = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3]]

    img_target_region = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3]]
    img_target[region_target[0]:region_target[2], region_target[1]:region_target[3]] = img_source * img_mask3 + \
                                                                                       img_target_region * (1. - img_mask3)

    return img_target


def synthesize_with_pedcut(do_poisson=False):
    ped_dir = '/data/vllab1/dataset/pedestrian/Ped'

    file_name = os.path.join(ped_dir, 'pedestrian_list.pkl')
    file_obj = open(file_name, 'r')
    ped_file_name = pickle.load(file_obj)

    dir = '/data/vllab1/Github/streetview_synthesize/FCN/test_visual'
    data = glob(os.path.join(dir, "*_pred.png"))
    data_len = len(data)

    for img_idx, img_name in enumerate(data):
        name = img_name.split('/')[-1].split('.')[0]
        name = name +'.png'
        print('{:d}/{:d} {}'.format(img_idx, data_len, name))
        pred = scipy.misc.imresize(scipy.misc.imread(img_name), 2.0).astype(np.float32)
        image = scipy.misc.imread(os.path.join('/data/vllab1/CVPR/dataset/copy/test/image', name)).astype(np.float32)
        visual = np.copy(image)

        # Surpression
        thre = 1
        peakMap_idx_head = find_peak(pred[:, :, 0], thre)
        peakMap_idx_body = find_peak(pred[:, :, 1], thre)
        peakMap_idx_foot = find_peak(pred[:, :, 2], thre)

        #if len(peakMap_idx_head) == 0 or len(peakMap_idx_body) == 0 or len(peakMap_idx_foot) == 0:
        #    continue
        # Candidate Position
        peakMap_idx = peakMap_idx_head

        # Search most likely
        for idx in peakMap_idx:
            # Search most likely
            ped_sel_idx = np.random.randint(785)
            ped_name, ped_height, ped_width, ped_aspect = ped_file_name[ped_sel_idx].name, ped_file_name[
                ped_sel_idx].height, ped_file_name[ped_sel_idx].width, ped_file_name[ped_sel_idx].aspect
            ped_image = scipy.misc.imread(os.path.join(ped_dir, 'image', ped_name)).astype(np.float32)
            ped_mask = scipy.misc.imread(os.path.join(ped_dir, 'mask', ped_name)).astype(np.float32) / 255.

            # Blending
            if do_poisson:
                ped_mask = scipy.ndimage.morphology.binary_dilation(ped_mask, iterations=5).astype(ped_mask.dtype)
                ped_mask[:, 0] = 0
                ped_mask[:, -1] = 0
                ped_mask[0, :] = 0
                ped_mask[-1, :] = 0
                # visual = poissonblending.blend(visual, ped_image, ped_mask, offset=(0, 0))
                visual = poissonblending.blend(visual, ped_image, ped_mask, idx)
            else:
                yo = idx - np.array([ped_height/2., ped_width/2.])
                visual = simpleblend(visual, ped_image, ped_mask, idx )

        scipy.misc.imsave(os.path.join('/data/vllab1/CVPR/demo/synthesize_pedcut_body', name), visual.astype(np.uint8))

        #if img_idx > 10:
        #    break


def synthesize_with_pedcut_segment(do_poisson=False):
    ped_dir = '/data/vllab1/dataset/pedestrian/Ped'

    file_name = os.path.join(ped_dir, 'pedestrian_list.pkl')
    file_obj = open(file_name, 'r')
    ped_file_name = pickle.load(file_obj)

    dir = '/data/vllab1/Github/streetview_synthesize/FCN/test_visual'
    data = glob(os.path.join(dir, "*_pred.png"))
    data_len = len(data)

    # pedestrian image, mask, list
    ped_single_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_new/image'
    mask_single_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_new/mask'
    dataset_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_new'
    file_name = os.path.join(dataset_dir, 'pedestrian_list.pkl')
    file_obj = open(file_name, 'r')
    pedestrian_list = pickle.load(file_obj)
    # pedestrian_multi image, mask, list
    ped_mul_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_multiple/image'
    mask_mul_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_multiple/mask'
    dataset_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES_multiple'
    file_name = os.path.join(dataset_dir, 'pedestrian_list.pkl')
    file_obj = open(file_name, 'r')
    pedestrian_multi_list = pickle.load(file_obj)

    # parameter
    sig, threshold = 1, 1
    print(sig, threshold)
    for img_idx, img_name in enumerate(data):
        name = img_name.split('/')[-1].split('.')[0]
        name = name +'.png'
        print('{:d}/{:d} {}'.format(img_idx, data_len, name))
        pred = scipy.misc.imresize(scipy.misc.imread(img_name), 2.0).astype(np.float32)
        image = scipy.misc.imread(os.path.join('/data/vllab1/CVPR/dataset/copy/test/image', name)).astype(np.float32)
        visual = np.copy(image)

        # Segment heatmap
        blurred = gaussian_filter(pred[:, :, 1], sigma=sig)
        blurred[np.nonzero(blurred >= threshold)] = 255
        blurred[np.nonzero(blurred < threshold)] = 0
        # Visual segment heatmap
        blurred_index = np.nonzero(blurred == 255)
        image_v = np.copy(image)
        image_v[blurred_index + (0,)] += 80
        image_v[blurred_index + (1,)] += 80
        image_v[blurred_index + (2,)] += 80
        image_v[np.nonzero(image_v > 255)] = 255

        # Use segment heatmap seythesis image
        all_labels = skimage.measure.label(blurred, background=0)
        for label in range(1, all_labels.max()+1):
            human_pixel = np.nonzero(all_labels == label)
            miny, maxy, minx, maxx = min(human_pixel[0]), max(human_pixel[0]), min(human_pixel[1]), max(human_pixel[1])
            height, width = maxy - miny, maxx - minx
            # invalid region
            if height < 20 or width < 15:
                continue
            aspect = float(width) / float(height)
            if aspect > 0.7:
                image_v[human_pixel + (1,)] = 255
                pedestrian_fit_all = pedestrian_multi_list
                ped_dir = ped_mul_dir
                mask_dir = mask_mul_dir
                heat_alpha = 1
            else:
                image_v[human_pixel + (2,)] = 255
                pedestrian_fit_all = pedestrian_list
                ped_dir = ped_single_dir
                mask_dir = mask_single_dir
                heat_alpha = 255
            # Blending
            '''
            if do_poisson:
                ped_mask = scipy.ndimage.morphology.binary_dilation(ped_mask, iterations=5).astype(ped_mask.dtype)
                ped_mask[:, 0] = 0
                ped_mask[:, -1] = 0
                ped_mask[0, :] = 0
                ped_mask[-1, :] = 0
                # visual = poissonblending.blend(visual, ped_image, ped_mask, offset=(0, 0))
                visual = poissonblending.blend(visual, ped_image, ped_mask, idx)
            else:
                yo = idx - np.array([ped_height/2., ped_width/2.])
                visual = simpleblend(visual, ped_image, ped_mask, idx )
            '''
            ceny, cenx = np.round(np.mean(human_pixel[0])), np.round(np.mean(human_pixel[1]))
            pedestrian_fit_all.sort(key=lambda x: abs(x.width-width) + abs(x.height-height))
            #random_sel = np.random.randint(0, len(pedestrian_fit_all))

            sel_ped_name, ped_width, ped_height = pedestrian_fit_all[0].name, pedestrian_fit_all[0].width, pedestrian_fit_all[0].height
            sel_ped_image = scipy.misc.imread(os.path.join(ped_dir, sel_ped_name)).astype(np.float32)
            sel_ped_mask = scipy.misc.imread(os.path.join(mask_dir, sel_ped_name)).astype(np.float32)/heat_alpha
            sel_ped_mask = np.dstack((sel_ped_mask, sel_ped_mask, sel_ped_mask))
            sel_ped_image *= sel_ped_mask


            #road_pixel = np.nonzero(semantic == 7)
            #sidewalk_pixel = np.nonzero(semantic == 8)
            #parking_pixel = np.nonzero(semantic == 9)
            #rail_pixel = np.nonzero(semantic == 10)
            #possible_random = np.hstack((road_pixel, sidewalk_pixel, parking_pixel, rail_pixel))
            #ramdom_sel = np.random.randint(0, len(possible_random[0]))
            #ceny, cenx = possible_random[0][ramdom_sel] / 4, possible_random[1][ramdom_sel] / 4
            #ceny -= ped_height/2


            ped_width, ped_height =  int(ped_width), int(ped_height)
            cut_upy, cut_dny, cut_lx, cut_rx = 0, 0, 0, 0
            lx = int(np.round(cenx - ped_width / 2))
            if lx < 0:
                cut_lx = -lx
                lx = 0

            upy = int(np.round(ceny - ped_height/2))
            if upy < 0:
                cut_upy = -upy
                upy = 0

            rx = lx + ped_width - cut_lx
            if rx > 512-1:
                cut_rx = rx - (512-1)
                rx = 512-1

            dny = upy + ped_height - cut_upy
            if dny > 256-1:
                cut_dny = dny - (256-1)
                dny = 256-1

            hole = (1-sel_ped_mask)[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :] * visual[upy:dny, lx:rx, :]
            fill = hole + sel_ped_image[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :]
            visual[upy:dny, lx:rx, :] = fill

        scipy.misc.imsave(os.path.join('/data/vllab1/CVPR/demo/synthesize_city_fit', name), visual.astype(np.uint8))

        #if img_idx > 10:
        #    break



def CVPR_instance_visual():
    dir = '/data/vllab1/CVPR/dataset/copy/train/image'
    data = glob(os.path.join(dir, "strasbourg_000000_030706*.png"))
    data_len = len(data)

    for img_idx, img_name in enumerate(data):
        name = img_name.split('/')[-1]
        print('{:d}/{:d} {}'.format(img_idx, data_len, name))
        img = scipy.misc.imread(img_name).astype(np.float32)
        heatmap = scipy.misc.imread(os.path.join('/data/vllab1/CVPR/dataset/copy/train/heatmap', name)).astype(np.float32)

        visual = np.copy(img)
        visual = img + heatmap * 0.3
        #visual[np.nonzero(visual > 255)] = 255

        #body = np.copy(heatmap[:, :, 1])
        #body_idx = np.nonzero(body == 255)
        #visual[body_idx + (0,)] += 100
        #visual[body_idx + (1,)] += 100
        #visual[body_idx + (2,)] += 100

        #visual[:, :, 0] += heatmap[:, :, 0] * 0.6
        #visual[:, :, 2] += heatmap[:, :, 2] * 0.6
        visual[np.nonzero(visual>255)] = 255



        scipy.misc.imsave(os.path.join('/data/vllab1/CVPR/demo/dataset_visual', name), visual.astype(np.uint8))

        #break


def CVPR_FCN_enhance():
    dir = '/data/vllab1/Github/streetview_synthesize/FCN/test_visual_use_valid'
    data = glob(os.path.join(dir, "frankfurt_000000_009561_leftImg8bit.png_pred.png"))
    data_len = len(data)

    for img_idx, img_name in enumerate(data):
        name = img_name.split('/')[-1].split('.')[0]
        name = name +'.png'
        print('{:d}/{:d} {}'.format(img_idx, data_len, name))
        pred = scipy.misc.imresize(scipy.misc.imread(img_name), 2.0).astype(np.float32)
        #image = scipy.misc.imread(os.path.join('/data/vllab1/CVPR/dataset/copy/valid/image', name)).astype(np.float32)
        image = scipy.misc.imread(os.path.join('/data/vllab1/dataset/CITYSCAPES/CITY_valid/fine_image', name)).astype(np.float32)

        #pred_visual = np.copy(pred)
        #pred_visual *= 2
        #pred_visual = gaussian_filter(pred_visual, sigma=3)
        #pred_visual[np.nonzero(pred_visual > 255)] = 255


        visual = np.copy(image)
        visual = image + pred * 100
        visual[np.nonzero(visual>255)] = 255


        scipy.misc.imsave(os.path.join('/data/vllab1/CVPR/demo/FCN_enhance_valid', name), visual.astype(np.uint8))

        if img_idx > 10:
            break


def CVPR_syn():

    dir = '/data/vllab1/CVPR/demo/synthesize_city_fit'
    data = sorted(glob(os.path.join(dir, "*.png")))
    data_len = len(data)

    dir = '/data/vllab1/CVPR/demo/FCN_enhance'
    data2 = sorted(glob(os.path.join(dir, "*.png")))

    for img_idx, img_name in enumerate(data):
        name = img_name.split('/')[-1].split('.')[0]
        name = name +'.png'
        print('{:d}/{:d} {}'.format(img_idx, data_len, name))
        img = scipy.misc.imread(img_name).astype(np.float32)
        pred = scipy.misc.imread(data2[img_idx]).astype(np.float32)

        combine = np.zeros((256, 512*2, 3))
        combine[:, 0:512, :] = img
        combine[:, 512:, :] = pred


        scipy.misc.imsave(os.path.join('/data/vllab1/CVPR/demo/syn_visual', name), combine.astype(np.uint8))

        #if img_idx > 10:
        #    break

if __name__ == '__main__':
    store_single('/data/vllab1/CVPR/workshop/predict_training/7200_pred.png', 128, 256, 3, 2)
    #wrong_similiar_random()
    pass

