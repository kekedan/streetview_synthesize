from glob import glob
import json
import scipy.ndimage
from PIL import Image, ImageDraw
import pickle
import os
import itertools
from scipy.ndimage.filters import gaussian_filter
import skimage.measure

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
    scipy.misc.imsave(name + '_single.' + extension, sel_image)


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
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/coarse_image/' + filePath.split('/')[-1], img)


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
        scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/coarse_mask/' + filePath.split('/')[-1], img)


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


def create_mask_img():
    data = sorted(glob(os.path.join('../../dataset/CITYSCAPES/CITY/human_image', "*.png")))
    label = sorted(glob(os.path.join('../../dataset/CITYSCAPES/CITY/human_mask', "*.png")))

    length = len(data)
    for i in range(0, length):
        print ('%d/%d' % (i, length))
        #fileName = filePath.split('/')[-1].split('.')[0]
        img = scipy.misc.imread(data[i]).astype(np.float)
        label2 = scipy.misc.imread(label[i]).astype(np.int)

        indices = np.nonzero(label2 == 255)
        img[indices + (0,)] = 0
        img[indices + (1,)] = 255
        img[indices + (2,)] = 0
        scipy.misc.imsave('../../dataset/CITYSCAPES/CITY/human_mask_inpainting/' + data[i].split('/')[-1], img)
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
        # should use if the inpainting method is context
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
    human_no_extra = []

    data_set_dir = '../../dataset/CITYSCAPES/CITY/fine_mask'
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

    file_obj = open('human_extra.pkl', 'wb')
    pickle.dump(human_no_extra, file_obj)
    file_obj.close()

    print len(human_no_extra)


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
    dir = '/data/vllab1/Github/streetview_synthesize/ICCV/context'
    ped_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES/image'
    mask_dir = '/data/vllab1/dataset/pedestrian/CITYSCAPES/mask'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    pedestrian_name = sorted(glob(os.path.join(ped_dir, "*.png")))
    mask_name = sorted(glob(os.path.join(mask_dir, "*.png")))
    data_len = len(heatmap_name)

    for i in range(0, data_len):
        print('{:d}/{:d}'.format(i, data_len))

        name = image_name[i].split('/')[-1]
        name = name.split('.')[0]

        image = scipy.misc.imread(image_name[i]).astype(np.float32)
        heatamp = scipy.misc.imread(heatmap_name[i]).astype(np.float32)
        pedestrian = scipy.misc.imread(pedestrian_name[500]).astype(np.float32)
        mask = scipy.misc.imread(mask_name[500]).astype(np.float32) / 255
        mask = np.dstack((mask, mask, mask))
        pedestrian *= mask
        ped_height, ped_width = 114, 59

        #scipy.misc.imshow(pedestrian)
        heatmap_index = np.nonzero(heatamp > 10)

        sig, threshold = 7, 10
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
            label_map = np.zeros((256*3, 512*3), dtype=np.float32)
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
            if rx > 512*3 -1:
                cut_rx = rx - (512*3 -1)
                rx = 512*3-1

            dny = upy + ped_height - cut_upy
            if dny > 256*3-1:
                cut_dny = dny - (256*3-1)
                dny = 256*3-1

            hole = (1-mask)[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :] * image[upy:dny, lx:rx, :]
            fill = hole + pedestrian[cut_upy:ped_height-cut_dny, cut_lx:ped_width-cut_rx, :]
            image[upy:dny, lx:rx, :] = fill
            #scipy.misc.imshow(label_map)
            #for ped_name in  pedestrian_name:
            #    ped_name = ped_name.split('/')[-1]
            #    ped_name_split = ped_name.split('_')
            #    height, width = int(ped_name_split[0]), int(ped_name_split[1])



        scipy.misc.imsave('ICCV/synthesis_visual/{}.png'.format(name), image.astype(np.uint8))
        scipy.misc.imsave('ICCV/synthesis_visual/{}_{}_{}.png'.format(name, sig, threshold), image_v.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_seg.png'.format(name), blurred.astype(np.uint8))
        #scipy.misc.imsave('ICCV/synthesis_visual/{}_ped.png'.format(name), pedestrian.astype(np.uint8))

        break

#synthesis_visual()

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
        heatamp = gaussian_filter(heatamp, sigma=1)
        image_heatmap += heatamp
        image_heatmap[np.nonzero(image_heatmap>255)] = 255



        scipy.misc.imsave('ICCV/heatmap_visual/test_low/{}.png'.format(name), image.astype(np.uint8))
        scipy.misc.imsave('ICCV/heatmap_visual/test_low/{}_heatmap.png'.format(name), image_heatmap.astype(np.uint8))

test_visual()

def heatmap_test_visual():
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test'
    image_name = sorted(glob(os.path.join(dir, "*_image.png")))
    dir = '/data/vllab1/Github/streetview_synthesize/Image synthesis/test_low'
    heatmap_name = sorted(glob(os.path.join(dir, "*_heatmap.png")))
    data_len = len(heatmap_name)

    sig, threshold = 2, 10
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

        blurred_index = np.nonzero(blurred > 1)
        image_heatmap = np.copy(image)
        image_heatmap[blurred_index] += 120
        image_heatmap[np.nonzero(image_heatmap > 255)] = 255

        scipy.misc.imsave('ICCV/heatmap_test_visual/test_low/{}.png'.format(name), image.astype(np.uint8))
        scipy.misc.imsave('ICCV/heatmap_test_visual/test_low/{}_heatmap.png'.format(name),
                          image_heatmap.astype(np.uint8))

        break


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


def pedestrian_from_city():
    out = '../../dataset/pedestrian/CITYSCAPES'
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
        image = scipy.misc.imread(os.path.join(dataset_dir, 'fine_image', image_name)).astype(np.float32)
        mask_ori = scipy.misc.imread(os.path.join(dataset_dir, 'fine_mask', mask_name)).astype(np.float32)
        instance = (os.path.join('../../dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
                                 city_name, instance_name))

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
                height, width = maxy - miny, maxx - minx
                if width < 60 or height < 240:
                    continue

                img = Image.new('F', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img)
                mask[np.nonzero(mask > 0)] = 255

                minx, miny, maxx, maxy = minx/4, miny/4, maxx/4, maxy/4
                minx -= 10
                if minx < 0:
                    minx = 0
                miny -= 10
                if miny < 0:
                    miny = 0
                maxx += 10
                if maxx > 511:
                    maxx = 511
                maxy += 10
                if maxy > 255:
                    maxy = 255

                pedestrian_image = image[miny:maxy, minx:maxx]
                #human_pixel = np.nonzero(mask == 255)
                # human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                # if human_ratio > alpha:
                #     mask_instance_sel.append(instance_num)
                instance_num += 1
                print(width, height, instance_num)
                scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES/image/{:d}_{:d}_{}.png'.format(
                    height, width/4, name), pedestrian_image.astype(np.uint8))


                scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES/mask/{:d}_{:d}_{}.png'.format(
                    height, width/4, name), mask[miny:maxy, minx:maxx].astype(np.uint8))

                #mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)

                mask_index = np.nonzero(mask==0)
                image_mask = np.copy(image)
                image_mask[mask_index + (0,)] = 0
                image_mask[mask_index + (1,)] = 0
                image_mask[mask_index + (2,)] = 0
                pedestrian_image = image_mask[miny:maxy, minx:maxx]
                scipy.misc.imsave('../../dataset/pedestrian/CITYSCAPES/crop/{:d}_{:d}_{}.png'.format(
                    height, width/4, name), pedestrian_image.astype(np.uint8))


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



