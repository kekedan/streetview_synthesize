from glob import glob
import json
from PIL import Image, ImageDraw
import pickle
import os

from utils import *

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
        scipy.misc.imsave('/mnt/data/andy/dataset/CITYSCAPES/image/' + filePath.split('/')[-1], img)
        #break


crop_images()
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
        scipy.misc.imsave('/mnt/data/andy/dataset/CITYSCAPES/mask/' + filePath.split('/')[-1], img)
        #break


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
    data = sorted(glob(os.path.join('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image', "*.png")))
    label = sorted(glob(os.path.join('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label2_big', "*.png")))

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
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image_mask/mask_' + data[i].split('/')[-1], img)
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
    dataset_dir = '/data/vllab1/dataset/CITYSCAPES/CITY'
    file_obj = open('human_fileName_new', 'r')
    human_file_name = pickle.load(file_obj)
    length = len(human_file_name)

    for index in range(0, length):
        name = human_file_name[index]
        print ('%d/%d : %s' % (index, length, name))
        image_name = '{}_leftImg8bit.png'.format(name)
        instance_name = '{}_gtFine_polygons.json'.format(name)
        city_name = instance_name.split('_', 1)[0]

        image = scipy.misc.imread(os.path.join(dataset_dir, 'human_new_image', image_name))
        instance = (os.path.join('/data/vllab1/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train',
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
                img = Image.new('L', (2048, 1024), 0)
                ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                img = scipy.misc.imresize(img, 0.25, interp='bilinear', mode=None)
                mask = np.array(img) * 255

                human_pixel = np.nonzero(mask == 255)
                human_ratio = float(len(human_pixel[0])) / float((img.shape[0] * img.shape[1]))
                print('instance: {:d}, ratio:{:f}'.format(instance_num, human_ratio))
                if human_ratio > alpha:
                    scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/human_new_instance/{}_{:d}.png'.format(
                        image_name.split('.')[0], instance_num), mask)

                    img = np.copy(image)
                    img[human_pixel + (0,)] = 0
                    img[human_pixel + (1,)] = 255
                    img[human_pixel + (2,)] = 0
                    scipy.misc.imsave('/data/vllab1/dataset/CITYSCAPES/CITY/human_new_instance_mask/{}_{:d}.png'.format(
                        image_name.split('.')[0], instance_num), img)

                    instance_num += 1

        #for key, value in sorted(label_instance.items()):
        #    print(key)


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
    beta = 0.01
    data_set_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/mask'
    data_set_image_dir = '/data/vllab1/dataset/CITYSCAPES/CITY/image'
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
        elif human_ratio > beta:
            h_human.append(name)
        else:
            wo_human.append(name)

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


def load_image_with_name():
    dataset_dir = '/data/vllab1/dataset/CITYSCAPES/CITY'
    file_obj = open('human_fileName', 'r')
    human_file_name = pickle.load(file_obj)
    for index in range(0, len(human_file_name)):
        name = human_file_name[index]
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)

        image = scipy.misc.imread(os.path.join(dataset_dir, 'image', image_name))
        mask = scipy.misc.imread(os.path.join(dataset_dir, 'mask', mask_name))
        scipy.misc.imsave(image_name, image)
        scipy.misc.imsave(mask_name, mask)
        break


def load_batch_with_name():
    dataset_dir = '/data/vllab1/dataset/CITYSCAPES/CITY'
    file_obj = open('human_w.pkl', 'r')
    human_file_name = pickle.load(file_obj)
    mask = np.zeros((4, 256, 512))
    for index in range(0, 4):
        name = human_file_name[index]
        image_name = '{}_leftImg8bit.png'.format(name)
        mask_name = '{}_gtFine_labelIds.png'.format(name)

        mask[index, :, :] = scipy.misc.imread(os.path.join(dataset_dir, 'mask', mask_name))

    scipy.misc.imsave('human_w.png', merge(mask, (2,2), is_gray=True))

