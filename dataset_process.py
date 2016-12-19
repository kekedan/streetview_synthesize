from glob import glob
import os

from utils import *

CITYSCAPES_dir = "/mnt/data/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train"
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
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/image/' + filePath.split('/')[-1], img)
        #break


def crop_images_label(dataset_dir, is_mask=True):
    """
    Read all labels under the different folders
    Crop, resize and store them
    example code:
        ddir = '/home/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train'
        crop_images_label(ddir, is_mask=True)
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

        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label2/' + filePath.split('/')[-1], img)
        #break


def crop_images_label_big(dataset_dir, is_mask=True):
    """
    Read all labels under the different folders
    Crop, resize and store them
    The mask range should be larger
    example code:
        ddir = '/home/andy/dataset/CITYSCAPES/gtFine_trainvaltest/gtFine/train'
        crop_images_label(ddir, is_mask=True)
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
        scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/label2_big/' + filePath.split('/')[-1], img)


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
    visual = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(0, 34):
        index = np.nonzero(img == i)
        visual[index + (0,)] = labels[i][0]
        visual[index + (1,)] = labels[i][1]
        visual[index + (2,)] = labels[i][2]

    scipy.misc.imsave('/home/andy/dataset/CITYSCAPES/for_wonderful_chou/' + img_dir.split('/')[-1], visual)


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

