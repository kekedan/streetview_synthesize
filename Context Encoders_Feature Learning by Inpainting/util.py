import skimage.io
import skimage.transform
import scipy.misc

import numpy as np


def read_mask(train_annotation_name):
    img = scipy.misc.imread(train_annotation_name).astype(np.uint8) / 255
    train_annotations = np.dstack((img, img, img))
    return train_annotations

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


#def load_image( path, height=128, width=128 ):
def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):

    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    rand_y = np.random.randint(0, pre_height - height)
    rand_x = np.random.randint(0, pre_width - width)

    resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    return (resized_img * 2)-1 #(resized_img - 127.5)/127.5

def crop_random(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap,height-overlap) if x is None else x
    random_x = np.random.randint(overlap,width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 0] = 2*117. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 1] = 2*104. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y
