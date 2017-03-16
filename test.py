import numpy as np
import scipy.misc
sample_images = np.array(scipy.misc.imread('/data/vllab1/dataset/CITYSCAPES/CITY/fine_image/tubingen_000072_000019_leftImg8bit.png').astype(np.float32))
sample_images_2 = np.array(scipy.misc.imread('/data/vllab1/dataset/CITYSCAPES/CITY/fine_image/tubingen_000072_000019_leftImg8bit.png').astype(np.uint8))
#sample_images = np.array(sample_images) / 255 * 2 - 1
diff = sample_images - sample_images_2
yo = np.concatenate((sample_images, sample_images_2), axis=1)
scipy.misc.imshow(yo.astype(np.float32))