import os, struct
from array import array as pyarray
from cvxopt.base import matrix
import numpy as np

from skimage import color, exposure, transform, io
import glob
import h5py

from skimage.io import imread, imsave

NUM_CLASSES = 43
IMG_SIZE = 32



def LABELS(index):
    labels = ['speed limit 20 (prohibitory)', 'speed limit 30 (prohibitory)',
     'speed limit 50 (prohibitory)', 'speed limit 60 (prohibitory)',
      'speed limit 70 (prohibitory)', 'speed limit 80 (prohibitory)',
       'restriction ends 80 (other)',  'speed limit 100 (prohibitory)',
        'speed limit 120 (prohibitory)', 'no overtaking (prohibitory)',
        'no overtaking (trucks) (prohibitory)', 'priority at next intersection (danger)',
         'priority road (other)', 'give way (other)', 'stop (other)',
         'no traffic both ways (prohibitory)', 'no trucks (prohibitory)',
          'no entry (other)', 'danger (danger)', 'bend left (danger)',
          'bend right (danger)', 'bend (danger)', 'uneven road (danger)',
           'slippery road (danger)', 'road narrows (danger)', 'construction (danger)',
           'traffic signal (danger)',
          'pedestrian crossing (danger)', 'school crossing (danger)',
          'cycles crossing (danger)','snow (danger)', 'animals (danger)',
          'restriction ends (other)', 'go right (mandatory)', 'go left (mandatory)',
          'go straight (mandatory)', 'go right or straight (mandatory)',
          'go left or straight (mandatory)', 'keep right (mandatory)',
           'keep left (mandatory)','roundabout (mandatory)',
            'restriction ends (overtaking) (other)', 'restriction ends (overtaking (trucks)) (other)']
    return labels[index]

def preprocess_img(img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])


def toVector(num):
    ar = np.zeros(NUM_CLASSES)
    ar[num-1] = 1
    return ar



# def read_dataset():
#
#     root_dir = 'networks/gtsrb/Final_Training/Images/'
#     imgs = []
#     labels = []
#
#     all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
#     print(len(all_img_paths))
#     np.random.shuffle(all_img_paths)
#     for img_path in all_img_paths:
#         #print(img_path)
#         try:
#             img = preprocess_img(io.imread(img_path))
#             label = get_class(img_path)
#             imgs.append(img)
#             labels.append(label)
#
#             if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
#         except (IOError, OSError):
#             print('missed', img_path)
#             pass
#
#     X = np.array(imgs, dtype='float32')
#     Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]
#
#     with h5py.File('networks/X.h5','w') as hf:
#         hf.create_dataset('imgs', data=X)
#         hf.create_dataset('labels', data=Y)
#     print("read_dataset")
#     print(X)
#     print(Y)
#    return X, Y

def read_dataset():

    try:
        with  h5py.File('networks/X.h5') as hf:
            X, Y = hf['imgs'][:], hf['labels'][:]
        X = np.array(map(lambda x: transform.resize(x, (3, IMG_SIZE,IMG_SIZE)), X))
        Y = np.array(map(toVector, Y))
        print("Loaded images from X.h5")


    except (IOError,OSError, KeyError, IndexError):
        print("Error in reading X.h5. Processing all images...")
        root_dir = 'networks/gtsrb/Final_Training/Images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]


        with h5py.File('networks/X.h5','w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y+1)







    return X, Y


def save(layer,image,filename):
    """
    """
    import cv2

    import copy

    image_cv = copy.deepcopy(image)
    image_cv = image_cv.transpose(1, 2, 0)

    #print(np.amax(image),np.amin(image))

    params = list()
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(0)

    cv2.imwrite(filename, image_cv * 255.0, params)

def saveim(layer,image,filename):
    imsave(filename,image)

'''


def read(digits=np.arange(10), dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """
    if dataset is "training":
        fname_img = os.path.join(path, 'mnist/train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist/train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 'mnist/t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist/t10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels




def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

'''
