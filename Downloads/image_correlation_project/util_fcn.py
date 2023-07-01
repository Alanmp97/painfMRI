import os
import nibabel as nib
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization
import tensorflow_datasets as tfds
from tensorflow.keras import optimizers
import functools
from scipy.io import savemat
import random
from sklearn.model_selection import train_test_split
from scipy import ndimage
from sklearn.utils import shuffle


class DefineParameters:
    def __init__(self, batch_train=0, epochs=1, batch_test=1, axis_select=0):
        #para Colab
        #self.data_dir_train = '/content/drive/MyDrive/rawdata'
        
        
        self.data_dir_train = 'C:/Users/Alan/Downloads/rawdata'
        
        #Para funcionales 
        subj = os.listdir(self.data_dir_train)
        sessions = []
        self.files = []
        self.train_file = []
        self.test_file = []
        for i in range(len(subj)): 
            sessions.append(os.listdir(self.data_dir_train + '/' + subj[i]))    
            for j in range(len(sessions[i][:])):
                for f in os.listdir(self.data_dir_train+'/'+subj[i]+'/'+sessions[i][j]+'/func'):
                    name, ext = os.path.splitext(f)
                    if ext == '.nii' and "dist" in name: #solo imagenes funcionales
                        self.files.append(self.data_dir_train + '/' + subj[i] + '/' +
                                           sessions[i][j] + '/func' +'/'+ name + '.nii')
        
        #self.data_dir_train = "C:/Users/Alan/Downloads/rawdata/sub-001/ses-01/func"
          
                        
                        
        """dividir en train y test"""
        self.train_file, self.test_file=train_test_split(self.files, test_size=0.3, random_state=42)
    
        self.axial_coords = ['right', 'forward', 'up', 'pitch', 'roll', 'yaw']
        self.batch_train = batch_train
        self.steps_per_epoch = int(np.ceil(len(self.train_file)/self.batch_train))
        self.epochs = epochs
        self.batch_test = batch_test
        self.steps = int(np.ceil(len(self.test_file) / self.batch_test))
        self.label = self.axial_coords[axis_select]
        self.axis_select = axis_select
        self.VOL_SIZE_WIDTH = 128
        self.VOL_SIZE_HEIGHT = 128
        self.VOL_SIZE_DEEP = 22
        # CNN hyper parameters
        self.CNN_Conv2D_filters = 8
        self.CNN_kernel_size = (3, 3)
        self.CNN_MaxPooling2D_size = (2, 2)
        self.CNN_Dropout_rate = 0.2
        self.CNN_first_layer = 32
        self.CNN_second_layer = 32
        self.CNN_loss = "mean_squared_error"
        self.CNN_learn_rate = 0.001
        self.CNN_optimizer = optimizers.Adam(learning_rate=0.001)
        self.CNN_metrics = ['mean_absolute_error', 'mean_absolute_percentage_error']
        
        
        
def just_one_subj(self, train = True):
    inputs = []
    targets = []
    
    file = "C:/Users/Alan/Downloads/rawdata/sub-001/ses-01/func/sub-001_ses-01_task-dist_bold.nii"
    temp = nib.load(file)
    data = temp.get_fdata()
    
    inputs.extend(func_images_aug(data,augmentation=True))
    targets.extend(labels_func())
    
    """
    #agregamos imagenes augmentadas de distención 
    dist_img, dist_labels = just_dist(inputs)
    inputs.extend(dist_img)
    targets.extend(dist_labels)
    """
    
    inputs, targets = shuffle(inputs, targets)    
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    first_dim = inputs.shape[0]
   
    # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
    data_ds = tf.data.Dataset.from_tensor_slices(inputs)
    data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
    # Convert the dataset to a generator and subsequently to numpy array
    data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
    inputs = np.asarray([data for data in data_ds]).reshape(first_dim, self.VOL_SIZE_WIDTH,
                                                            self.VOL_SIZE_HEIGHT, self.VOL_SIZE_DEEP)
    
    train_x, train_y, test_x, test_y =train_test_split(data_ds, inputs, test_size=0.3, random_state=42)
    
    if train:
        yield train_x, train_y
    
    else:
        yield test_x, test_y

"""data generator para func"""
def train_data_generator_func(self):
    j = 0
    while True:
        if j * self.batch_train >= len(self.train_file):  # This loop is used to run the generator indefinitely.
            j = 0
            random.shuffle(self.train_file)
        else:
            file_chunk = self.train_file[j * self.batch_train : (j + 1) * self.batch_train]

            inputs = []
            targets = []
            for file in file_chunk:
                temp = nib.load(file)
                data = temp.get_fdata()
                
                inputs.extend(func_images_aug(data,augmentation=True))
                targets.extend(labels_func())
                
                #agregamos imagenes augmentadas de distención 
                dist_img, dist_labels = just_dist(inputs)
                inputs.extend(dist_img)
                targets.extend(dist_labels)
                

            inputs, targets = shuffle(inputs, targets)    
            inputs = np.asarray(inputs)
            targets = np.asarray(targets)
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs = np.asarray([data for data in data_ds]).reshape(first_dim, self.VOL_SIZE_WIDTH,
                                                                    self.VOL_SIZE_HEIGHT, self.VOL_SIZE_DEEP)
            yield inputs, targets
            # return inputs, targets
            j = j + 1

def func_images_aug(vol, augmentation = False):
    inputs = []
    if augmentation:
        for i in range(int(np.ceil(900/1.51069))):
            volume = vol[:,:,:,i]
            volume = rotate(volume) #rotamos
            volume = flip(volume) #Flip
            inputs.append(volume)
        
    else:
        for i in range(int(np.ceil(900/1.51069))):
            inputs.append(vol[:,:,:,i])
            
    return inputs

"""
def func_images(volume):
    inputs = []
    for i in range(int(np.ceil(900/1.51069))):
        inputs.append(volume[:,:,:,i])
    return inputs
"""
def just_dist(volume):
    temp = []
    temp1 = []
    temp2 = []
    inputs = []
    for i in range(14):
        temp = volume[(40*(i+1))-10:40*(i+1)]
        for j in range(len(temp)):
            temp1 = flip(temp[j])
            temp2 = rotate(temp[j])
            inputs.append(temp1)
            inputs.append(temp2)
    labels = np.ones(len(inputs))
    return inputs, labels 
    

def labels_func():
    targets = np.zeros(int(np.ceil(900/1.51069)))
    for i in range(14):
        targets[40*i:40*i+30] = -1 #Rest
        targets[(40*(i+1))-10:40*(i+1)] = 1 #Dist
    return targets 

def flip(volume):
    axis = [0,1,2]
    ax = random.choice(axis)
    volume = np.flip(volume, ax)
    return volume 
    
    
    
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = []
        for i in range(0,360,10):
            angles.append(i)
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume
"""
def train_data_generator(self):
    j = 0
    while True:
        if j * self.batch_train >= len(self.train_file):  # This loop is used to run the generator indefinitely.
            j = 0
            random.shuffle(self.train_file)
        else:
            file_chunk = self.train_file[j * self.batch_train : (j + 1) * self.batch_train]
            inputs = []
            targets = []
            for file in file_chunk:
                temp = nib.load(file)
                inputs.append(temp.get_fdata())
                #data_y = (np.genfromtxt(self.data_dir_train + 'label/' + file[:len(file) - 13] +
                 #                       file[len(file) - 7:len(file) - 4] + '.csv', delimiter=",",
                  #                      usecols=self.axis_select + 2))
                #targets.append(data_y)
            inputs = np.asarray(inputs)
            targets = np.asarray(targets)
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs = np.asarray([data for data in data_ds]).reshape(first_dim, self.VOL_SIZE_WIDTH,
                                                                    self.VOL_SIZE_HEIGHT, self.VOL_SIZE_DEEP)
            yield inputs, targets
            # return inputs, targets
            j = j + 1
"""

def test_data_generator(self):
    i = 0
    while True:
        if i * self.batch_test >= len(self.test_file):  # This loop is used to run the generator indefinitely.
            i = 0
        else:
            file_chunk = self.test_file[i * self.batch_test:(i + 1) * self.batch_test]
            inputs = []
            for file in file_chunk:
                temp = nib.load(file)
                data = temp.get_fdata()
                inputs = func_images_aug(data, augmentation=False)
                
            inputs = np.asarray(inputs)
            first_dim = inputs.shape[0]
            # Create tensorflow dataset so that we can use `map` function that can do parallel computation.
            data_ds = tf.data.Dataset.from_tensor_slices(inputs)
            data_ds = data_ds.batch(batch_size=first_dim).map(normalize,
                                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Convert the dataset to a generator and subsequently to numpy array
            data_ds = tfds.as_numpy(data_ds)  # This is where tensorflow-datasets library is used.
            inputs2 = np.asarray([data for data in data_ds]).reshape(first_dim, self.VOL_SIZE_WIDTH,
                                                                     self.VOL_SIZE_HEIGHT, self.VOL_SIZE_DEEP)
            yield inputs2
            i = i + 1


def normalize(volume):
    numerator = tf.subtract(x=volume, y=tf.reduce_min(volume))
    denominator = tf.subtract(tf.reduce_max(volume), tf.reduce_min(volume))
    volume = tf.math.divide(numerator, denominator)
    return volume

# -----------------------------------------------------------------------------------------
def build_CNN_model(self):
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(filters=self.CNN_Conv2D_filters, kernel_size=self.CNN_kernel_size, activation='relu',
                     input_shape=[self.VOL_SIZE_WIDTH, self.VOL_SIZE_HEIGHT, self.VOL_SIZE_DEEP]))
    model.add(MaxPooling2D(pool_size=self.CNN_MaxPooling2D_size))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(self.CNN_Dropout_rate))
    model.add(Conv2D(filters=self.CNN_Conv2D_filters, kernel_size=self.CNN_kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=self.CNN_MaxPooling2D_size))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(self.CNN_Dropout_rate))
    model.add(Conv2D(filters=self.CNN_Conv2D_filters, kernel_size=self.CNN_kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=self.CNN_MaxPooling2D_size))
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(self.CNN_Dropout_rate))
    model.add(Flatten())
    model.add(Dense(self.CNN_first_layer, activation='relu'))
    model.add(Dense(self.CNN_first_layer, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=self.CNN_loss, optimizer=self.CNN_optimizer,
                  metrics=self.CNN_metrics)
    return model


def load_test_targets(self):
    targets = []
    for h in range(len(self.test_file)):
        data_y = (np.genfromtxt(self.data_dir_test + 'label/' + self.test_file[h][:len(self.test_file[h]) - 10] +
                                '.csv', delimiter=",", usecols=self.axis_select + 2))
        targets.append(data_y)
    targets = np.asarray(targets)
    return targets


def loadData_nii_set(file_name):
    data = nib.load('datain/raw/data/' + file_name)
    volume = normalize(data.get_fdata())
    return volume, data


def get_translation(right, forward, up):
    trans_matrix = np.diag(np.ones(4))
    trans_matrix[0, 3] = right  # x-axis
    trans_matrix[1, 3] = forward  # y-axis
    trans_matrix[2, 3] = up  # z-axis
    return trans_matrix


def get_rotations(pitch, roll, yaw):
    pitch_matrix = np.diag(np.ones(4))
    pitch_matrix[1, 1] = np.cos(pitch)
    pitch_matrix[1, 2] = np.sin(pitch)
    pitch_matrix[2, 1] = -np.sin(pitch)
    pitch_matrix[2, 2] = np.cos(pitch)

    roll_matrix = np.diag(np.ones(4))
    roll_matrix[0, 0] = np.cos(roll)
    roll_matrix[0, 2] = np.sin(roll)
    roll_matrix[2, 0] = -np.sin(roll)
    roll_matrix[2, 2] = np.cos(roll)

    yaw_matrix = np.diag(np.ones(4))
    yaw_matrix[0, 0] = np.cos(yaw)
    yaw_matrix[0, 1] = np.sin(yaw)
    yaw_matrix[1, 0] = -np.sin(yaw)
    yaw_matrix[1, 1] = np.cos(yaw)
    rot_matrix = functools.reduce(np.dot, [pitch_matrix, roll_matrix, yaw_matrix])
    return rot_matrix


def get_predicteAffine(prediction):
    affine_tra = get_translation(prediction[0], prediction[1], prediction[2])
    affine_rot = get_rotations(prediction[3], prediction[4], prediction[5])
    affine_final = functools.reduce(np.dot, [affine_tra, affine_rot])
    return affine_final


def save_matfile(data, filename, patch):
    FrameStack = np.empty((len(data),), dtype=np.object)
    for i in range(len(data)):
        FrameStack[i] = data[i]
    savemat(patch + "/" + filename + "_reorient" + ".mat", {"FrameStack": FrameStack})
    np.savetxt(patch + "/" + filename + "_reorient" + ".txt", data)
    return None


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

