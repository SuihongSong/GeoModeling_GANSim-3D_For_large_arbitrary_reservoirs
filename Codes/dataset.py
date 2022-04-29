#--------------------------------------------------------------------------------------------------
# This file contains functions about how training dataset of tfrecords files are loaded.  

# The code goes with our paper as following, please refer to it:
#  Suihong Song, Tapan Mukerji, Jiagen Hou, Dongxiao Zhang, Xinrui Lyu. GANSim-3D for conditional geomodelling: theory and field application. 
# Now available at my ResearchGate (https://www.researchgate.net/profile/Suihong-Song)  
#--------------------------------------------------------------------------------------------------

import os
import glob
import numpy as np
import tensorflow as tf
import tfutil

#----------------------------------------------------------------------------
# Parse individual cube from a tfrecords file.

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([4], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_tf_float16(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([4], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.float16)
    return tf.reshape(data, features['shape']) 

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

def parse_tfrecord_np_float16(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.float16).reshape(shape)

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        labeltypes      = None,     # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'
        repeat          = True,     # Repeat dataset indefinitely.
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2,        # Number of concurrent threads.
        well_enlarge    = False):   # If enlarged well points are outputted    

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channel, length, height, width]
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self.label_file         = label_file
        self.label_size         = 0      
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_probcubes_dataset = None
        self._tf_wellfacies_dataset = None        
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1
        self.well_enlarge       = well_enlarge  

        # List realcube tfrecords files and inspect their shapes.
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        tfr_realcube_files = tfr_files[:-2] #as tfrecord files include 02-06 real cube files, one prob_cube file and one well_facies file, [:-2] ensures only reale cube files are selected.
        assert len(tfr_realcube_files) >= 1
        tfr_realcube_shapes = []  # tfr_realcube_shapes
        for tfr_realcube_file in tfr_realcube_files:  # 
            tfr_realcube_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for record in tf.python_io.tf_record_iterator(tfr_realcube_file, tfr_realcube_opt):
                tfr_realcube_shapes.append(parse_tfrecord_np(record).shape)
                break

        # List probcube tfrecord files and inspect its shape.
        tfr_probcube_file = tfr_files[-2] #as tfrecord files include 02-06 real cube files, one prob_cube file and one well_facies file, [-2] ensures only prob cube file are selected.
        tfr_probcube_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(tfr_probcube_file, tfr_probcube_opt):
            tfr_probcube_shape = parse_tfrecord_np_float16(record).shape
            break
        
        # List well facies tfrecord files and inspect its shape.
        tfr_wellfacies_file = tfr_files[-1] #as tfrecord files include 02-06 real cube files, one prob_cube file and one well_facies file, [-1] ensures only well facies file are selected.
        tfr_wellfacies_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(tfr_wellfacies_file, tfr_wellfacies_opt):
            tfr_wellfacies_shape = parse_tfrecord_np(record).shape   # well facies only contain code of 0 (no well facies), 1 (mud well facies), 2 (levee), 3 (channel)
            break
        
        # Autodetect label filename.
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Determine shape and resolution of realcube. some parameters are marked with _realcube_, but some are not. All probcube related parameters are marked with _probcube_.
        max_realcube_shape = max(tfr_realcube_shapes, key=lambda shape: np.prod(shape))
        self.resolution = resolution if resolution is not None else max_realcube_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_realcube_shape[0], self.resolution, self.resolution, self.resolution]
        tfr_realcube_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_realcube_shapes]
        assert all(shape[0] == max_realcube_shape[0] for shape in tfr_realcube_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_realcube_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_realcube_shapes, tfr_realcube_lods))
        assert all(lod in tfr_realcube_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        self.label_size = len(labeltypes)
        assert self.label_size >= 0
        self._np_labels = np.zeros([1<<17, 0], dtype=np.float32)
        if self.label_size > 0: self._np_labels = np.load(self.label_file)[:, labeltypes]                  
        if self.label_size == 0: self._np_labels = np.load(self.label_file)[:, :0]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            tf_labels_init = tf.zeros(self._np_labels.shape, self._np_labels.dtype)
            self._tf_labels_var = tf.Variable(tf_labels_init, name='labels_var')
            tfutil.set_vars({self._tf_labels_var: self._np_labels})
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)        

            for tfr_realcube_file, tfr_realcube_shape, tfr_realcube_lod in zip(tfr_realcube_files, tfr_realcube_shapes, tfr_realcube_lods):
                if tfr_realcube_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(tfr_realcube_file, compression_type='', buffer_size=buffer_mb<<17)
                dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
                bytes_per_item = np.prod(tfr_realcube_shape) * np.dtype(self.dtype).itemsize  
                if shuffle_mb > 0:
                    dset = dset.shuffle(((shuffle_mb << 17) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(((prefetch_mb << 17) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_realcube_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}
            
            tf_probcubes_dset = tf.data.TFRecordDataset(tfr_probcube_file, compression_type='', buffer_size=buffer_mb<<17)
            tf_probcubes_dset = tf_probcubes_dset.map(parse_tfrecord_tf_float16, num_parallel_calls=num_threads)
            self._tf_probcubes_dataset = tf_probcubes_dset
            tf_wellfacies_dset = tf.data.TFRecordDataset(tfr_wellfacies_file, compression_type='', buffer_size=buffer_mb<<17)
            tf_wellfacies_dset = tf_wellfacies_dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)             
            self._tf_wellfacies_dataset = tf_wellfacies_dset 
            self._tf_probcubes_wellfacies_dset = tf.data.Dataset.zip((self._tf_probcubes_dataset, self._tf_wellfacies_dataset)) 
            if shuffle_mb > 0:
                self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.shuffle(((shuffle_mb << 17) - 1) // bytes_per_item + 1)
            if repeat:
                self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.repeat()
            if prefetch_mb > 0:
                self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.prefetch(((prefetch_mb << 17) - 1) // bytes_per_item + 1)         
            self._tf_probcubes_wellfacies_dset = self._tf_probcubes_wellfacies_dset.batch(self._tf_minibatch_in)
            self._tf_probcubes_wellfacies_iterator = tf.data.Iterator.from_structure(self._tf_probcubes_wellfacies_dset.output_types,  self._tf_probcubes_wellfacies_dset.output_shapes)
            self._tf_probcubes_wellfacies_init_ops = self._tf_probcubes_wellfacies_iterator.make_initializer(self._tf_probcubes_wellfacies_dset)              

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._tf_probcubes_wellfacies_init_ops.run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self, minibatch_size): # => cubes, labels, probcubes
        cubes, labels = self._tf_iterator.get_next()  
        probcubes, wellfacies = self._tf_probcubes_wellfacies_iterator.get_next()      
        if self.well_enlarge:
            # 1) to enlarge area influenced by well facies; 2) since cpu device only accept max_pool opt with data_format = 'NHWC' instead of 'NCHW', so transpose twice to deal with that
            wellfacies = tf.cast(wellfacies, tf.float16)
            wellfacies = tf.transpose(wellfacies, perm=[0, 2, 3, 4, 1])
            wellfacies = tf.nn.max_pool3d(wellfacies, ksize = [1,4,4,4,1], strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC') 
            wellfacies = tf.transpose(wellfacies, perm=[0, 4, 1, 2, 3])                        
        return cubes, labels, probcubes, wellfacies

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => cubes, labels, probcubes
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf(minibatch_size)
        return tfutil.run(self._tf_minibatch_np)
    
    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_cubeandlabel_tf(self, minibatch_size): # => cubes, labels
        cubes, labels = self._tf_iterator.get_next() 
        return cubes, labels

    # Get next minibatch as NumPy arrays.
    def get_minibatch_cubeandlabel_np(self, minibatch_size, lod=0): # => cubes, labels
        self.configure(minibatch_size, lod)
        return tfutil.run(self.get_minibatch_cubeandlabel_tf(minibatch_size))
        
     # Get next minibatch as TensorFlow expressions.
    def get_minibatch_probandwell_tf(self, minibatch_size): # => probcubes, wellfacies
        probcubes, wellfacies = self._tf_probcubes_wellfacies_iterator.get_next() 
        if self.well_enlarge:
            # 1) to enlarge area influenced by well facies; 2) since cpu device only accept max_pool opt with data_format = 'NHWC' instead of 'NCHW', so transpose twice to deal with that
            wellfacies = tf.cast(wellfacies, tf.float16)
            wellfacies = tf.transpose(wellfacies, perm=[0, 2, 3, 4, 1])
            wellfacies = tf.nn.max_pool3d(wellfacies, ksize = [1,4,4,4,1], strides=[1,1,1,1,1], padding='SAME', data_format='NDHWC') 
            wellfacies = tf.transpose(wellfacies, perm=[0, 4, 1, 2, 3])         
        return probcubes, wellfacies

    # Get next minibatch as NumPy arrays.
    def get_minibatch_probandwell_np(self, minibatch_size, lod=0): # => probcubes, wellfacies
        self.configure(minibatch_size, lod)
        return tfutil.run(self.get_minibatch_probandwell_tf(minibatch_size))      

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        if self.label_size > 0:
            return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
        else:
            return tf.zeros([minibatch_size, 0], self.label_dtype)
     
    # Get random probcubes as TensorFlow expression.
    def get_random_probcubes_tf(self, minibatch_size): # => probcubes
        return self._tf_iterator.get_next()[2]  # has problem when using multiple gpus, because it always generate minibatch_size probcubes, while get_random_labels_tf generate minibatch_size_split labels, see G_loss function for more details. 
 

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        else:
            return np.zeros([minibatch_size, 0], self.label_dtype)
        

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name='dataset.TFRecordDataset', data_dir=None, verbose=False, **kwargs):
    adjusted_kwargs = dict(kwargs)
    if 'tfrecord_dir' in adjusted_kwargs and data_dir is not None:
        adjusted_kwargs['tfrecord_dir'] = os.path.join(data_dir, adjusted_kwargs['tfrecord_dir'])
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = tfutil.import_obj(class_name)(**adjusted_kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset
