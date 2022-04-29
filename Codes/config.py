#--------------------------------------------------------------------------------------------------
# This file contains hyperparameters, directory pathes, names, etc. which are called in train.py.
# These parameters should be set before training. 

# The code goes with our paper as following, please refer to it:
#  Suihong Song, Tapan Mukerji, Jiagen Hou, Dongxiao Zhang, Xinrui Lyu. GANSim-3D for conditional geomodelling: theory and field application. 
# Now available at my ResearchGate (https://www.researchgate.net/profile/Suihong-Song)  
#--------------------------------------------------------------------------------------------------

# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.
data_dir = '/scratch/users/suihong/CaveSimulation/DatasetsforGAN_ave_3D/'  # Training data path
result_dir = '/scratch/users/suihong/CaveSimulation/GANTrainingResults/'  # result data path

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '0'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan3D'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict(tfrecord_dir='TrainingData')         # Options for dataset.load_dataset(). dataset is from 'TrainingData' folder of data_dir 
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.

desc += '-4gpu-V100'; num_gpus = 4; sched.minibatch_base = 32; sched.minibatch_dict = {4: 32, 8: 32, 16: 32, 32: 32, 64: 16}; sched.G_lrate_dict = {4: 0.0025, 8: 0.005, 16: 0.005, 32: 0.0035, 64: 0.0025}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 600000
sched.max_minibatch_per_gpu = {32: 32, 64: 16}

#----------------------------------------------
# Settings for condition to global features
desc += '-NoLabelCond';           
labeltypes = []  # can include: 0 for 'channelorientation', 1 for 'mudproportion', 2 for 'channelwidth', 3 for 'channelsinuosity'; but the loss for channel orientation has not been designed in loss.py.
# [] for no label conditioning. 
dataset.labeltypes = labeltypes
G_loss.labeltypes = labeltypes

#----------------------------------------------
# Settings for condition to well facies data
desc += '-CondWell_0.35';          
dataset.well_enlarge = True; desc += '-Enlarg';  # uncomment this line to let the dataset output enlarged well facies data; comment to make it unenlarged.
#----------------------------------------------
# Settings for condition to probability data
desc += '-CondProb_2.5';          
#----------------------------------------------
# Setting if loss normalization (into standard Gaussian) is used 
G_loss.lossnorm = True
#----------------------------------------------
# Set if no growing, i.e., the conventional training method. Can be used if only global features are conditioned.
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 10000