import sys
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

# Set the path to directory containing code of this case
new_path = r'/home/users/suihong/5-karstcave3D_Complete_cond_well_globalfeatures_probmap_Upload/' 
sys.path.append(new_path)

#================================================================================================================
# 1. Import pre-trained Network
#================================================================================================================
# Set path to trained network
network_dir = '/scratch/users/suihong/CaveSimulation/GANTrainingResults/110-pgan3D-4gpu-V100-OnlyCV-LC_8-GANw_2-NoLabelCond-CondWell_0.7-Enlarg-CondProb_1/'
network_name = 'network-snapshot-003360.pkl'

# Import pre-trained Network
# Initialize TensorFlow session.
tf.InteractiveSession()
# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, Gs = pickle.load(file)    
del G

#================================================================================================================
# 2. Import conditioning data from field measurements
#================================================================================================================
#-------------------------
# 2.1 Import the prob cube
#-------------------------
# Read data from Petrel Gslib file.
# In the file the data are like following with I, J, K, wellfacies code (-99 means no well)
# 1 1 1 -99.00 0.140967 
# 2 1 1 -99.00 0.148176 
# 3 1 1 -99.00 0.154223

# well_probdata directory
wellfacies_path = '/scratch/users/suihong/CaveSimulation/GANTrainingResults/S65_96by256by336/well_probdata'

# Define the size of reservoir (expected output size of the generator)
resolution_z = 96
resolution_x = 336
resolution_y = 256

allele=[] 
with open(wellfacies_path, 'r') as f:
    for line in f:
        eles = line.strip().split(' ')
        allele.append(eles)

allfiledata = np.array(allele).reshape((-1, 5))
probdata_orig = allfiledata[:, 4].reshape((1, 1, resolution_z, resolution_x, resolution_y)).transpose(0,1,3, 4, 2) 
probdata_orig = probdata_orig[:,:,::-1,:,:].astype(np.float)
del allele
probdata = probdata_orig 

#-------------------------
# 2.1 Import well data cube
#--------------------------
wellfacies = allfiledata[:, 3].reshape((1, 1, resolution_z, resolution_x, resolution_y)).transpose(0,1,3, 4, 2) 
wellfacies = wellfacies[:,:,::-1,:,:].astype(np.float)
print(wellfacies.shape)
del allfiledata

well_points = np.where(wellfacies == -99, 0, 1)
well_facies_types = np.where(wellfacies == -99, 0, wellfacies)
well_facies_input = np.concatenate([well_points, well_facies_types], 1)      

#Enlarge areas of well points into 4 x 4 as inputs
well_facies_input = well_facies_input.astype(np.float32)
with tf.device('/gpu:0'):
    well_facies_input_enlarge = tf.nn.max_pool3d(well_facies_input, ksize = [1,1,4,4,4], strides=[1,1,1,1,1], padding='SAME', data_format='NCDHW') 

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    well_facies_input_el = sess.run(well_facies_input_enlarge)

#====================================================================================================================
# 3. Build a new generator having large-size inputs/output and copy the trained parameters from the trained generator
#====================================================================================================================

latent_size_z = int(resolution_z/16)  
latent_size_x = int(resolution_x/16)
latent_size_y = int(resolution_y/16)

label_size = 0

import config
import tfutil
Gs_enlarged = tfutil.Network('Gs_enlarged', num_channels=1, label_size=label_size, 
                             resolution_z = resolution_z, resolution_x = resolution_x, resolution_y = resolution_y,
                             latent_size_z = latent_size_z, latent_size_x = latent_size_x,latent_size_y = latent_size_y,    
                             **config.G) #

Gs_enlarged.copy_trainables_from(Gs)
del Gs

#===============================================================================================================================================
# 4. Generate facies models using the newly built generator (with large-size inputs/output) by taking the large-size conditioning data as inputs
#================================================================================================================================================
import time
start_time = time.time()
TotalReal_no_syn = 200
condfakemodels_syn = np.zeros([TotalReal_no_syn, 1, resolution_x, resolution_y, resolution_z])
latents_syn = np.random.randn(TotalReal_no_syn, 8, latent_size_x, latent_size_y, latent_size_z)  # In paper it is np.random.RandomState(1315) used.

for i in range (int(TotalReal_no_syn/1)): 
    latents_i_syn = latents_syn[i*1:(i+1)*1]
    labels_i_syn = np.random.randn(1, 0, latent_size_x, latent_size_y, latent_size_z)  #.RandomState(816)
    well_facies_plt = np.repeat(well_facies_input_el, 1, axis=0)
    probcube_plt = np.repeat(probdata, 1, axis=0)  
    fakemodels_i = Gs_enlarged.run(latents_i_syn, labels_i_syn, well_facies_plt, probcube_plt)
    fakemodels_i = np.where(fakemodels_i< 0, -1, 1)
    condfakemodels_syn[i*1:(i+1)*1] = fakemodels_i
    
end_time = time.time()
time_each = (end_time - start_time)/TotalReal_no_syn
print(time_each)

#===============================================================================================================================================
# 5. Export the generate facies models for displaying in Petrel platform
#================================================================================================================================================
#-------------------------------------------------
# 5.1 Preparing index of I J K requireded in the file
#-------------------------------------------------
onecube = np.zeros([resolution_z,resolution_x,resolution_y])
print(onecube.shape)
coords = np.argwhere(onecube>-1)
print(coords.shape)
coords_output = np.zeros(coords.shape)
coords_output[:,0] = coords[:,2]+1
coords_output[:,1] = coords[:,1]+1
coords_output[:,2] = resolution_z - coords[:,0]
coords_output = coords_output.astype(np.int16)

#-------------------------------------------------
# 5.2 Export 30 realizations
#-------------------------------------------------
condfakemodels_syn_output = condfakemodels_syn[0:30].transpose(0,1,4,2,3)[:,:,:,::-1,:].reshape(-1, coords.shape[0]).T.astype(np.int)
condfakemodels_syn_output = np.where(condfakemodels_syn_output>0.5,1,-99)
condfakemodels_syn_output_txt = np.concatenate((coords_output, condfakemodels_syn_output), axis = 1)
                                               
condfakemodels_syn_fname = '/scratch/users/suihong/CaveSimulation/GANTrainingResults/S65_96by256by336/Random30CondFakeFaciesModels_110_3360.txt'
np.savetxt(condfakemodels_syn_fname, condfakemodels_syn_output_txt, fmt='%i', newline='\n')   # "%03d" + "%.10f"*7
                                               