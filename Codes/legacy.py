#--------------------------------------------------------------------------------------------------
# This file shows one small helper function used in misc.py

# The code goes with our paper as following, please refer to it:
#  Suihong Song, Tapan Mukerji, Jiagen Hou, Dongxiao Zhang, Xinrui Lyu. GANSim-3D for conditional geomodelling: theory and field application. 
# Now available at my ResearchGate (https://www.researchgate.net/profile/Suihong-Song)  
#--------------------------------------------------------------------------------------------------

import pickle
import tfutil

#----------------------------------------------------------------------------
# Custom unpickler that is able to load network pickles produced by
# the old Theano implementation.

class LegacyUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == 'network' and name == 'Network':
            return tfutil.Network
        return super().find_class(module, name)

