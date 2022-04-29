#--------------------------------------------------------------------------------------------------
# This file shows how networks of Generator and Discriminator are constucted. 
# These network functions/classes are called by the main function in train.py file.

# The code goes with our paper as following, please refer to it:
#  Suihong Song, Tapan Mukerji, Jiagen Hou, Dongxiao Zhang, Xinrui Lyu. GANSim-3D for conditional geomodelling: theory and field application. 
# Now available at my ResearchGate (https://www.researchgate.net/profile/Suihong-Song)  
#--------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.
  # x with shape  (None, 128, 4, 4, 4)
  # input shape should be [batch, in_channels, in_depth, in_height, in_width], if with data_format='NCDHW'
  # the output shape is the same as input.
  # filter shape [filter_depth, filter_height, filter_width, in_channels,out_channels]
def conv3d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding='SAME', data_format='NCDHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1, 1])
    
#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale3d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale3d'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1, s[4], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor, s[4] * factor])
        return x

#----------------------------------------------------------------------------
# Box filter downscaling layer.
def downscale3d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale3d'):
        ksize = [1, 1, factor, factor, factor]
        return tf.nn.avg_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Box filter wellfc_downscale3d_process layer.
# x: [minibatch, 2, resolution, resolution, resolution] where number 2 refers: a channel for well locations (where 1 for well locations, 0 for no well locations); another channel for facies code (where 0 for background facies, 1 for channel and levee facies).
def wellfc_downscale3d_process(x, factor=2):  
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale3D'):
        ksize = [1, 1, factor, factor, factor]  
        sum_pool = tf.nn.avg_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW') * (factor**2) # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True
        wellfc_downscale_av = tf.where(sum_pool[:,0:1]>0, sum_pool[:,1:2]/sum_pool[:,0:1], sum_pool[:,0:1])  # e.g., [Minibatch, 1, 4, 4]
        wellloc_downscale = tf.cast((sum_pool[:,0:1] > 0), tf.float32)
        return tf.concat([wellloc_downscale, wellfc_downscale_av], axis=1)

#----------------------------------------------------------------------------
# Fused conv3d + downscale3d.
# Faster and uses less memory than performing the operations separately.

def conv3d_downscale3d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:, 1:], w[1:, 1:, :-1], w[1:, :-1, 1:], w[1:, :-1, :-1],
                  w[:-1, 1:, 1:], w[:-1, 1:, :-1], w[:-1, :-1, 1:], w[:-1, :-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,2,2,2], padding='SAME', data_format='NCDHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCDHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3], s[4]])   # [GMCDHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCDHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCDHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCDHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCDHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3,4], keepdims=True)      # [M1111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M1111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3], s[4]])             # [N1DHW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCDHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Generator network.

def G_paper(
    latents_in,                         # First input: Latent vectors 
    labels_in,                          # Second input: global features 
    wellfacies_in,                      # Third input: wellfacies [minibatch, 2, resolution, resolution, resolution]: well locations and facies code. 
    probcubes_in,                       # Forth input: probcubes [minibatch, 1, resolution, resolution, resolution].
    latent_cube_num     = 8,            # Number of input latent cube (64x64x64).
    num_channels        = 1,            # Number of output cube channels, which is always set as 1 in this project. Overridden based on dataset.
    resolution_z        = 64,           # Output resolution. Overridden based on dataset.
    resolution_x        = 64,
    resolution_y        = 64,
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 2048,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.
    latent_size_z       = 4,            # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
    latent_size_x       = 4,
    latent_size_y       = 4,
    normalize_latents   = False,        # Normalize latent vectors before feeding them to the network?
    use_wscale          = True,         # Enable equalized learning rate?
    use_pixelnorm       = True,         # Enable pixelwise feature vector normalization?
    pixelnorm_epsilon   = 1e-8,         # Constant epsilon for pixelwise feature vector normalization.
    use_leakyrelu       = True,         # True = leaky ReLU, False = ReLU.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = False,         # True = use fused upscale3d + conv3d, False = separate upscale3d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    wellfc_conv_channels = 16,
    prob_conv_channels  = 16,
    **kwargs):                          # Ignore unrecognized keyword args.
    
    assert resolution_z >= 64 and resolution_x >= 64 and resolution_y >= 64
    assert resolution_z/16 == latent_size_z and resolution_x/16 == latent_size_x and resolution_y/16 == latent_size_y
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu if use_leakyrelu else tf.nn.relu   

    latents_in.set_shape([None, latent_cube_num, latent_size_x, latent_size_y, latent_size_z]) # (None, N, 4, 4, 4)
    labels_in.set_shape([None, label_size, latent_size_x, latent_size_y, latent_size_z])       # (None, number of global features, 4, 4, 4)
    combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)   # fuse latents_in and labels_in together
    
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)  # assigned by sched.lod in the main function, changes during training
 
    wellfacies_in.set_shape([None, 2, resolution_x, resolution_y, resolution_z])
    wellfacies_in = tf.cast(wellfacies_in, tf.float32)
    probcubes_in.set_shape([None, 1, resolution_x, resolution_y, resolution_z])
    probcubes_in = tf.cast(probcubes_in, tf.float32)

    # Building blocks.
    def block(x, prob, wellfc, res): 
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res == 2: # 4x4
                if normalize_latents: x = pixel_norm(x, epsilon=pixelnorm_epsilon)
                with tf.variable_scope('Conv'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))                                        
                with tf.variable_scope('Add_Prob'):
                    prob_downscaled = downscale3d(prob, factor=int(2**(6-res)))
                    prob_downscaled_conv = apply_bias(conv3d(prob_downscaled, fmaps=prob_conv_channels, kernel=1, gain=1, use_wscale=use_wscale))
                    x = tf.concat([x, prob_downscaled_conv], axis=1)
                with tf.variable_scope('Add_Wellfc'):
                    wellfc_downscaled = wellfc_downscale3d_process(wellfc, factor=int(2**(6-res)))
                    wellfc_downscaled_conv = apply_bias(conv3d(wellfc_downscaled, fmaps=wellfc_conv_channels, kernel=1, gain=1, use_wscale=use_wscale))
                    x = tf.concat([x, wellfc_downscaled_conv], axis=1)

            else: # 8x8 and up
                x = upscale3d(x)
                with tf.variable_scope('Add_Prob'):
                    prob_downscaled = downscale3d(prob, factor=int(2**(6-res)))
                    prob_downscaled_conv = apply_bias(conv3d(prob_downscaled, fmaps=prob_conv_channels, kernel=1, gain=1, use_wscale=use_wscale))
                    x = tf.concat([x, prob_downscaled_conv], axis=1)
                with tf.variable_scope('Add_Wellfc'):
                    wellfc_downscaled = wellfc_downscale3d_process(wellfc, factor=int(2**(6-res)))
                    wellfc_downscaled_conv = apply_bias(conv3d(wellfc_downscaled, fmaps=wellfc_conv_channels, kernel=1, gain=1, use_wscale=use_wscale))
                    x = tf.concat([x, wellfc_downscaled_conv], axis=1)
                with tf.variable_scope('Conv0'):
                    x = PN(act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale))))
            return x
        
    def torgb(x, res): 
        lod = 6 - res  
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv3d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))

    # Linear structure: simple but inefficient.
    if structure == 'linear':
        x = block(combo_in, probcubes_in, wellfacies_in, 2)
        cubes_out = torgb(x, 2)
        for res in range(3, 6 + 1):  
            lod = 6 - res  
            x = block(x, probcubes_in, wellfacies_in, res)
            cube = torgb(x, res)
            cubes_out = upscale3d(cubes_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                cubes_out = lerp_clip(cube, cubes_out, lod_in - lod)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(x, prob, wellfc, res, lod):
            y = block(x, prob, wellfc, res)
            cube = lambda: upscale3d(torgb(y, res), 2**lod)
            if res > 2: cube = cset(cube, (lod_in > lod), lambda: upscale3d(lerp(torgb(y, res), upscale3d(torgb(x, res - 1)), lod_in - lod), 2**lod))
            if lod > 0: cube = cset(cube, (lod_in < lod), lambda: grow(y, prob, wellfc, res + 1, lod - 1))
            return cube()
        cubes_out = grow(combo_in, probcubes_in, wellfacies_in, 2, 6 - 2)  
        
    assert cubes_out.dtype == tf.as_dtype(dtype)
    cubes_out = tf.identity(cubes_out, name='cubes_out')
    return cubes_out

#----------------------------------------------------------------------------
# Discriminator network used in the paper.

def D_paper(
    cubes_in,                           # Input: cubes [minibatch, channel, lenght, height, width].
    num_channels        = 1,            # Number of input cube channels. Overridden based on dataset.
    resolution          = 64,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 2048,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv3d + downscale3d, False = separate downscale3d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    cubes_in.set_shape([None, num_channels, resolution, resolution, resolution])
    cubes_in = tf.cast(cubes_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv3d_downscale3d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv3d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale3d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv3d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=24, use_wscale=use_wscale)))  # fmaps=nf(res-2)
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        cube = cubes_in
        x = fromrgb(cube, resolution_log2)
        for res in range(resolution_log2, 2, -1):  # 
            lod = resolution_log2 - res
            x = block(x, res)
            cube = downscale3d(cube)
            y = fromrgb(cube, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale3d(cubes_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale3d(cubes_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out

#----------------------------------------------------------------------------