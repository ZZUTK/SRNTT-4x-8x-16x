import tensorflow as tf
from .tensorlayer import *
from .tensorlayer.layers import *
from os.path import join, exists, split, isfile
from os import makedirs, environ
from .vgg19 import *
from .swap import *
from glob import glob
from scipy.misc import imread, imresize, imsave
from .download_vgg19_model import *
import logging

# set logging level for TensorFlow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# set logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
    filename='SRNTT.log'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# some global variables
MODEL_FOLDER = 'model'
SAMPLE_FOLDER = 'sample'
SRNTT_MODEL_NAMES = {
    'init': 'srntt_init.npz',
    'conditional_texture_transfer': 'srntt.npz',
    'content_extractor': 'upscale.npz',
    'discriminator': 'discrim.npz',
    'weighted': 'srntt_weighted.npz'
}


class SRNTT(object):

    MAX_IMAGE_SIZE = 2046 ** 2

    def __init__(
            self,
            srntt_model_path='models/SRNTT',
            vgg19_model_path='models/VGG19/imagenet-vgg-verydeep-19.mat',
            save_dir=None,
            num_res_blocks=16,
            scale=8
    ):
        self.srntt_model_path = srntt_model_path
        self.vgg19_model_path = vgg19_model_path
        self.save_dir = save_dir
        self.num_res_blocks = int(num_res_blocks)
        self.is_model_built = False
        self.scale = scale
        download_vgg19(self.vgg19_model_path)

    def model(
            self,
            inputs,     # LR images, in range of [-1, 1]
            maps=None,  # texture feature maps after texture swapping
            weights=None,  # weights of each pixel on the maps
            is_train=True,
            reuse=False,
            concat=False  # concatenate weights to feature
    ):

        # ********************************************************************************
        # *** content extractor
        # ********************************************************************************
        # print('\tcontent extractor')
        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = None
        g_init = tf.random_normal_initializer(1., 0.02)
        with tf.variable_scope("content_extractor", reuse=reuse):
            layers.set_name_reuse(reuse)
            net = InputLayer(inputs=inputs, name='input')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='n64s1/c')
            temp = net
            for i in range(16):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='n64s1/b1/%s' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='n64s1/b2/%s' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='b_residual_add/%s' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
            content_feature = ElementwiseLayer(layer=[net, temp], combine_fn=tf.add, name='add3')

            # upscaling (4x) for texture extractor
            net = Conv2d(net=content_feature, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='n256s1/1')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='n256s1/2')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

            # if self.scale == 8:
            #     net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
            #                  padding='SAME', W_init=w_init, name='n256s1/3')
            #     net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/3')
            #
            # if self.scale == 16:
            #     net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
            #                  padding='SAME', W_init=w_init, name='n256s1/3')
            #     net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/3')
            #     net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
            #                  padding='SAME', W_init=w_init, name='n256s1/4')
            #     net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/4')
            #



            # output value range is [-1, 1]
            net_upscale = Conv2d(net=net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                                 padding='SAME', W_init=w_init, name='out')
            if maps is None:
                return net_upscale, None

        # ********************************************************************************
        # *** conditional texture transfer
        # ********************************************************************************
        with tf.variable_scope("texture_transfer", reuse=reuse):
            layers.set_name_reuse(reuse)
            assert isinstance(maps, (list, tuple))
            # fusion content and texture maps at the smallest scale
            # print('\tfusion content and texture maps at SMALL scale')
            map_in = InputLayer(inputs=content_feature.outputs, name='content_feature_maps')
            if weights is not None and concat:
                self.a1 = tf.get_variable(dtype=tf.float32, name='small/a', initializer=1.)
                self.b1 = tf.get_variable(dtype=tf.float32, name='small/b', initializer=0.)
                map_ref = maps[0] * tf.nn.sigmoid(self.a1 * weights + self.b1)
            else:
                map_ref = maps[0]
            map_ref = InputLayer(inputs=map_ref, name='reference_feature_maps1')
            net = ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation1')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='small/conv1')
            for i in range(self.num_res_blocks):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='small/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='small/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='small/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='small/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='small/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='small/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='small/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='small/add2')
            # upscaling (2x)
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='small/conv3')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='small/subpixel')

            # fusion content and texture maps at the medium scale
            # print('\tfusion content and texture maps at MEDIUM scale')
            map_in = net
            if weights is not None and concat:
                self.a2 = tf.get_variable(dtype=tf.float32, name='medium/a', initializer=1.)
                self.b2 = tf.get_variable(dtype=tf.float32, name='medium/b', initializer=0.)
                map_ref = maps[1] * tf.nn.sigmoid(self.a2 * tf.image.resize_bicubic(
                    weights, [weights.get_shape()[1] * 2, weights.get_shape()[2] * 2]) + self.b2)
            else:
                map_ref = maps[1]
            map_ref = InputLayer(inputs=map_ref, name='reference_feature_maps2')
            net = ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation2')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='medium/conv1')
            for i in range(int(self.num_res_blocks / 2)):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='medium/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='medium/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='medium/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='medium/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='medium/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='medium/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='medium/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='medium/add2')
            # upscaling (2x)
            net = Conv2d(net=net, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='medium/conv3')
            net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='medium/subpixel')

            # fusion content and texture maps at the large scale
            # print('\tfusion content and texture maps at LARGE scale')
            map_in = net
            if weights is not None and concat:
                self.a3 = tf.get_variable(dtype=tf.float32, name='large/a', initializer=1.)
                self.b3 = tf.get_variable(dtype=tf.float32, name='large/b', initializer=0.)
                map_ref = maps[2] * tf.nn.sigmoid(self.a3 * tf.image.resize_bicubic(
                    weights, [weights.get_shape()[1] * 4, weights.get_shape()[2] * 4]) + self.b3)
            else:
                map_ref = maps[2]
            map_ref = InputLayer(inputs=map_ref, name='reference_feature_maps3')
            net = ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation3')
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,
                         padding='SAME', W_init=w_init, name='large/conv1')
            for i in range(int(self.num_res_blocks / 4)):  # residual blocks
                net_ = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='large/resblock_%d/conv1' % i)
                net_ = BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                                      gamma_init=g_init, name='large/resblock_%d/bn1' % i)
                net_ = Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                              padding='SAME', W_init=w_init, b_init=b_init, name='large/resblock_%d/conv2' % i)
                net_ = BatchNormLayer(layer=net_, is_train=is_train,
                                      gamma_init=g_init, name='large/resblock_%d/bn2' % i)
                net_ = ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='large/resblock_%d/add' % i)
                net = net_
            net = Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, b_init=b_init, name='large/conv2')
            net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='large/bn2')
            net = ElementwiseLayer(layer=[net, map_in], combine_fn=tf.add, name='large/add2')
            net = Conv2d(net=net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=None,
                         padding='SAME', W_init=w_init, name='large/conv3')
            # net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='large/bn2')

            if self.scale == 8:
                net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='large/subpixel')

            if self.scale == 16:
                net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='large/subpixel')
                net = Conv2d(net=net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=None,
                             padding='SAME', W_init=w_init, name='final/conv')
                net = SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu, name='final/subpixel')


            # output of SRNTT, range [-1, 1]
            net_srntt = Conv2d(net=net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.nn.tanh,
                               padding='SAME', W_init=w_init, name='out')

            return net_upscale, net_srntt

    def test(
            self,
            input_dir,  # original image
            ref_dir=None,  # reference images
            use_pretrained_model=True,
            use_init_model_only=False,  # the init model is trained only with the reconstruction loss
            use_weight_map=False,
            result_dir=None,
            ref_scale=1.0,
            is_original_image=True,
            max_batch_size=16,
            save_ref=True
    ):
        logging.info('Testing mode')

        assert ref_dir is not None

        # ********************************************************************************
        # *** check input and reference images
        # ********************************************************************************
        # check input_dir
        img_input, img_hr = None, None
        if isinstance(input_dir, np.ndarray):
            assert len(input_dir.shape) == 3
            img_input = np.copy(input_dir)
        elif isfile(input_dir):
            img_input = imread(input_dir, mode='RGB')
        else:
            logging.error('Unrecognized input_dir %s' % input_dir)
            exit(0)

        h, w, _ = img_input.shape
        if is_original_image:
            # ensure that the size of img_input can be divided by 4 with no remainder
            h = int(h // self.scale * self.scale)
            w = int(w // self.scale * self.scale)
            img_hr = img_input[0:h, 0:w, ::]
            img_input = imresize(img_hr, 1. / self.scale, interp='bicubic')
            h, w, _ = img_input.shape
        img_input_copy = np.copy(img_input)

        if h * w * self.scale ** 2 > SRNTT.MAX_IMAGE_SIZE:  # avoid OOM
            # split img_input into patches
            patches = []
            grids = []
            patch_size = 512 // self.scale
            stride = patch_size // 2
            for ind_row in range(0, h - (patch_size - stride), stride):
                for ind_col in range(0, w - (patch_size - stride), stride):
                    patch = img_input[ind_row:ind_row + patch_size, ind_col:ind_col + patch_size, :]
                    if patch.shape != (patch_size, patch_size, 3):
                        patch = np.pad(patch,
                                       ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)),
                                       'reflect')
                    patches.append(patch)
                    grids.append((ind_row * self.scale, ind_col * self.scale, patch_size * self.scale))
            grids = np.stack(grids, axis=0)
            img_input = np.stack(patches, axis=0)
        else:
            grids = None
            img_input = np.expand_dims(img_input, axis=0)

        # check ref_dir
        img_ref = []
        if not isinstance(ref_dir, (list, tuple)):
            ref_dir = [ref_dir]

        for ref in ref_dir:
            if isinstance(ref, np.ndarray):
                assert len(ref.shape) == 3
                img_ref.append(np.copy(ref))
            elif isfile(ref):
                img_ref.append(imread(ref, mode='RGB'))
            else:
                logging.error('Unrecognized ref_dir type!')
                exit(0)

        if ref_scale <= 0:  # keep the same scale as HR image
            img_ref = [imresize(img, (h * self.scale, w * self.scale), interp='bicubic') for img in img_ref]
        elif ref_scale != 1:
            img_ref = [imresize(img, float(ref_scale), interp='bicubic') for img in img_ref]

        for i in xrange(len(img_ref)):
            h2, w2, _ = img_ref[i].shape
            h2 = int(h2 // self.scale * self.scale)
            w2 = int(w2 // self.scale * self.scale)
            img_ref[i] = img_ref[i][0:h2, 0:w2, ::]

        # create result folder
        if result_dir is None:
            result_dir = join(self.save_dir, 'test')
        if not exists(result_dir):
            makedirs(result_dir)
        if not exists(join(result_dir, 'tmp')):
            makedirs(join(result_dir, 'tmp'))

        # ********************************************************************************
        # *** build graph
        # ********************************************************************************
        if not self.is_model_built:
            self.is_model_built = True
            logging.info('Building graphs ...'.format(self.scale))
            # input image, range [-1, 1]
            self.input_srntt = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)

            # reference images, range [0, 255]
            self.input_vgg19 = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)


            # swapped feature map and weights
            self.maps = (
                tf.placeholder(
                    dtype=tf.float32,
                    shape=(1, None, None, 256)),
                tf.placeholder(
                    dtype=tf.float32,
                    shape=(1, None, None, 128)),
                tf.placeholder(
                    dtype=tf.float32,
                    shape=(1, None, None, 64))
            )

            self.weights = tf.placeholder(
                dtype=tf.float32,
                shape=(1, None, None))

            # SRNTT network
            logging.info('Build SRNTT {}x model'.format(self.scale))
            if use_weight_map:
                self.net_upscale, self.net_srntt = self.model(
                    self.input_srntt, self.maps, weights=tf.expand_dims(self.weights, axis=-1), is_train=False)
            else:
                self.net_upscale, self.net_srntt = self.model(self.input_srntt, self.maps, is_train=False)

            # VGG19 network, input range [0, 255]
            logging.info('Build VGG19 model')
            self.net_vgg19 = VGG19(
                input_image=self.input_vgg19,
                model_path=self.vgg19_model_path,
                final_layer='relu3_1'
            )

            # ********************************************************************************
            # *** load models
            # ********************************************************************************
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = False
            self.sess = tf.Session(config=config)

            # instant of Swap()
            logging.info('Initialize the swapper')
            self.swaper = Swap(sess=self.sess)

            logging.info('Loading models ...')
            self.sess.run(tf.global_variables_initializer())

            # load pre-trained content extractor, including upscaling.
            model_path = join('_'.join(self.srntt_model_path.split('_')[:-1])+'_4x', SRNTT_MODEL_NAMES['content_extractor'])
            if files.load_and_assign_npz(
                    sess=self.sess,
                    name=model_path,
                    network=self.net_upscale) is False:
                logging.error('FAILED load %s' % model_path)
                exit(0)

            # load the conditional texture transfer model
            if use_init_model_only:
                model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['init'])
                if files.load_and_assign_npz(
                        sess=self.sess,
                        name=model_path,
                        network=self.net_srntt):
                    logging.info('SUCCESS load %s' % model_path)
                else:
                    logging.error('FAILED load %s' % model_path)
                    exit(0)
            else:
                model_path = join(self.srntt_model_path, SRNTT_MODEL_NAMES['conditional_texture_transfer'])
                if files.load_and_assign_npz(
                        sess=self.sess,
                        name=model_path,
                        network=self.net_srntt):
                    logging.info('SUCCESS load %s' % model_path)
                else:
                    logging.error('FAILED load %s' % model_path)
                    exit(0)

        logging.info('**********'
                     ' Start testing '
                     '**********')

        matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']

        logging.info('Get VGG19 Feature Maps')

        logging.info('\t[1/2] Getting feature map of Ref image ...')
        t_start = time.time()
        map_ref = []
        for i in img_ref:
            map_ref.append(
                self.net_vgg19.get_layer_output(
                    sess=self.sess, layer_name=matching_layer,
                    feed_image=imresize(i, (int(h2 * 4 / self.scale), int(w2 * 4 / self.scale)), interp='bicubic'))
            )
        styles = [[] for _ in xrange(len(matching_layer))]
        for i in map_ref:
            for j in xrange(len(styles)):
                styles[j].append(i[j])

        logging.info('\t[2/2] Getting feature map of LR->SR Ref image ...')
        map_ref_sr = []
        for i in img_ref:
            img_ref_downscale = imresize(i, 1. / self.scale, interp='bicubic')
            img_ref_upscale = imresize(img_ref_downscale, 4., interp='bicubic')
            map_ref_sr.append(
                self.net_vgg19.get_layer_output(
                    sess=self.sess, layer_name=matching_layer[0],
                    feed_image=img_ref_upscale)
            )

        # swap ref to in
        logging.info('Patch-Wise Matching and Swapping')
        for idx, patch in enumerate(img_input):
            logging.info('\tPatch %03d/%03d' % (idx + 1, img_input.shape[0]))

            # skip if the results exists
            if exists(join(result_dir, 'tmp', 'srntt_%05d.png' % idx)):
                logging.warn('\tPatch result already exists. Skip this step. '
                             '(Please remove current result folder or assign a new result dir)')
                continue

            logging.info('\tGetting feature map of input LR image ...')
            img_input_upscale = imresize(patch, 4., interp='bicubic')
            map_sr = self.net_vgg19.get_layer_output(
                sess=self.sess, layer_name=matching_layer[0], feed_image=img_input_upscale)

            logging.info('\tMatching and swapping features ...')
            map_target, weight, _ = self.swaper.conditional_swap_multi_layer(
                content=map_sr,
                style=styles[0],
                condition=map_ref_sr,
                other_styles=styles[1:],
                is_weight=use_weight_map
            )

            logging.info('Obtain SR patches')
            if use_weight_map:
                weight = np.pad(weight, ((1, 1), (1, 1)), 'edge')
                out_srntt, out_upscale = self.sess.run(
                    fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
                    feed_dict={
                        self.input_srntt: [patch / 127.5 - 1],
                        self.maps: [np.expand_dims(m, axis=0) for m in map_target],
                        self.weights: [weight]
                    }
                )
            else:
                time_step_1 = time.time()
                out_srntt, out_upscale = self.sess.run(
                    fetches=[self.net_srntt.outputs, self.net_upscale.outputs],
                    feed_dict={
                        self.input_srntt: [patch / 127.5 - 1],
                        self.maps: [np.expand_dims(m, axis=0) for m in map_target],
                    }
                )
                time_step_2 = time.time()

                logging.info('Time elapsed: PM: %.3f sec, SR: %.3f sec' %
                             ((time_step_1 - t_start), (time_step_2 - time_step_1)))


            imsave(join(result_dir, 'tmp', 'srntt_%05d.png' % idx),
                   np.round((out_srntt.squeeze() + 1) * 127.5).astype(np.uint8))
            imsave(join(result_dir, 'tmp', 'upscale_%05d.png' % idx),
                   np.round((out_upscale.squeeze() + 1) * 127.5).astype(np.uint8))
            logging.info('Saved to %s' % join(result_dir, 'tmp', 'srntt_%05d.png' % idx))
        t_end = time.time()
        logging.info('Reconstruct SR image')
        out_srntt_files = sorted(glob(join(result_dir, 'tmp', 'srntt_*.png')))
        out_upscale_files = sorted(glob(join(result_dir, 'tmp', 'upscale_*.png')))

        if grids is not None:
            patch_size = grids[0, 2]
            h_l, w_l = grids[-1, 0] + patch_size, grids[-1, 1] + patch_size
            out_upscale_large = np.zeros((h_l, w_l, 3), dtype=np.float32)
            out_srntt_large = np.copy(out_upscale_large)
            counter = np.zeros_like(out_srntt_large, dtype=np.float32)
            for idx in xrange(len(grids)):
                out_upscale_large[
                grids[idx, 0]:grids[idx, 0] + patch_size,
                grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(out_upscale_files[idx], mode='RGB').astype(np.float32)

                out_srntt_large[
                grids[idx, 0]:grids[idx, 0] + patch_size,
                grids[idx, 1]:grids[idx, 1] + patch_size, :] += imread(out_srntt_files[idx], mode='RGB').astype(np.float32)

                counter[
                grids[idx, 0]:grids[idx, 0] + patch_size,
                grids[idx, 1]:grids[idx, 1] + patch_size, :] += 1

            out_upscale_large /= counter
            out_srntt_large /= counter
            out_upscale = out_upscale_large[:h * 4, :w * 4, :]
            out_srntt = out_srntt_large[:h * 4, :w * 4, :]
        else:
            out_upscale = imread(out_upscale_files[0], mode='RGB')
            out_srntt = imread(out_srntt_files[0], mode='RGB')

        # log run time
        with open(join(result_dir, 'run_time.txt'), 'w') as f:
            line = '%02d min %02d sec\n' % ((t_end - t_start) // 60, (t_end - t_start) % 60)
            f.write(line)
            f.close()

        # save results
        # save HR image if it exists
        if img_hr is not None:
            imsave(join(result_dir, 'HR.png'), img_hr)
        # save LR (input) image
        imsave(join(result_dir, 'LR.png'), img_input_copy)
        # save reference image(s)
        if save_ref:
            for idx, ref in enumerate(img_ref):
                imsave(join(result_dir, 'Ref_%02d.png' % idx), ref)
        # save bicubic
        imsave(join(result_dir, 'Bicubic.png'), imresize(img_input_copy, self.scale * 1., interp='bicubic'))
        # save SR images
        # imsave(join(result_dir, 'Upscale.png'), np.array(out_upscale).squeeze().round().clip(0, 255).astype(np.uint8))
        imsave(join(result_dir, 'SRNTT.png'), np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8))
        logging.info('Saved results to folder %s' % result_dir)

        return np.array(out_srntt).squeeze().round().clip(0, 255).astype(np.uint8)

