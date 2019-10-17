# Unet implementation based on https://github.com/jocicmarko/ultrasound-nerve-segmentation
# Major changes:
# - Downsizing the images to 128x128 and then resizing back to 512x512 for submission.
# - Added stand-alone activation layers and batch normalization after each of them.
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Reshape, Lambda, Dropout
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import multi_gpu_model
from keras.utils import Sequence
from .structures import *

from skimage.transform import resize
from time import time
from os import path, mkdir
import argparse
import keras.backend as K
import logging
import numpy as np
import pickle
import tifffile as tiff

import sys
sys.path.append('.')
from src.utils.runtime import funcname, gpu_selection
from src.utils.model import dice_coef, jaccard_coef, jaccard_coef_int, KerasHistoryPlotCallback, KerasSimpleLoggerCallback, \
    dice_coef_loss, tversky_loss, wce_loss, wce_plus_dice_loss, \
    dice_loss_both, wce_plus_tversky_loss, binary_focal_loss, dice_loss_zero, focal_plus_dice_loss, \
    focal_tversky_loss, focal_plus_tversky_loss
from src.utils.data import random_transforms, normalization2


class UNet():

    def __init__(self):

        self.config = {
            'checkpoint_path_net': None,   # save path weight
            'checkpoint_path_config': None,   # pickle save this self.config 
            'checkpoint_path_history': None,   # pickle save history and callback save history graph
            'data_path': 'data',
            'img_shape': (512, 512),
            'input_shape': (256, 256, 1),
            'output_shape': (256, 256, 1),
            'output_shape_onehot': (256, 256, 2),
            'prop_trn': 30. / 30.,  # from beginning train has 30 images
            'prop_val':  30. / 30.,  # from endding, valid has 30 images
            'montage_trn_shape': (5, 6),  # stich together as montage
            'montage_val_shape': (6, 5),
            'transform_train': True,
            'batch_size': 16,
            'nb_epoch': 100,
            'seed': 42,
            'nb_gpu': 1,
            'early_stop_patience': 15,
            'aug': False,
            'steps': 1000,
            'random_split': False
        }

        self.net = None
        self.imgs_montage_trn = None
        self.msks_montage_trn = None
        self.imgs_montage_val = None
        self.msks_montage_val = None
        self.history = None
        self.checkpoint_name = 'checkpoints/unet_21_%d' % self.config['input_shape'][0]

        return

    @property
    def cp_name(self):
        return self.checkpoint_name
        # 'checkpoints/unet_21_%d' % self.config['input_shape'][0]
        # return 'checkpoints/1'

    @cp_name.setter
    def cp_name(self, value):
        self.checkpoint_name = value

    def load_data(self):
        """output range [0,255]
        """
        logger = logging.getLogger(funcname())
        logger.info('Reading images from %s.' % self.config['data_path'])
        
        imgs = tiff.imread('%s/train-volume.tif' % self.config['data_path'])
        msks = tiff.imread('%s/train-labels.tif' % self.config['data_path']).round()

        nb_trn = int(len(imgs) * self.config['prop_trn'])
        nb_val = int(len(imgs) * self.config['prop_val'])
        idx = np.arange(len(imgs))

        # Randomize selection for training and validation.
        if self.config['random_split']:
            np.random.shuffle(idx)
            idx_trn, idx_val = idx[:nb_trn], idx[-nb_val:]
        else:
            idx_trn, idx_val = idx[-nb_trn:], idx[:nb_val]
            np.random.shuffle(idx_trn)
            np.random.shuffle(idx_val)

        print(idx_trn)
        print(idx_val)
        
        H, W = self.config['img_shape']
        logger.info('Combining images and masks into montages.')
        imgs_trn, msks_trn = imgs[idx_trn], msks[idx_trn]
        nb_row, nb_col = self.config['montage_trn_shape']
        assert nb_row * nb_col == len(imgs_trn) == len(msks_trn)
        self.imgs_montage_trn = np.empty((nb_row * H, nb_col * W))
        self.msks_montage_trn = np.empty((nb_row * H, nb_col * W))
        imgs_trn, msks_trn = iter(imgs_trn), iter(msks_trn)
        for y0 in range(0, nb_row * H, H):
            for x0 in range(0, nb_col * W, W):
                y1, x1 = y0 + H, x0 + W
                self.imgs_montage_trn[y0:y1, x0:x1] = next(imgs_trn)
                self.msks_montage_trn[y0:y1, x0:x1] = next(msks_trn)

        logger.info('Combining validation images and masks into montages')
        imgs_val, msks_val = imgs[idx_val], msks[idx_val]
        nb_row, nb_col = self.config['montage_val_shape']
        assert nb_row * nb_col == len(imgs_val) == len(msks_val)
        self.imgs_montage_val = np.empty((nb_row * H, nb_col * W))
        self.msks_montage_val = np.empty((nb_row * H, nb_col * W))
        imgs_val, msks_val = iter(imgs_val), iter(msks_val)
        for y0 in range(0, nb_row * H, H):
            for x0 in range(0, nb_col * W, W):
                y1, x1 = y0 + H, x0 + W
                self.imgs_montage_val[y0:y1, x0:x1] = next(imgs_val)
                self.msks_montage_val[y0:y1, x0:x1] = next(msks_val)

        # Correct the types.
        self.imgs_montage_trn = self.imgs_montage_trn.astype(np.float32)
        self.msks_montage_trn = self.msks_montage_trn.astype(np.uint8)
        self.imgs_montage_val = self.imgs_montage_val.astype(np.float32)
        self.msks_montage_val = self.msks_montage_val.astype(np.uint8)

        return

    def _img_preprocess(self, img):
        """set image value in range [-1, 1]
        """
        img -= np.min(img) # [ 0, ?]
        img /= np.max(img) # [ 0, 1]
        img *= 2           # [ 0, 2]
        img -= 1           # [-1, 1]
        return img      

    def batch_gen(self, imgs, msks, batch_size, transform=False, infinite=False, re_seed=False):
        """
        infinite: False means only output once
        re_seed: every time train on different set of data
        """
        assert imgs.dtype == np.float32
        if msks is None:
            msks = np.random.rand(imgs.shape).round().astype(np.uint8)
        else:
            msks = (msks > 0).astype('uint8')
            assert msks.dtype == np.uint8
            assert np.min(msks) == 0 and np.max(msks) == 1, "Masks should be in [0,1]."
            assert len(np.unique(msks)) == 2, "Masks should be binary."

        X_batch = np.empty((batch_size,) + self.config['input_shape'])
        Y_batch = np.empty((batch_size,) + self.config['output_shape'])
        H, W = imgs.shape
        wdw_H, wdw_W, _ = self.config['input_shape']

        while True:
            if re_seed:
                np.random.seed(int(time()) + np.random.randint(0, 2**16))
            for batch_idx in range(batch_size):
                # Pick a random window to sample.
                y0, x0 = np.random.randint(0, H - wdw_H), np.random.randint(0, W - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W
                img_wdw = imgs[y0:y1, x0:x1].copy()
                msk_wdw = msks[y0:y1, x0:x1].copy()

                if transform:
                    [img_wdw, msk_wdw] = random_transforms([img_wdw, msk_wdw], nb_max=3, aug=self.config['aug'])  # transform <5 times
                    # img_wdw = self._img_preprocess(img_wdw)
                    img_wdw = normalization2(img_wdw)
                img_wdw = self._img_preprocess(img_wdw)

                X_batch[batch_idx] = img_wdw.reshape(self.config['input_shape'])
                Y_batch[batch_idx] = msk_wdw.reshape(self.config['output_shape'])

            # assert np.min(X_batch) == -1
            assert np.max(X_batch) == 1
            assert len(np.unique(Y_batch)) <= 2
            yield (X_batch, Y_batch)

            if not infinite:
                break

    def batch_gen_submit(self, img_stack):
        nb_imgs, img_H, img_W = img_stack.shape
        wdw_H, wdw_W, _ = self.config['input_shape']
        nb_wdws = (img_W / wdw_W) * (img_H / wdw_H)  # float?
        X_batch = np.empty((int(nb_wdws) * nb_imgs, ) + self.config['input_shape'])
        coords = []
        for img_idx, img in enumerate(img_stack):
            for y0 in range(0, img_H, wdw_H):
                for x0 in range(0, img_W, wdw_W):
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    coords.append((img_idx, y0, y1, x0, x1))
                    X_batch[len(coords) - 1] = img[y0:y1, x0:x1].reshape(self.config['input_shape'])

        X_batch -= np.min(X_batch)
        X_batch /= np.max(X_batch)
        X_batch *= 2
        X_batch -= 1

        assert np.min(X_batch) == -1
        assert np.max(X_batch) == 1

        return X_batch, coords

    def compile(self, multi_gpu=False):
        K.set_image_dim_ordering('tf')
        inputs = Input(shape=self.config['input_shape'])

        # conv7 = unet3(inputs, up_depth=256, drop=True)
        conv7 = unet3_deconv(inputs, up_depth=256, drop=True)

        # conv7 = unet3_BN(inputs, up_depth=256, drop=True)
        # conv7 = unet3_aspp(inputs, up_depth=256, drop=True)

        # conv7 = unet4(inputs) 

        conv10 = Conv2D(1, (1, 1))(conv7)  # channel is 1 for sigmoid
        output = Activation('sigmoid')(conv10)      

        self.net = Model(inputs=inputs, outputs=output)
        if multi_gpu:
            self.net = ModelMGPU(self.net, gpus=2) 

        self.net.compile(optimizer=Adam(lr=1e-4),
                        #  loss=dice_loss_both,
                         loss=wce_plus_tversky_loss,
                         metrics=[dice_coef, jaccard_coef]) 

        return

    def train(self):
        logger = logging.getLogger(funcname())

        gen_trn = self.batch_gen(imgs=self.imgs_montage_trn, msks=self.msks_montage_trn, infinite=True, re_seed=True,
                                 batch_size=self.config['batch_size'], transform=self.config['transform_train'])
        gen_val = self.batch_gen(imgs=self.imgs_montage_val, msks=self.msks_montage_val, infinite=True, re_seed=True,
                                 batch_size=self.config['batch_size'])

        cb = []
        cb.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=3, cooldown=5, min_lr=1e-8, verbose=1))
        cb.append(EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                patience=self.config['early_stop_patience'], verbose=1, mode='min'))
        cb.append(ModelCheckpoint(self.checkpoint_name + '_val_loss.net',
                                  monitor='val_loss', save_best_only=True, verbose=1))
        cb.append(ModelCheckpoint(self.checkpoint_name + '_trn_loss.net',
                                  monitor='loss', save_best_only=True, verbose=1))
        cb.append(TensorBoard(log_dir=self.checkpoint_name, 
                                      histogram_freq=0, 
                                      batch_size=1, 
                                      write_graph=True, 
                                      write_grads=False, write_images=True,
                                      update_freq='epoch'))
        history_plot_cb = KerasHistoryPlotCallback()
        history_plot_cb.file_name = self.checkpoint_name + '.history.png'
        cb.append(history_plot_cb)

        logger.info('Training for %d epochs.' % self.config['nb_epoch'])

        result = self.net.fit_generator(
            generator=gen_trn,
            steps_per_epoch=self.config['steps'],
            # samples_per_epoch=max(self.config['batch_size'] * 50, 2048),
            validation_data=gen_val,
            validation_steps=100,
            # nb_val_samples=max(self.config['batch_size'] * 25, 1024),
            epochs=self.config['nb_epoch'],
            callbacks=cb,
            # initial_epoch=0,
            # class_weight='auto', #??? what is this
            verbose=1
        )

        self.history = result.history
        if self.config['checkpoint_path_history'] != None:
            logger.info('Saving history to %s.' % self.config['checkpoint_path_history'])
            f = open(self.config['checkpoint_path_history'], 'wb')
            pickle.dump(self.history, f)
            f.close()

        return

    def evaluate(self):
        np.random.seed(777)
        data_gen = self.batch_gen(imgs=self.imgs_montage_val, msks=self.msks_montage_val,
                                  batch_size=self.config['batch_size'])
        X, Y = next(data_gen)
        metrics = self.net.evaluate(X, Y, verbose=1, batch_size=self.config['batch_size'])
        return zip(self.net.metrics_names, metrics)

    def save_config(self):
        logger = logging.getLogger(funcname())

        if self.config['checkpoint_path_config']:
            logger.info('Saving model config to %s.' % self.config['checkpoint_path_config'])
            f = open(self.config['checkpoint_path_config'], 'wb')
            pickle.dump(self.config, f)
            f.close()

        return

    def load_config(self, checkpoint_path):
        f = open(checkpoint_path, 'rb')
        config = pickle.load(f)
        f.close()
        self.config = config
        return


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def train(args):

    logger = logging.getLogger(funcname())

    model = UNet()
    model.config['checkpoint_path_config'] = model.checkpoint_name + '.config'
    model.config['checkpoint_path_history'] = model.checkpoint_name + '.history'
    model.config['transform_train'] = True
    model.config['nb_epoch'] = 250

    np.random.seed(model.config['seed'])
    model.load_data()
    model.save_config()
    model.compile()
    model.net.summary()
    if args['net']:
        logger.info('Loading saved weights from %s.' % args['net'])
        model.net.load_weights(args['net'])

    model.train()
    logger.info(model.evaluate())
    model.save_config()
    return


def submit(args):
    logger = logging.getLogger(funcname())

    model = UNet()

    if args['model']:
        logger.info('Loading model from %s.' % args['model'])
        model.load_config(args['model'])

    # Get the checkpoint name before tweaking input shape, etc.
    chkpt_name = model.checkpoint_name

    model.config['input_shape'] = model.config['img_shape'] + model.config['input_shape'][-1:]
    model.config['output_shape'] = model.config['img_shape'] + model.config['output_shape'][-1:]
    model.config['output_shape_onehot'] = model.config['img_shape'] + model.config['output_shape_onehot'][-1:]

    model.compile()
    model.net.summary()

    if args['net']:
        logger.info('Loading saved weights from %s.' % args['net'])
        model.net.load_weights(args['net'])

    logger.info('Loading testing images...')
    img_stack = tiff.imread('data/test-volume.tif') 
    X_batch, coords = model.batch_gen_submit(img_stack)

    logger.info('Making predictions on batch...')
    prd_batch = model.net.predict(X_batch, batch_size=model.config['batch_size'])

    logger.info('Reconstructing images...')
    prd_stack = np.empty(img_stack.shape)
    for prd_wdw, (img_idx, y0, y1, x0, x1) in zip(prd_batch, coords):
        prd_stack[img_idx, y0:y1, x0:x1] = prd_wdw.reshape(y1 - y0, x1 - x0)
    prd_stack = prd_stack.astype('float32')

    logger.info('Saving full size predictions...')
    tiff.imsave(chkpt_name + '.submission.tif', prd_stack)
    logger.info('Done - saved file to %s.' % (chkpt_name + '.submission.tif'))


def main():

    logging.basicConfig(level=logging.INFO)

    prs = argparse.ArgumentParser()
    prs.add_argument('--train', help='train', action='store_true')
    prs.add_argument('--submit', help='submit', action='store_true')
    prs.add_argument('--net', help='path to network weights', type=str)
    prs.add_argument('--model', help='path to serialized model', type=str)
    prs.add_argument('--gpu', help='gpu visible device', type=str, default='1')
    args = vars(prs.parse_args())

    gpu_selection(visible_devices=args['gpu'])

    if args['train']:
        train(args)

    elif args['submit']:
        submit(args)


if __name__ == "__main__":
    main()
