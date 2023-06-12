import tensorflow as tf

from graphs.adversarial.AAE_graph import inference_discriminate_encode_fn
from graphs.builder import layer_stuffing, clone_model
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder
from training.callbacks.early_stopping import EarlyStopping
from utils.swe.codes import copy_fn


class AAE(autoencoder):
    def __init__(
            self,
            strategy=None,
            **kwargs
    ):
        self.strategy = strategy
        autoencoder.__init__(
            self,
            **kwargs
        )
        self.ONES = tf.ones(shape=[self.batch_size, 1])
        self.ZEROS = tf.zeros(shape=[self.batch_size, 1])

        self.adversarial_models = {
            'inference_discriminator_real':
                {
                    'variable': None,
                    'adversarial_item': 'generative',
                    'adversarial_value': self.ONES
                },
            'inference_discriminator_fake':
                {
                    'variable': None,
                    'adversarial_item': 'generative',
                    'adversarial_value': self.ZEROS
                },
            'inference_generator_fake':
                {
                    'variable': None,
                    'adversarial_item': 'generative',
                    'adversarial_value': self.ONES
                }
        }

    # combined models special
    def adversarial_get_variables(self):
        return {**self.ae_get_variables(), **self.get_discriminators()}


    def get_discriminators(self):
        return {k: model['variable'] for k, model in self.adversarial_models.items()}

    def create_batch_cast(self, models):
        def batch_cast_fn(batch):
            if self.input_kw:
                x = tf.cast(batch[self.input_kw], dtype=tf.float32) / self.input_scale
            else:
                x = tf.cast(batch, dtype=tf.float32) / self.input_scale
            outputs_dict =  {k+'_outputs': model['adversarial_value'] for k, model in models.items()}
            outputs_dict = {'x_logits': x, **outputs_dict}

            encoded = autoencoder.__encode__(self, inputs={'inputs': x})
            return {'inference_inputs': x, 'generative_inputs': encoded['z_latents'] }, outputs_dict

        return batch_cast_fn

    # override function
    def fit(
            self,
            x,
            validation_data=None,
            **kwargs
    ):
        print()
        print(f'training {autoencoder}')
        # 1- train the basic basicAE
        autoencoder.fit(
            self,
            x=x,
            validation_data=validation_data,
            **kwargs
        )


        def create_discriminator():
            for model in self.get_variables().values():
                layer_stuffing(model)

            for k, model in self.adversarial_models.items():
                model['variable'] = clone_model(old_model=self.get_variables()[model['adversarial_item']],  new_name=k,
                                                restore=self.filepath)

        # 2- create a latents discriminator
        if self.strategy:
            with self.strategy:
                create_discriminator()
        else:
            create_discriminator()

        # 3- clone autoencoder variables
        self.ae_get_variables = copy_fn(self.get_variables)

        # 4- switch to discriminate
        if self.strategy:
            if self.strategy:
                self.discriminators_compile()
        else:
            self.discriminators_compile()

        verbose = kwargs.pop('verbose')
        callbacks = kwargs.pop('callbacks')
        kwargs.pop('input_kw')

        for k, model in self.adversarial_models.items():
            print()
            print(f'training {k}')
            # 5- train the latents discriminator
            model['variable'].fit(
                x=x.map(self.create_batch_cast({k: model})),
                validation_data=None if validation_data is None else validation_data.map(self.create_batch_cast({k: model})),
                callbacks=[EarlyStopping()],
                verbose=1,
                **kwargs
            )

        kwargs['verbose'] = verbose
        kwargs['callbacks'] = callbacks

        # 6- connect all for inference_adversarial training
        if self.strategy:
            if self.strategy:
                self.__models_init__()
        else:
            self.__models_init__()

        print()
        print('training adversarial models')
        cbs = [cb for cb in callbacks or [] if isinstance(cb, tf.keras.callbacks.CSVLogger)]
        for cb in cbs:
            cb.filename = cb.filename.split('.csv')[0] + '_together.csv'
            mertic_names = [fn for sublist in [[k + '_' + fn.__name__ for fn in v] for k, v in self.ae_metrics.items()]
                            for fn in sublist]
            cb.keys = ['loss'] + [fn+'_loss' for fn in self._AA.output_names] + mertic_names
            cb.append_header = cb.keys

        # 7- training together
        self._AA.fit(
            x=x.map(self.create_batch_cast(self.adversarial_models)),
            validation_data=None if validation_data is None else \
                validation_data.map(self.create_batch_cast(self.adversarial_models)),
            **kwargs
        )

    def __models_init__(self):
        self.get_variables = self.adversarial_get_variables
        self.encode_fn = inference_discriminate_encode_fn
        inputs_dict= {
            'inputs': self.get_variables()['inference'].inputs[0]
        }
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(encoded['z_latents'])

        outputs_dict = {k+'_predictions': encoded[k+'_predictions'] for k in self.adversarial_models.keys()}
        outputs_dict = {'x_logits': x_logits, **outputs_dict}

        self._AA = tf.keras.Model(
            name='adverasarial_model',
            inputs= inputs_dict,
            outputs=outputs_dict
        )

        for i, outputs_dict in enumerate(self._AA.output_names):
            if 'x_logits' in outputs_dict:
                self._AA.output_names[i] = 'x_logits'
            for k in self.adversarial_models.keys():
                if k in outputs_dict:
                    self._AA.output_names[i] = k+'_outputs'

        generator_weight = self.adversarial_weights['generator_weight']
        discriminator_weight = self.adversarial_weights['discriminator_weight']
        generator_losses = [k for k in self.adversarial_losses.keys() if 'generator' in k]
        dlen = len(self.adversarial_losses)-len(generator_losses)
        aeloss_weights = {k: (1-discriminator_weight)*(1-generator_weight)/len(self.ae_losses) for k in self.ae_losses.keys()}
        gloss_weights = {k: (1-discriminator_weight)*(generator_weight)/len(generator_losses) for k in generator_losses}
        discriminator_weights = {k:  discriminator_weight/dlen for k in self.adversarial_losses.keys() if k not in generator_losses}
        adversarial_losses = {k: fn() for k, fn in self.adversarial_losses.items() }
        self._AA.compile(
            optimizer=self.optimizer,
            loss={**self.ae_losses, **adversarial_losses},
            metrics=self.ae_metrics,
            loss_weights={**aeloss_weights, **gloss_weights, **discriminator_weights}
        )

        self._AA.generate_sample = self.generate_sample
        self._AA.get_variable = self.get_variable
        self._AA.inputs_shape = self.get_inputs_shape()
        self._AA.get_inputs_shape = self.get_inputs_shape
        self._AA.latents_dim = self.latents_dim
        self._AA.save = self.save

        print(self._AA.summary())

    # override function
    def compile(
            self,
            adversarial_losses,
            adversarial_weights,
            **kwargs
    ):
        self.adversarial_losses=adversarial_losses
        self.adversarial_weights=adversarial_weights
        autoencoder.compile(
            self,
            **kwargs
        )

    def discriminators_compile(self, **kwargs):
        for k, model in self.adversarial_models.items():
            model['variable'].compile(
                optimizer=self.optimizer,
                loss=self.adversarial_losses[k+'_outputs']()
            )

            print(model['variable'].summary())


import os

import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam

from evaluation.quantitive_metrics.metrics import create_metrics
from graphs.basics.AE_graph import create_graph, create_losses, encode_fn, decode_fn, generate_sample
from graphs.builder import load_models, save_models

class autoencoder(tf.keras.Model):
    def __init__(
            self,
            name,
            latents_dim,
            variables_params,
            batch_size=32,
            filepath=None,
            model_fn=create_graph,
            encode_fn=encode_fn,
            **kwargs
    ):
        self.get_variables = model_fn(
            name=name,
            variables_params=variables_params,
            restore=filepath
        )

        self._name = name
        self.filepath = filepath
        self.latents_dim = latents_dim
        self.batch_size = batch_size
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.generate_sample = generate_sample
        self.save_models = save_models
        self.load_models = load_models
        self.__init_autoencoder__(**kwargs)
        self.__rename_outputs__()

    def get_variable(self, var_name, param):
        return self.get_variables()[var_name](*param)

    def __init_autoencoder__(self, **kwargs):
        # connect the graph x' = decode(encode(x))
        inputs_dict= {k: v.inputs[0] for k, v in self.get_variables().items() if k == 'inference'}
        latents = self.__encode__(inference_inputs=inputs_dict)
        x_logits = self.decode(latents)
        outputs_dict =  [x_logits]

        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=inputs_dict,
            outputs=outputs_dict,
            **kwargs
        )

    def __rename_outputs__(self):
        # rename the outputs
        self.output_names = ['x_logits']

    def get_flat_shape(self):
        return (self.batch_size, ) + self.get_variables()['generative'].outputs[0].shape[1:][-3:]

    # override function
    def compile(
            self,
            optimizer=RectifiedAdam(),
            loss=None,
            **kwargs
    ):

        ae_losses = create_losses()
        loss = loss or {}
        for k in loss:
            ae_losses.pop(k)
        self.ae_losses = {**ae_losses, **loss}

        if 'metrics' in kwargs.keys():
            self.ae_metrics = kwargs.pop('metrics', None)
        else:
            self.ae_metrics = create_metrics(self.get_flat_shape())

        tf.keras.Model.compile(self, optimizer=optimizer, loss=self.ae_losses, metrics=self.ae_metrics, **kwargs)
        print(self.summary())

    # override function
    def fit(
            self,
            x,
            y=None,
            input_kw='images',
            steps_per_epoch=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            validation_steps=None,
            validation_split=0.0,
            validation_freq=1,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0
    ):
        self.input_kw = input_kw
        return \
            tf.keras.Model.fit(
                self,
                x=x.map(self.batch_cast),
                y=y,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=None if validation_data is None else validation_data.map(self.batch_cast),
                validation_steps=validation_steps,
                validation_freq=validation_freq,
                validation_split=validation_split,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                shuffle=shuffle,
                initial_epoch=initial_epoch
            )

    # override function
    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        file_Name = os.path.join(filepath, self.name)
        self.save_models(file_Name, self.get_variables())

    def get_inputs_shape(self):
        return list(self.get_variables()['inference'].inputs[0].shape[1:])

    def get_outputs_shape(self):
        return list(self.get_variables()['generative'].outputs[0].shape[1:])

    def __encode__(self, **kwargs):
        if  'inputs' in kwargs:
            # print(kwargs['inputs'])
            # print(kwargs['inputs'].keys())
            inputs = kwargs['inputs']
        else:
            inputs = kwargs['inference_inputs']
        # inputs = kwargs['inference_inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)
        return self.encode_fn(**kwargs)

    # autoencoder function
    def encode(self, x):
        return self.__encode__(inference_inputs={'inference_inputs': x})['z_latents']

    # autoencoder function
    def decode(self, latents):
        return self.decode_fn(model=self.get_variable, latents={'generative_inputs': latents}, output_shape=self.get_outputs_shape())

    # autoencoder function
    def reconstruct(self, images):
        if len(images.shape)==3:
            images = tf.reshape(images, ((1,) + images.shape))
        return tf.sigmoid(self.decode(self.encode(images)))

    # autoencoder function
    def generate_random_images(self, num_images=None):
        num_images = num_images or self.batch_size
        latents_shape = [num_images, self.latents_dim]
        random_latents = tf.random.normal(shape=latents_shape)
        generated = self.generate_sample(model=self.get_variable,
                                         input_shape=self.get_inputs_shape(),
                                         latents_shape=latents_shape,
                                         epsilon=random_latents)
        return generated

    def batch_cast(self, batch):
        if self.input_kw:
            x = batch[self.input_kw]
        else:
            x = batch

        return {
                   'inference_inputs': x,
               }, \
               {
                   'x_logits': x
               }



import tensorflow as tf

from evaluation.unsupervised_metrics.disentangle_api import unsupervised_metrics
from utils.data_and_files.file_utils import log


class DisentanglementUnsuperviedMetrics(tf.keras.callbacks.Callback):
    def __init__(
            self,
            ground_truth_data,
            representation_fn,
            random_state,
            file_Name,
            num_train=1000,
            num_test=200,
            batch_size=32,
            gt_freq=10,
            **kws
    ):
        self.gt_data = ground_truth_data
        self.representation_fn = representation_fn
        self.random_state = random_state
        self.file_Name = file_Name
        self.num_train = num_train
        self.num_test = num_test
        self.batch_size = batch_size
        self.gt_freq = gt_freq
        tf.keras.callbacks.Callback.__init__(self, **kws)

    def on_train_end(self, logs=None):
        self.score_metrics(-999, logs)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.gt_freq == 0:  # or save after some epoch, each k-th epoch etc.
            self.score_metrics(epoch, logs)

    def score_metrics(self, epoch, logs={}):
         us_scores = unsupervised_metrics(
             ground_truth_data=self.gt_data,
             representation_fn=self.representation_fn,
             random_state=self.random_state,
             num_train=self.num_train,
             batch_size=self.batch_size
         )

         gt_metrics = {'Epoch': epoch, **us_scores}
         log(file_name=self.file_Name, message=dict(gt_metrics))




import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam


from graphs.disentangled_inferred_prior.AE_graph import create_regularized_losses
from evaluation.quantitive_metrics.metrics import create_metrics
from training.autoencoding_basic.autoencoders.autoencoder import autoencoder as basicAE
from training.disentangled_inferred_prior.DIP_shared import infer_prior

class Covariance_AE(basicAE):

    # override function
    def compile(
            self,
            optimizer=RectifiedAdam(),
            loss=None,
            **kwargs
    ):

        ae_losses = create_regularized_losses()
        loss = loss or {}
        for k in loss:
            ae_losses.pop(k)
        self.ae_losses = {**ae_losses, **loss}

        if 'metrics' in kwargs.keys():
            self.ae_metrics = kwargs.pop('metrics', None)
        else:
            self.ae_metrics = create_metrics(self.get_flat_shape())

        tf.keras.Model.compile(self, optimizer=optimizer, loss=self.ae_losses, metrics=self.ae_metrics, **kwargs)
        print(self.summary())

    def __encode__(self, **kwargs):
        inputs = kwargs['inputs']
        for k, v in  inputs.items():
            if inputs[k].shape == self.get_inputs_shape():
                inputs[k] = tf.reshape(inputs[k], (1, ) + self.get_inputs_shape())
            inputs[k] = tf.cast(inputs[k], tf.float32)
        kwargs['model']  = self.get_variable
        kwargs['latents_shape'] = (self.batch_size, self.latents_dim)

        encoded = self.encode_fn(**kwargs)
        _, covariance_regularizer = infer_prior(latent_mean=encoded['z_latents'], \
                                                regularize=True, lambda_d=self.lambda_d, lambda_od=self.lambda_od)
        return {**encoded, 'covariance_regularized': covariance_regularizer}

    def __init_autoencoder__(self, **kwargs):
        #  disentangled_inferred_prior configuration
        self.lambda_d = 50
        self.lambda_od = 100

        # connect the graph x' = decode(encode(x))
        inputs_dict= {k: v.inputs[0] for k, v in self.get_variables().items() if k == 'inference'}
        encoded = self.__encode__(inputs=inputs_dict)
        x_logits = self.decode(latents={'z_latents': encoded['z_latents']})
        covariance_regularizer = encoded['covariance_regularized']

        outputs_dict = {
            'x_logits': x_logits,
            'covariance_regularized': covariance_regularizer
        }
        tf.keras.Model.__init__(
            self,
            name=self.name,
            inputs=inputs_dict,
            outputs=outputs_dict,
            **kwargs
        )

    def __rename_outputs__(self):
        # rename the outputs
        ## rename the outputs
        for i, output_name in enumerate(self.output_names):
            if 'x_logits' in output_name:
                self.output_names[i] = 'x_logits'
            elif 'covariance_regularized' in output_name:
                self.output_names[i] = 'covariance_regularized'
            else:
                pass

    def batch_cast(self, batch):
        if self.input_kw:
            x = tf.cast(batch[self.input_kw], dtype=tf.float32)/self.input_scale
        else:
            x = tf.cast(batch, dtype=tf.float32)/self.input_scale

        return {
                   'inference_inputs': x,
               }, \
               {
                   'x_logits': x,
                   'covariance_regularized': 0.0
               }



import glob
import hashlib
import logging
import os
import re
import warnings

from keras.preprocessing.image import ImageDataGenerator

from utils.data_and_files.data_utils import as_bytes
from utils.reporting.logging import log_message
from .image_iterator import ImageIterator


class FileImageGenerator(ImageDataGenerator):
    def flow_from_image_lists(self, image_lists,
                              category,
                              image_dir,
                              target_size,
                              batch_size,
                              episode_len=None,
                              episode_shift=None,
                              color_mode='rgb',
                              class_mode=None,
                              shuffle=True,
                              seed=None,
                              save_to_dir=None,
                              save_prefix='',
                              save_format='jpg'):

        return ImageIterator(image_lists, self,
                             category,
                             image_dir,
                             target_size=target_size,
                             color_mode=color_mode,
                             class_mode=class_mode,
                             data_format=self.data_format,
                             batch_size=batch_size,
                             episode_len=episode_len,
                             episode_shift=episode_shift,
                             shuffle=shuffle, seed=seed,
                             save_to_dir=save_to_dir,
                             save_prefix=save_prefix,
                             save_format=save_format)


def create_image_lists(image_dir, validation_pct, valid_imgae_formats, max_num_images_per_class=2**27-1,
                       sequenced=None, verbose=1):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    # Arguments
        image_dir: string path to a folder containing subfolders of images.
        validation_pct: integer percentage of images reserved for validation.

    # Returns
        dictionary of label subfolder, with images split into training
        and validation sets within each label.
    """
    if not os.path.isdir(image_dir):
        raise ValueError("Image directory {} not found.".format(image_dir))
    image_lists = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]

    sub_dirs_without_root = sub_dirs[1:]  # first element is root directory
    sub_dirs_without_root = sorted(sub_dirs_without_root, key=lambda x: int(x.split(os.sep)[-1]))

    for sub_dir in sub_dirs_without_root:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        if verbose == 1:
            log_message("Looking for images in '{}'".format(dir_name), logging.DEBUG)

        if isinstance(valid_imgae_formats, str):
            valid_imgae_formats = [valid_imgae_formats]

        for extension in valid_imgae_formats:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            msg = 'No files found'
            if verbose == 1:
                log_message(msg, logging.WARN)
            warnings.warn(msg)
            continue
        else:
            if verbose == 1:
                log_message('{} file found'.format(len(file_list)), logging.INFO)
        if len(file_list) < 20:
            msg = 'Folder has less than 20 images, which may cause issues.'
            if verbose == 1:
                log_message(msg, logging.WARN)
            warnings.warn(msg)
        elif len(file_list) > max_num_images_per_class:
            msg='WARNING: Folder {} has more than {} images. Some '\
                          'images will never be selected.' \
                          .format(dir_name, max_num_images_per_class)
            log_message(msg, logging.WARN)
            warnings.warn(msg)
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        validation_images = []
        if sequenced is True:
            #Sequenced in the case of
            try:
                file_list = sorted(file_list, key=lambda x: int(x.split(os.sep)[-1].split('.')[0]))
            except:
                msg = 'WARNING: Sorting folder {} has failed!' \
                    .format(dir_name)
                log_message(msg, logging.WARN)
                warnings.warn(msg)

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if sequenced is True:
                hash_pct = int((int(file_name.split(os.sep)[-1].split('.')[0]) / len(file_list))*100)
            else:
                # Get the hash of the file name and perform variant assignment.
                hash_name = hashlib.sha1(as_bytes(base_name)).hexdigest()
                hash_pct = ((int(hash_name, 16) % (max_num_images_per_class  + 1)) *
                            (100.0 / max_num_images_per_class))
            if hash_pct < validation_pct:
                validation_images.append(base_name)
            else:
                training_images.append(base_name)
        image_lists[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'validation': validation_images,
        }
    return image_lists

def get_generators(images_list, image_dir, image_size, batch_size, class_mode, episode_len=None, episode_shift=None, scaler=255.0):

    train_datagen = FileImageGenerator(rescale=1. / scaler)

    valid_datagen = FileImageGenerator(rescale=1. / scaler)

    train_generator = train_datagen.flow_from_image_lists(
        image_lists=images_list,
        category='training',
        image_dir=image_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        episode_len=episode_len,
        episode_shift=episode_shift,
        seed=0)

    validation_generator = valid_datagen.flow_from_image_lists(
        image_lists=images_list,
        category='validation',
        image_dir=image_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        episode_len=episode_len,
        episode_shift=episode_shift,
        seed=0)

    return train_generator, validation_generator
