from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np

from util import log

from model import Model
from input_ops import create_input_ops, check_data_id
from PIL import Image

import tensorflow as tf
import time
import imageio
import scipy.misc as sm
import numpy

class EvalManager(object):

    def __init__(self):
        # collection of batches (not flattened)
        self._ids = []
        self._predictions = []
        self._groundtruths = []

    def add_batch(self, id, prediction, groundtruth):

        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(id)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    def compute_loss(self, pred, gt):
        return np.sum(np.abs(pred - gt))/np.prod(pred.shape)

    def report(self):
        log.info("Computing scores...")
        total_loss = []

        for id, pred, gt in zip(self._ids, self._predictions, self._groundtruths):
            total_loss.append(self.compute_loss(pred, gt))
        avg_loss = np.average(total_loss)
        log.infov("Average loss : %.4f", avg_loss)


class Evaler(object):
    def __init__(self,
                 config,
                 dataset,
                 dataset_train):
        self.config = config
        self.train_dir = config.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset
        self.dataset_train = dataset_train

        check_data_id(dataset, config.data_id)
        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         data_id=config.data_id,
                                         is_training=False,
                                         shuffle=False)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.total_y = np.concatenate((y_train,y_test))        
        
        # --- create model ---
        self.model = Model(config)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        tf.set_random_seed(123)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint_path = config.checkpoint_path
        if self.checkpoint_path is None and self.train_dir:
            self.checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
        if self.checkpoint_path is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint_path)

    def eval_run(self):
        # load checkpoint///
        if self.checkpoint_path:
            self.saver.restore(self.session, self.checkpoint_path)
            log.info("Loaded from checkpoint!")

        log.infov("Start Inference and Evaluation")

        log.info("# of testing examples = %d", len(self.dataset))
        length_dataset = len(self.dataset)

        max_steps = int(length_dataset / self.batch_size) + 1

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        evaler = EvalManager()

        if not (self.config.recontrain or self.config.interpolate or self.config.generate or self.config.reconstruct):
            raise ValueError('Please specify at least one task by indicating' +
                             '--reconstruct, --generate, or --interpolate.')
            return

        if self.config.recontrain:
            try:
               #r = open('batch_ids_codes_for_few_class_2.txt', "w+")
               #for s in xrange(max_steps):
               loss_z_update = 100000
               s = 0
               #while (loss_z_update > 0.1):
               max_steps = 100000
               min_loss = 10000000 
               for s in xrange(max_steps):
                   step, z,  loss_g_update, loss_z_update, batch_chunk, step_time = \
                       self.run_single_z_update_step(self.batch, self.dataset, step=s, is_train=False)
                   if (loss_z_update < min_loss):
                      min_loss = loss_z_update
                   if s % 10000 == 0: 
                      self.log_train_step_message(step, z,  loss_g_update, loss_z_update, min_loss, step_time)
        #              for i in range(len(batch_chunk['id'])):
                      m, l = self.dataset.get_data(batch_chunk['id'][0])
                      x_a = self.generator(z)
                      imageio.imwrite('generate_z_batch_step_{}.png'.format(s), x_a[0])
                      imageio.imwrite('original_img_from_batch_img_step_{}.png'.format(s), m)
               #r.write("max steps" + str(max_steps))
               #r.close()
 
            except Exception as e:
                coord.request_stop(e)

            log.warning('Completed reconstruction.')
 
        if self.config.reconstruct:
            try:
                for s in xrange(max_steps):
                    step, loss, step_time, batch_chunk, prediction_pred, prediction_gt = \
                        self.run_single_step(self.batch)
                    self.log_step_message(s, loss, step_time)
                    evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)
                    imageio.imwrite('pred_pred_{}.png'.format(s), self.image_grid(prediction_pred))
                    imageio.imwrite('pred_gt_{}.png'.format(s), self.image_grid(prediction_gt)) 
            except Exception as e:
                coord.request_stop(e)

            evaler.report()
            log.warning('Completed reconstruction.')



        if self.config.generate:
            r = open('img_ids_for_small_sample_testtxt', "r")
            if (r.mode == "r"):
             lines = r.readlines()
             z_all = []
             count = 0
             for line in lines:
                 line_split = line.split(":")
                 z_i = line_split[-1]
                 imgi, z_x = self.dataset_train.get_data(z_i.strip())
                 imageio.imwrite('original_img_{}.png'.format(count), imgi) 
                 z_all.append(np.array(z_x))
                 count+=1
             for idx in range(len(z_all) - 1):
                 z_a = np.sum([z_all[idx]*0.3,  z_all[idx+1]*0.7], axis=0)
                 z_b = np.sum([z_all[idx]*0.5,  z_all[idx+1]*0.5], axis=0)
                 z_c = np.sum([z_all[idx]*0.7,  z_all[idx+1]*0.3], axis=0) 
                 x_a = self.generator(z_a[np.newaxis,:])
                 imageio.imwrite('generate_3_7_{}.png'.format(idx), self.image_grid(x_a))
                 x_b = self.generator(z_b[np.newaxis,:])
                 imageio.imwrite('generate_5_5_{}.png'.format(idx), self.image_grid(x_b)) 
                 x_c = self.generator(z_c[np.newaxis,:])
                 imageio.imwrite('generate_7_3_{}.png'.format(idx), self.image_grid(x_c))
            r.close()
        if self.config.interpolate:
            x = self.interpolator(self.dataset_train, self.batch_size)
            img = self.image_grid(x)
            imageio.imwrite('interpolate_{}.png'.format(self.config.prefix), img)
            log.warning('Completed interpolation. Interpolated samples are save' +
                        'as interpolate_{}.png'.format(self.config.prefix))

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

        log.infov("Completed evaluation.")

    def generator(self, z_1):
        x_hat = self.session.run(self.model.x_recon, feed_dict={self.model.z: z_1})
        return x_hat

    def interpolator(self, dataset, bs, num=15):
        transit_num = num - 2
        img = []
        for i in range(num):
            idx = np.random.randint(len(dataset.ids)-1)
            img1, z1 = dataset.get_data(dataset.ids[idx])
            img2, z2 = dataset.get_data(dataset.ids[idx+1])
            z = []
            for j in range(transit_num):
                z_int = (z2 - z1) * (j+1) / (transit_num+1) + z1
                z.append(z_int / np.linalg.norm(z_int))
            z = np.stack(z, axis=0)
            z_aug = np.concatenate((z, np.zeros((bs-transit_num, z.shape[1]))), axis=0)
            x_hat = self.session.run(self.model.x_recon, feed_dict={self.model.z: z_aug})
            img.append(np.concatenate((np.expand_dims(img1, 0),
                                       x_hat[:transit_num], np.expand_dims(img2, 0))))
        return np.reshape(np.stack(img, axis=0), (num*(transit_num+2),
                                                  img1.shape[0], img1.shape[1], img1.shape[2]))

    def image_grid(self, x, shape=(2048, 2048)):
        n = int(np.sqrt(x.shape[0]))
        h, w, c = self.config.data_info[0], self.config.data_info[1], self.config.data_info[2]
        I = np.zeros((n*h, n*w, c))
        for i in range(n):
            for j in range(n):
                I[h * i:h * (i+1), w * j:w * (j+1), :] = x[i * n + j]
        if c == 1:
            I = I[:, :, 0]
        return numpy.array(Image.fromarray(I).resize(shape))
	#return sm.imresize(I, shape)


    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        total_y_filter = self.total_y[batch_chunk['id'].astype(int)]
        expanded_y =  np.expand_dims(total_y_filter, axis=1)
 
        [step, loss, all_targets, all_preds, _] = self.session.run(
            [self.global_step, self.model.loss, self.model.x, self.model.x_recon, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk, labels = expanded_y)
        )

        _end_time = time.time()

        return step, loss, (_end_time - _start_time), batch_chunk, all_preds, all_targets


    def run_single_z_update_step(self, batch, dataset, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)


        total_y_filter = self.total_y[batch_chunk['id'].astype(int)]
        expanded_y =  np.expand_dims(total_y_filter, axis=1)
        
       # Optimize the latent vectors {{{
        fetch = [self.model.z, self.model.z_grad, self.model.loss]
        
        fetch_values = self.session.run(
            fetch, feed_dict=self.model.get_feed_dict(batch_chunk, labels=expanded_y , step=step)
        )

        [z, z_grad, loss_g_update] = fetch_values

        alpha = 0.05
        z_update = z - alpha * z_grad[0]
        norm = np.sqrt(np.sum(z_update ** 2, axis=1))
        z_update_norm = z_update / norm[:, np.newaxis]

        loss_z_update = self.session.run(
            self.model.loss, feed_dict={ self.model.labels : expanded_y,  self.model.x: batch_chunk['image'], self.model.z: z_update_norm}
        )
        for i in range(len(batch_chunk['id'])):
            dataset.set_data(batch_chunk['id'][i], z_update_norm[i, :])
        # }}}

        _end_time = time.time()

        return step, z,  loss_g_update, loss_z_update, batch_chunk , (_end_time - _start_time)


    def log_train_step_message(self, step, z, loss_g_update,
                         loss_z_update, min_loss, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "z : {z}"
                "G update: {loss_g_update:.5f} " +
                "Z update: {loss_z_update:.5f} " +
                "Min Loss: {min_loss:.5f}" +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         z=z, 
                         loss_z_update=loss_z_update,
                         loss_g_update=loss_g_update,
                         min_loss=min_loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )
    def log_step_message(self, step, loss, step_time, is_train=False):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss (test): {loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time,
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--reconstruct', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--interpolate', action='store_true', default=False)
    parser.add_argument('--recontrain', action='store_true', default=False)  
    parser.add_argument('--data_id', nargs='*', default=None)
    config = parser.parse_args()

    if config.dataset == 'MNIST':
        import sys
        sys.path.insert(1, '/scratch')
        import datasets.mnist as dataset
    elif config.dataset == 'SVHN':
        import datasets.svhn as dataset
    elif config.dataset == 'CIFAR10':
        import datasets.cifar10 as dataset
    else:
        raise ValueError(config.dataset)

    config.conv_info = dataset.get_conv_info()
    config.deconv_info = dataset.get_deconv_info()
    dataset_train, dataset_test = dataset.create_default_splits(is_few_shot=True,few_shot_class=5)


    m, l = dataset_train.get_data(dataset_train.ids[0])
    config.data_info = np.concatenate([np.asarray(m.shape), np.asarray(l.shape)])

    evaler = Evaler(config, dataset_test, dataset_train)

    imageio.imwrite('pre_tr_eval_{}.png'.format(dataset_train.ids[0]), evaler.image_grid(m))
    
    m, l = dataset_test.get_data(dataset_test.ids[0])
    imageio.imwrite('pre_test_eval_{}.png'.format(dataset_train.ids[0]), evaler.image_grid(m))

    m, l = dataset_test.get_data('10280')
    imageio.imwrite('pre_eval_10280.png', evaler.image_grid(m))    
 
    log.warning("dataset: %s", config.dataset)
    with tf.device('/GPU:0'):
        evaler.eval_run()

if __name__ == '__main__':
    main()
