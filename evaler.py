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
        # load checkpoint
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

        if not (self.config.interpolate or self.config.generate or self.config.reconstruct):
            raise ValueError('Please specify at least one task by indicating' +
                             '--reconstruct, --generate, or --interpolate.')
            return

        if self.config.reconstruct:
            try:
                for s in xrange(max_steps):
                    step, loss, step_time, batch_chunk, prediction_pred, prediction_gt = \
                        self.run_single_step(self.batch)
                    self.log_step_message(s, loss, step_time)
                    evaler.add_batch(batch_chunk['id'], prediction_pred, prediction_gt)

            except Exception as e:
                coord.request_stop(e)

            evaler.report()
            log.warning('Completed reconstruction.')

        if self.config.generate:
            x = self.generator(self.batch_size)
            img = self.image_grid(x)
            imageio.imwrite('generate_{}.png'.format(self.config.prefix), img)
            log.warning('Completed generation. Generated samples are save' +
                        'as generate_{}.png'.format(self.config.prefix))

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

    def generator(self, num):
        z1 = np.random.randn(num, self.config.data_info[3])
        img2, z_2 = self.dataset_train.get_data('17269')
        r = open('img_ids_for_class.txt', "r")
        f = open('thisisz_i', "w+")
        if (r.mode == "r"):
           lines = r.readlines()
           z_all = []
           for line in lines:
                line_split = line.split(":")
                z_i = line_split[-1]
                f.write(z_i.strip() + " \n")
                imgi, z_i = self.dataset_train.get_data(z_i.strip())
                z_all.append(z_i)
           z_avg = np.mean(np.array(z_all), axis = 0)
           f.close()
           #imageio.imwrite('classFor44657_{}.png'.format(self.config.prefix), img1)
           #z3 = np.mean(np.array([z, z_2]), axis=0)
           #row_sums = np.sqrt(np.sum(z1 ** 2, axis=0))
           #z2 = z1 / row_sums[np.newaxis, :]
           z_avg = z_avg[np.newaxis,:]
           r.close()
           x_hat = self.session.run(self.model.x_recon, feed_dict={self.model.z: z_avg})
           return x_hat
        return self.session.run(self.model.x_recon, feed_dict={self.modle.z: z_2})

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

        [step, loss, all_targets, all_preds, _] = self.session.run(
            [self.global_step, self.model.loss, self.model.x, self.model.x_recon, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, loss, (_end_time - _start_time), batch_chunk, all_preds, all_targets

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
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--reconstruct', action='store_true', default=False)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--interpolate', action='store_true', default=False)
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
    dataset_train, dataset_test = dataset.create_default_splits()

    m, l = dataset_train.get_data(dataset_train.ids[0])
    config.data_info = np.concatenate([np.asarray(m.shape), np.asarray(l.shape)])

    evaler = Evaler(config, dataset_test, dataset_train)

    log.warning("dataset: %s", config.dataset)
    with tf.device('/GPU:0'):
        evaler.eval_run()

if __name__ == '__main__':
    main()
