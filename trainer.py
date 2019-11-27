from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from util import log
from pprint import pprint

from model import Model
from input_ops import create_input_ops

import os
import time
import h5py
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class Trainer(object):

    def __init__(self,
                 config,
                 dataset_train,
                 dataset_test):
        self.config = config
        hyper_parameter_str = config.dataset+'_lr_'+str(config.learning_rate)
        self.train_dir = './train_dir/%s-%s-%s' % (
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(dataset_train, self.batch_size,
                                               is_training=True)
        _, self.batch_test = create_input_ops(dataset_test, self.batch_size,
                                              is_training=False)
     

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.total_y = np.concatenate((y_train,y_test))        
        
	# --- create model ---
        self.model = Model(config)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                config.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )

        self.check_op = tf.no_op()

        # --- checkpoint and monitoring ---
        log.warn("********* var ********** ")
        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)

        self.g_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
            name='g_optimizer_loss',
        )

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.checkpoint_secs = 600  # 10 min

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
	    log_device_placement=True 
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self, dataset):
        log.infov("Training Starts!")
        pprint(self.batch_train)

        max_steps = 100000

        output_save_step = 1000

        for s in xrange(max_steps):
            step, summary, x, loss, loss_g_update, loss_z_update, step_time = \
                self.run_single_step(self.batch_train, dataset, step=s, is_train=True)

            if s % 10 == 0:
                self.log_step_message(step, loss, loss_g_update, loss_z_update, step_time)

            self.summary_writer.add_summary(summary, global_step=step)

            if s % output_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                save_path = self.saver.save(self.session,
                                            os.path.join(self.train_dir, 'model'),
                                            global_step=step)
                if self.config.dump_result:
                    f = h5py.File(os.path.join(self.train_dir, 'dump_result_'+str(s)+'.hdf5'), 'w')
                    f['image'] = x
                    f.close()

    def run_single_step(self, batch, dataset, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        # Optmize the generator {{{
        # ========
        fetch = [self.global_step, self.summary_op, self.model.loss,
                 self.model.x_recon, self.check_op, self.g_optimizer]

        total_y_filter = self.total_y[batch_chunk['id'].astype(int)]
        expanded_y =  np.expand_dims(total_y_filter, axis=1)
        
        fetch_values = self.session.run(
            fetch, feed_dict=self.model.get_feed_dict(batch_chunk,labels=expanded_y, step=step)
        )
        [step, summary, loss, x] = fetch_values[:4]
        # }}}

        # Optimize the latent vectors {{{
        fetch = [self.model.z, self.model.z_grad, self.model.loss]
        
        fetch_values = self.session.run(
            fetch, feed_dict=self.model.get_feed_dict(batch_chunk, labels=expanded_y , step=step)
        )

        [z, z_grad, loss_g_update] = fetch_values

        z_update = z - self.config.alpha * z_grad[0]
        norm = np.sqrt(np.sum(z_update ** 2, axis=1))
        z_update_norm = z_update / norm[:, np.newaxis]

        loss_z_update = self.session.run(
            self.model.loss, feed_dict={ self.model.labels : expanded_y,  self.model.x: batch_chunk['image'], self.model.z: z_update_norm}
        )
        for i in range(len(batch_chunk['id'])):
            dataset.set_data(batch_chunk['id'][i], z_update_norm[i, :])
        # }}}

        _end_time = time.time()

        return step, summary, x, loss, loss_g_update, loss_z_update, (_end_time - _start_time)

    def run_test(self, batch, is_train=False, repeat_times=8):

        batch_chunk = self.session.run(batch)

        loss = self.session.run(
            self.model.loss, feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False)
        )

        return loss

    def log_step_message(self, step, loss, loss_g_update,
                         loss_z_update, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "G update: {loss_g_update:.5f} " +
                "Z update: {loss_z_update:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         loss_z_update=loss_z_update,
                         loss_g_update=loss_g_update,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False


def main():
    tf.debugging.set_log_device_placement(True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'SVHN', 'CIFAR10'])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--dump_result', action='store_true', default=False)
    parser.add_argument('--few_shot_class', type=int, default=None)
    parser.add_argument('--few_shot_cap', type=bool, default=False) 
    parser.add_argument('--train_sample_cap', type=int, default=None)
    parser.add_argument('--test_sample_cap', type=int, default=None)
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
    dataset_train, dataset_test  = dataset.create_default_splits(config)

    m, l = dataset_train.get_data(dataset_train.ids[0])
    config.data_info = np.concatenate([np.asarray(m.shape), np.asarray(l.shape)])

    trainer = Trainer(config,
		      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f",
		config.dataset, config.learning_rate)
    with tf.device('/GPU:0'):
	    trainer.train(dataset_train)

if __name__ == '__main__':
    main()
