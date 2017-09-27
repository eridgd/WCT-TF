from __future__ import print_function, division

import argparse
import functools
import time
import tensorflow as tf, numpy as np, os, random
from utils import get_files, get_img_random_crop
from model import WCTModel
import threading


parser = argparse.ArgumentParser()
### Directories
parser.add_argument('--checkpoint', type=str,
                    dest='checkpoint', help='Checkpoint save dir', 
                    required=True)
parser.add_argument('--log-path', type=str,
                    dest='log_path', help='Logging dir path')
parser.add_argument('--relu-target', type=str, required=True,
                    help='Target VGG19 relu layer to decode from, e.g. relu4_1')
parser.add_argument('--content-path', type=str, required=True,
                    dest='content_path', help='Content images folder')
parser.add_argument('--val-path', type=str, default=None,
                    dest='val_path', help='Validation images folder')
parser.add_argument('--vgg-path', type=str,
                    dest='vgg_path', help='Path to vgg_normalised.t7', 
                    default='models/vgg_normalised.t7')

### Loss weights
parser.add_argument('--feature-weight', type=float,
                    dest='feature_weight', help='Feature loss weight',
                    default=1)
parser.add_argument('--pixel-weight', type=float,
                    dest='pixel_weight', help='Pixel reconstruction loss weight',
                    default=1)
parser.add_argument('--tv-weight', type=float,
                    dest='tv_weight', help='Total variation loss weight',
                    default=0)

### Train opts
parser.add_argument('--learning-rate', type=float,
                    dest='learning_rate', help='Learning rate',
                    default=1e-4)
parser.add_argument('--lr-decay', type=float,
                    dest='lr_decay', help='Learning rate decay',
                    default=0)
parser.add_argument('--max-iter', type=int,
                    dest='max_iter', help='Max # of training iterations',
                    default=16000)
parser.add_argument('--batch-size', type=int,
                    dest='batch_size', help='Batch size',
                    default=8)
parser.add_argument('--save-iter', type=int,
                    dest='save_iter', help='Checkpoint save frequency',
                    default=200)
parser.add_argument('--summary-iter', type=int,
                    dest='summary_iter', help='Summary write frequency',
                    default=20)
parser.add_argument('--max-to-keep', type=int,
                    dest='max_to_keep', help='Max # of checkpoints to keep around',
                    default=10)

args = parser.parse_args()


def batch_gen(folder, batch_shape):
    '''Resize images to 512, randomly crop a 256 square, and normalize'''
    files = np.asarray(get_files(folder))
    while True:
        X_batch = np.zeros(batch_shape, dtype=np.float32)

        idx = 0

        while idx < batch_shape[0]:  # Build batch sample by sample
            try:
                f = np.random.choice(files)

                X_batch[idx] = get_img_random_crop(f, resize=512, crop=256).astype(np.float32)
                X_batch[idx] /= 255.    # Normalize between [0,1]
                
                assert(not np.isnan(X_batch[idx].min()))
            except Exception as e:
                # Do not increment idx if we failed 
                print(e)
                continue
            idx += 1

        yield X_batch


def train():
    batch_shape = (args.batch_size,256,256,3)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        ### Setup data loading queue
        queue_input_content = tf.placeholder(tf.float32, shape=batch_shape)
        queue_input_val = tf.placeholder(tf.float32, shape=batch_shape)
        queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32], shapes=[[256,256,3], [256,256,3]])
        enqueue_op = queue.enqueue_many([queue_input_content, queue_input_val])
        dequeue_op = queue.dequeue()
        content_batch_op, val_batch_op = tf.train.batch(dequeue_op, batch_size=args.batch_size, capacity=100)

        def enqueue(sess):
            content_images = batch_gen(args.content_path, batch_shape)
            
            val_path = args.val_path if args.val_path is not None else args.content_path
            val_images = batch_gen(val_path, batch_shape)

            while True:
                content_batch = next(content_images)
                val_batch     = next(val_images)

                sess.run(enqueue_op, feed_dict={queue_input_content: content_batch,
                                                queue_input_val:     val_batch})

        ### Build the model graph & train/summary ops, and get the EncoderDecoder
        model = WCTModel(mode='train',
                         relu_targets=[args.relu_target],
                         vgg_path=args.vgg_path,
                         batch_size=args.batch_size,
                         feature_weight=args.feature_weight, 
                         pixel_weight=args.pixel_weight,
                         tv_weight=args.tv_weight,
                         learning_rate=args.learning_rate,
                         lr_decay=args.lr_decay).encoder_decoders[0]

        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            enqueue_thread = threading.Thread(target=enqueue, args=[sess])
            enqueue_thread.isDaemon()
            enqueue_thread.start()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            log_path = args.log_path if args.log_path is not None else os.path.join(args.checkpoint,'log')
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)

            sess.run(tf.global_variables_initializer())
 
            def load_latest():
                if os.path.exists(os.path.join(args.checkpoint,'checkpoint')):
                    print("Restoring checkpoint")
                    saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            load_latest()

            for iteration in range(args.max_iter):
                start = time.time()
                
                content_batch = sess.run(content_batch_op)

                fetches = {
                    'train':        model.train_op,
                    'global_step':  model.global_step,
                    # 'summary':      model.summary_op,
                    'lr':           model.learning_rate,
                    'feature_loss': model.feature_loss,
                    'pixel_loss':   model.pixel_loss,
                    'tv_loss':      model.tv_loss
                }

                feed_dict = { model.content_input: content_batch }

                try:
                    results = sess.run(fetches, feed_dict=feed_dict)
                except Exception as e:
                    print(e)
                    print("Exception encountered, re-loading latest checkpoint")
                    load_latest()
                    continue

                ### Run a val batch and log the summaries
                if iteration % args.summary_iter == 0:
                    val_batch = sess.run(val_batch_op)
                    summary = sess.run(model.summary_op, feed_dict={ model.content_input: val_batch })
                    summary_writer.add_summary(summary, results['global_step'])

                ### Save checkpoint
                if iteration % args.save_iter == 0:
                    save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'), results['global_step'])
                    print("Model saved in file: %s" % save_path)

                ### Log training stats
                print("Step: {}  LR: {:.7f}  Feature: {:.5f}  Pixel: {:.5f}  TV: {:.5f}  Time: {:.5f}".format(results['global_step'], 
                                                                                                              results['lr'], 
                                                                                                              results['feature_loss'], 
                                                                                                              results['pixel_loss'], 
                                                                                                              results['tv_loss'], 
                                                                                                              time.time() - start))

            # Last save
            save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'), results['global_step'])
            print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    train()
