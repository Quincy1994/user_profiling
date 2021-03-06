# coding=utf-8

import argparse, os, time, logging, threading, traceback
import numpy as np
import tensorflow as tf
import sys

from Config import Config
from my_model.cnn_rnn_attn import CNN_RNN_ATT
from my_utils.utils import score
from my_utils.data_iterator import get_train_data, get_test_data, get_val_data

_REVISION = 'flatten'
parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
parser.add_argument('--restore-ckpt', action='store_true', dest='restore_ckpt', default=False)
parser.add_argument('--retain-gpu', action='store_true', dest='retain_gpu', default=False)

parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

args = parser.parse_args()

DEBUG = args.debug_enable
if not DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def debug(s):
    if DEBUG:
        print(s)
    pass

class Train:

    def __init__(self, args):

        config = Config()
        config.set_class()
        self.config = config
        gpu_lock = threading.Lock()
        gpu_lock.acquire()
        def retain_gpu():
            if args.retain_gpu:
                with tf.Session():
                    gpu_lock.acquire()
            else:
                pass

        lockThread = threading.Thread(target=retain_gpu)
        lockThread.start()
        try:
            self.args = args

            self.args = args
            self.weight_path = args.weight_path

            if args.load_config == False:
                config.saveConfig(self.weight_path + '/config')
                print('default configuration generated, please specify --load-config and run again.')
                gpu_lock.release()
                lockThread.join()
                sys.exit()
            else:
                if os.path.exists(self.weight_path + '/config'):
                    config.loadConfig(self.weight_path + '/config')
                else:
                    raise ValueError('No config file in %s' % self.weight_path + '/config')

            # if config.revision != _REVISION:
            #     raise ValueError('revision dont match %s over %s' % (config.revision, _REVISION))

            # 模型建立
            self.model = CNN_RNN_ATT(config)
            print("model create!")

        except Exception as e:
            traceback.print_exc()
            gpu_lock.release()
            lockThread.join()
            exit()

        gpu_lock.release()
        lockThread.join()


    def get_epoch(self, sess):
        epoch = sess.run(self.model.on_epoch)
        return epoch

    def run_epoch(self, sess, input_data):
        """
        Runs an epoch of training.
        Trains the model for one-epoch
        :param sess:
        :param inpu_data:
        :param verbose:
        :return:
        """
        total_loss = []
        total_ce_loss = []
        collect_time = []
        accuarcy_collect = []
        step = 0

        try:
            for batch_data in input_data:
                step += 1
                start_stamp = time.time()
                feed_dict = self.model.create_feed_dict(batch_data=batch_data, dropout_keep_prob=self.config.dropout_keep_prob)
                accuarcy, global_step, summary, ce_loss, lr, _ = sess.run(
                    [self.model.accuracy, self.model.global_step, self.merged,
                     self.model.loss, self.model.learning_rate, self.model.train_op],
                    feed_dict = feed_dict
                )
                self.train_writer.add_summary(summary, global_step)
                self.train_writer.flush()
                end_stamp = time.time()

                time_stamp = end_stamp - start_stamp

                print('step: %d, time: %.3fs, loss: %.4f, accu: %.4f, lr: %f' %(
                    step, time_stamp, ce_loss, accuarcy, lr
                ))

                # total_loss.append(opt_loss)
                total_ce_loss.append(ce_loss)
                collect_time.append(time_stamp)
                accuarcy_collect.append(accuarcy)
                # break

        except:
            traceback.print_exc()
            exit()

        epoch = sess.run(self.model.on_epoch_accu)
        print('==' * 20)
        print('train, epoch: %d, total loss: %.4f, total timestamp: %.3fs, total acc: %.4f' % (
            epoch, np.mean(total_ce_loss), np.mean(collect_time), np.mean(accuarcy_collect)
        ))

        return np.mean(total_ce_loss), np.mean(collect_time), np.mean(accuarcy_collect)


    def fit_predict(self, sess, input_data):

        # total_loss = []
        total_ce_loss = []
        collect_time = []
        total_pred = []
        total_labels = []
        step = 0

        try:
            for batch_data in input_data:
                step += 1
                labels = np.argmax(batch_data[2], axis=1)
                total_labels = np.concatenate([total_labels, labels])
                feed_dict = self.model.create_feed_dict(batch_data=batch_data, dropout_keep_prob=1.0)

                start_stamp = time.time()
                global_step, summary, ce_loss, y_pred = sess.run(
                    [self.model.global_step, self.merged, self.model.loss, self.model.prob],
                    feed_dict=feed_dict
                )

                self.test_writer.add_summary(summary, step+global_step)
                self.test_writer.flush()

                end_stamp = time.time()
                time_stamp = end_stamp - start_stamp
                collect_time.append(time_stamp)
                total_ce_loss.append(ce_loss)
                # total_loss.append(opt_loss)
                total_pred = np.concatenate([total_pred, y_pred])
        except:
            traceback.print_exc()
            exit()

        # total_labels = np.argmax(total_labels, axis=1)
        correct_predictions = float(sum(total_pred == total_labels))
        accuracy = correct_predictions / float(len(total_labels))
        return np.mean(total_ce_loss), total_labels, total_pred, accuracy

    def test_case(self, sess, input_data, onset='VALIDATION'):
        print('#' * 20, 'ON ' + onset + ' SET START ', '#' * 20)
        print('='*10 + ' '.join(sys.argv) + '=' * 10)
        epoch = self.get_epoch(sess)
        ce_loss, label, pred, accuracy = self.fit_predict(sess, input_data)

        precision, recall, f1 = score(pred, label)
        # print(precision)
        # print(recall)
        # print(f1)
        print('the Epoch %d ---- %s Overall accuary is: %f, Overall ce_loss is: %f' %(epoch, onset, accuracy, ce_loss))
        print('the Epoch %d ---- %s Overall precision is: %f, Overall recall is: %f, Overall f1 is: %f' % (epoch, onset, precision, recall, f1))
        logging.info('%d the Epoch ---- %s Overall accuary is: %f, Overall ce_loss is: %f' %(epoch, onset, accuracy, ce_loss))
        logging.info('%d the Epoch ---- %s Overall precision is: %f, Overall recall is: %f, Overall f1 is: %f' % (epoch, onset, precision, recall, f1))
        return accuracy, ce_loss

    def train_run(self):
        logging.info('Training start')
        logging.info("Parameter count is: %d" % (self.model.param_cnt))
        if not args.restore_ckpt:
            self.remove_file(self.args.weight_path + '/summary.log')
        saver = tf.train.Saver(max_to_keep=10)

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.allow_soft_placement = True

        with tf.Session(config=conf) as sess:

            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_train',
                                                      sess.graph)
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')

            sess.run(tf.global_variables_initializer())

            if args.restore_ckpt:
                saver.restore(sess, self.args.weight_path + '/classifier.weights')

            best_loss = np.Inf
            best_accuracy = 0
            best_val_epoch = self.get_epoch(sess)



            # 第一次网络训练
            for _ in range(self.config.max_first_epochs):

                # 生成数据集
                self.train_data = get_train_data(shuffle=True)
                self.val_data = get_val_data(shuffle=False)
                self.test_data = get_test_data(shuffle=False)
                print("get data")

                epoch = self.get_epoch(sess)
                print("=" * 20 + "Epoch ", epoch, "=" * 20)
                ce_loss, opt_loss, accuracy  = self.run_epoch(sess, self.train_data)
                logging.info('Mean training accuracy is :  %.4f' % accuracy)
                logging.info('Mean ce_loss in %dth epoch is: %f, Mean ce_loss is: %f' %(epoch, ce_loss, opt_loss))
                val_accuracy, val_loss = self.test_case(sess, self.val_data, onset='VALIDATION')
                test_accuracy, test_loss = self.test_case(sess, self.test_data,onset='TEST')
                self.save_loss_accu(self.args.weight_path + '/summary.log', train_loss=ce_loss,
                                    valid_loss=val_loss, test_loss=test_loss,
                                    valid_accu=val_accuracy, test_accu=test_accuracy, epoch=epoch)
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(self.args.weight_path):
                        os.makedirs(self.args.weight_path)
                    logging.info('best epoch is %dth epoch' % best_val_epoch)
                    saver.save(sess, self.args.weight_path + '/classifier_one.weights')

        #

    def test_run(self):

        # 生成数据集
        # self.train_data = get_train_data(shuffle=True)
        # self.val_data = get_val_data(shuffle=False)
        self.test_data = get_test_data(shuffle=False)
        print("get data")

        saver = tf.train.Saver(max_to_keep=10)
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.allow_soft_placement = True

        with tf.Session(config=conf) as sess:
            self.merged = tf.summary.merge_all()
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.args.weight_path + '/classifier.weights')

            self.test_case(sess, self.test_data, onset='TEST')

    def main_run(self):


        if not os.path.exists(self.args.weight_path):
            os.makedirs(self.args.weight_path)
        logFile = self.args.weight_path + '/run.log'

        if self.args.train_test == 'train':

            try:
                os.remove(logFile)
            except OSError:
                pass
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            debug('_main_run_')
            self.train_run()
            self.test_run()
        else:
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            self.test_run()

    @staticmethod
    def remove_file(fileName):
        if os.path.exists(fileName):
            os.remove(fileName)


    @staticmethod
    def save_loss_accu(fileName, train_loss, valid_loss,
                       test_loss, valid_accu, test_accu, epoch):
        with open(fileName, 'a') as fd:
            fd.write('epoch: %3d\t train_loss: %.4f\t valid_loss: %.4f\t test_loss: %.4f\t valid_accu: %.4f\t test_accu: %.4f\n' %
                     (
                         epoch, train_loss, valid_loss,
                         test_loss, valid_accu, test_accu
                     ))

if __name__ == '__main__':
    trainer = Train(args)
    trainer.main_run()









