# coding=utf-8
import argparse, sys, os, time, logging, threading, traceback
import numpy as np
import tensorflow as tf
import _pickle as pkl
import sys

from Config import *
from model.cnn_model import CNN
from Metrics import score
from data_iterator import get_cnn_data


class Parser:

    def __init__(self):
        self.weight_path = config.cache_dir + "/weight/" + config.task + '/' + config.model_name
        self.train_test = 'train'
        self.load_config = True
        self.restore_ckpt = False
        self.debug_enable = False

# parser = argparse.ArgumentParser(description="training options")
# parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
# parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
# parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
# parser.add_argument('--weight-path', action='store', dest='weight_path',required=True)
# parser.add_argument('--restore-ckpt', action='store_true', dest='restorce_ckpt', default=False)
# parser.add_argument('--retain-gpu', action='store_true', dest='retain_gpu', default=False)
# parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)


args = Parser()

# args = parser.parse_args()


class Train:

    def __init__(self, args):

        config = Config()
        config.set_class()
        self.config = config

        try:
            self.args = args
            self.weight_path = args.weight_path

            if args.load_config == False:
                config.saveConfig(self.weight_path + '/config')
                print('default configuration generated, please specify -- load-config and run again.')
                sys.exit()
            else:
                if os.path.exists(self.weight_path + '/config'):
                    config.loadConfig(self.weight_path + '/config')
                else:
                    raise ValueError('No config file in %s' % (self.weight_path + '/config'))

        except Exception as e:
            traceback.print_exc()
            exit()

        # bulid model
        self.model = CNN(config)


    def get_epoch(self, sess):
        epoch = sess.run(self.model.on_epoch)
        return epoch

    def run_epoch(self, sess, input_data):
        """
        Run an epoch of training.
        Trains the model for one epoch
        :param sess:
        :param input_data:
        :param topic_data:
        :return:
        """
        total_prediction = []
        total_labels = []
        accuarcy_collect = []
        total_loss = []
        collect_time  = []
        step = 0

        try:
            for batch_data in input_data:
                batch_x, batch_wNum, batch_labels = batch_data[0], batch_data[1], batch_data[2]
                step += 1
                start_stamp = time.time()
                feed_dict = {
                    self.model.ph_input:batch_x,
                    self.model.ph_labels: batch_labels,
                    self.model.ph_train: config.isdroput,
                }
                (accuracy, prediction, global_step, summary, loss, lr,_)=sess.run(
                    [self.model.accuracy, self.model.prediction, self.model.global_step, self.merged, self.model.loss, self.model.learning_rate, self.model.train_op],
                    feed_dict=feed_dict
                )

                self.train_writer.add_summary(summary, global_step)
                self.train_writer.flush()
                end_stamp = time.time()
                time_stamp = end_stamp - start_stamp

                print('step: %d, time: %.3fs, loss: %.4f,  accu: %.4f, lr: %f' %(
                    step, time_stamp, loss, accuracy, lr
                ))
                total_prediction.extend(prediction)
                total_labels.extend(batch_labels)
                accuarcy_collect.append(accuracy)
                total_loss.append(loss)
                collect_time.append(time_stamp)
        except:
            traceback.print_exc()
            exit()

        total_labels = np.argmax(total_labels, axis=1)
        epoch = sess.run(self.model.on_epoch_add)
        print('==' * 20)
        print('epoch: %d, total loss: %.4f, total timestamp: %.3fs, total acc: %.4f' % (
            epoch, np.mean(total_loss), np.mean(collect_time), np.mean(accuarcy_collect)
        ))
        return np.mean(total_loss), np.mean(collect_time), np.mean(accuarcy_collect)

    def fit_predict(self, sess, input_data):

        total_loss = []
        collect_time = []
        total_pred = []
        total_labels  = []
        step = 0

        try:
            for batch_data in input_data:
                batch_x, batch_wNum, batch_labels = batch_data[0], batch_data[1], batch_data[2]
                step += 1
                total_labels.extend(batch_labels)
                feed_dict = {
                    self.model.ph_input: batch_x,
                    self.model.ph_labels: batch_labels,
                    self.model.ph_train: config.isdroput,
                }

                start_stamp = time.time()
                global_step, summary, loss, y_pred = sess.run(
                    [self.model.global_step, self.merged, self.model.loss, self.model.prediction],
                    feed_dict = feed_dict
                )

                self.test_writer.add_summary(summary, step+global_step)
                self.test_writer.flush()

                end_stamp = time.time()
                time_stamp = end_stamp - start_stamp
                collect_time.append(time_stamp)
                total_loss.append(loss)
                total_pred.extend(y_pred)

        except:
            traceback.print_exc()
            exit()
        total_labels = np.argmax(total_labels, axis=1)
        correct_predictions = float(sum(total_pred == total_labels))
        accuracy = correct_predictions / float(len(total_labels))
        pre, recall, f1 = score(total_pred, total_labels)
        avg_loss = np.mean(total_loss)
        return avg_loss, accuracy, pre, recall, f1

    def test_case(self, sess, input_data, onset='VALIDATION'):
        print('#' * 20, 'ON ' + onset + ' SET START ', '#' * 20)
        print('='*10, ' '.join(sys.argv) + '='* 10)
        epoch = self.get_epoch(sess)
        avg_loss, accuracy, pre, recall, f1 = self.fit_predict(sess, input_data)
        print('%d the Epoch ---- %s Overall accuracy is: %f, Overall loss is: %f' %(epoch, onset, accuracy, avg_loss))
        print('%d the Epoch ---- %s Overall precision is: %f, Overall recall is: %f, Overall f1 is: %f' %(epoch, onset, pre, recall, f1))
        logging.info('%d the Epoch ---- %s Overall accuracy is: %f, Overall loss is: %f' % (epoch, onset, accuracy, avg_loss))
        logging.info('%d the Epoch ---- %s Overall precision is: %f, Overall recall is: %f, Overall f1 is: %f' % (epoch, onset, pre, recall, f1))
        return accuracy, avg_loss

    def train_run(self):
        logging.info('Training start')
        logging.info("Parameter count is %d"%(self.model.param_cnt))
        if not args.restore_ckpt:
            self.remove_file(self.args.weight_path + '/summary.log')
        saver = tf.train.Saver(max_to_keep=10)

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.allow_soft_placement = True

        with tf.Session(config=conf) as sess:

            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_train', sess.graph)
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')

            sess.run(tf.global_variables_initializer())

            if args.restore_ckpt:
                saver.restore(sess, self.args.weight_path + '/classifier.weights')

            best_loss = np.Inf
            best_accuarcy = 0
            best_val_epoch = self.get_epoch(sess)

            for _ in range(config.max_epochs):
                train_data = get_cnn_data(config.train_data_path, shuffle=True)
                valid_data = get_cnn_data(config.val_data_path, shuffle=False)
                test_data = get_cnn_data(config.test_data_path, shuffle=False)
                print("get data")
                epoch = self.get_epoch(sess)
                print('=' * 20 + "Epoch ", epoch, "=" * 20)

                train_loss, collect_time, train_accuarcy = self.run_epoch(sess, train_data)
                logging.info('Mean training accuracy is : %.4f' %(train_accuarcy))
                logging.info('Mean loss in %dth epoch is: %f' %(epoch, train_loss))
                valid_accuracy, valid_loss = self.test_case(sess, valid_data, onset='VALIDATION')
                test_accuracy, test_loss = self.test_case(sess, test_data, onset='TEST')
                self.save_loss_accu(self.args.weight_path + '/summary.log', train_loss=train_loss,
                                    valid_loss=valid_loss, test_loss=test_loss,
                                    valid_accu=valid_accuracy, test_accu=test_accuracy,epoch=epoch)
                if best_accuarcy < valid_accuracy:
                    best_accuarcy = valid_accuracy
                    best_val_epoch = epoch
                    best_loss = valid_loss
                    if not os.path.exists(self.args.weight_path):
                        os.makedirs(self.args.weight_path)
                    logging.info('best epoch is %dth epoch' % best_val_epoch)
                    saver.save(sess, self.args.weight_path + '/classifier_one.weights')

    def test_run(self):
        test_data = get_cnn_data(config.test_data_path, shuffle=False)
        print("get data")

        saver = tf.train.Saver(max_to_keep=10)
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.allow_soft_placement = True

        with tf.Session(config=conf) as sess:
            self.merged = tf.summary.merge_all()
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.args.weight_path + '/classfier.weights')
            self.test_case(sess, test_data, onset='TEST')

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