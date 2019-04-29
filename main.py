import model
import os
import json
import argparse
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    # parser.add_argument('--load', type=int, default=99,
                        # help='Epoch id of pre-trained model')
    parser.add_argument("--data", type= str, help='training_data')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--epoch', type=int, default = 10, 
                        help = "Max Epoch")
    parser.add_argument('--bsize', type=int, default=60,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.4,
                        help='Keep probability for dropout')
    parser.add_argument('--class_num', type=int, default = 2, 
                        help='class number')
    # parser.add_argument('--maxepoch', type=int, default=100,
    #                     help='Max number of epochs for training')

    # parser.add_argument('--im_name', type=str, default='.png',
    #                     help='Part of image name')

    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = get_args()
    if FLAGS.train:
        cnn_model = model.Model(FLAGS.bsize, FLAGS.lr, FLAGS.keep_prob, FLAGS.class_num, is_training=True)
        trainer = model.Trainer(FLAGS.data, cnn_model, FLAGS.epoch, FLAGS.class_num)
        writer = tf.summary.FileWriter("./log/")

        with tf.Session() as sess:
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())
            trainer.train(sess, writer)
        
        writer.close()