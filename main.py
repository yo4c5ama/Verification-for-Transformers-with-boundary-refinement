# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import os
# if not "CUDA_VISIBLE_DEVICES" in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import sys, random, time, shutil, copy, nltk, json
from multiprocessing import Pool
from Logger import Logger
from Parser import Parser, update_arguments
from data_utils import load_data, get_batches, set_seeds
from Models import Transformer
from Verifiers import VerifierForward, VerifierBackward, VerifierDiscrete
from eval_words import eval_words
import seaborn as sns
import matplotlib.pyplot as plt


def attn_distribution(ret, sent):
    label = ret[2]['pred_labels']
    attn_probs = ret[2]['attention_probs']
    last_layer_attention = attn_probs[-1]
    cls_attention = last_layer_attention[0, :, :, :]
    avg_attention = cls_attention.mean(dim=0)
    normalized_attention = avg_attention / avg_attention.sum()
    a = normalized_attention.detach().cpu().numpy()
    sns.set(font_scale=0.5)
    sns.heatmap(normalized_attention.detach().cpu().numpy(), cmap="Blues")
    plt.show()
    return

argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
args = update_arguments(args)
print(args.device)
set_seeds(args.seed)
data_train, data_valid, data_test, _, _ = load_data(args)
set_seeds(args.seed)

import tensorflow as tf
config = tf.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with sess.as_default():
    target = Transformer(args, data_train)

    # test
    # verify_test.main(target)


    random.shuffle(data_valid)
    random.shuffle(data_test)
    valid_batches = get_batches(data_valid, args.batch_size)
    test_batches = get_batches(data_test, args.batch_size)
    print("Dataset sizes: %d/%d/%d" % (len(data_train), len(data_valid), len(data_test)))

    summary_names = ["loss", "accuracy"]
    summary_num_pre = 2

    logger = Logger(sess, args, summary_names, 1)

    print("\n")

    if args.train:          
        while logger.epoch.eval() <= args.num_epoches:
            random.shuffle(data_train)
            train_batches = get_batches(data_train, args.batch_size)

            for i, batch in enumerate(train_batches):
                logger.next_step(target.step(batch, is_train=True)[:summary_num_pre])
            target.save(logger.epoch.eval())                     
            logger.next_epoch()
            for batch in valid_batches:
                logger.add_valid(target.step(batch)[:summary_num_pre])
            logger.save_valid(log=True)   
            for batch in test_batches:
                logger.add_test(target.step(batch)[:summary_num_pre])
            logger.save_test(log=True)

    data = data_valid if args.use_dev else data_test

    if args.verify:
        print("Verifying robustness...")
        if args.method == "forward" or args.method == "ibp":
            verifier = VerifierForward(args, target, logger)
        elif args.method == "backward" or args.method == "baf":
            verifier = VerifierBackward(args, target, logger)
        elif args.method == "discrete":
            verifier = VerifierDiscrete(args, target, logger)
        else:
            raise NotImplementedError("Method not implemented".format(args.method))
        verifier.run(data)
        exit(0)

    if args.word_label:
        eval_words(args, target, data_test)
        exit(0)

    # test the accuracy   
    acc = 0
    for batch in test_batches:
        sent = batch[0]["sent_a"]
        ret = target.step(batch)
        # attn_distribution(ret, sent)
        acc += target.step(batch)[1] * len(batch)
    acc = float(acc / len(data_test))
    print("Accuracy: {:.3f}".format(acc))
    with open(args.log, "w") as file:
        file.write("{:.3f}".format(acc))



