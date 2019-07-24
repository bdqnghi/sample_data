"""Train the cnn model as  described in Lili Mou et al. (2015) 
https://arxiv.org/pdf/1409.5718.pdf"""

import os
import logging
import pickle
import tensorflow as tf
import numpy as np
import network as network
import sampling as sampling
import sys
import random
from parameters import LEARN_RATE, EPOCHS, CHECKPOINT_EVERY, BATCH_SIZE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train_model(logdir, infile, embedfile, epochs=EPOCHS, training="True", testing="True"):
    """Train a classifier to label ASTs"""

    print("Loading trees...")
    with open(infile, 'rb') as fh:
        trees, test_trees, labels = pickle.load(fh)

        random.shuffle(trees)
        

    print(labels)
    print("Loading embeddings....")
    with open(embedfile, 'rb') as fh:
        embeddings, embed_lookup = pickle.load(fh)
        num_feats = len(embeddings[0])

    # build the inputs and outputs of the network


    num_batches = len(trees) // BATCH_SIZE + (1 if len(trees) % BATCH_SIZE != 0 else 0)
    for epoch in range(1, epochs+1):
        for i, batch in enumerate(sampling.batch_samples(
            sampling.gen_samples(trees, labels, embeddings, embed_lookup), BATCH_SIZE
        )):
            nodes, children, batch_labels = batch
            step = (epoch - 1) * num_batches + i * BATCH_SIZE

            if not nodes:
                continue # don't try to train on an empty batch
            # print(batch_labels)
            _, summary, err, out = sess.run(
                [train_step, summaries, loss_node, out_node],
                feed_dict={
                    nodes_node: nodes,
                    children_node: children,
                    labels_node: batch_labels
                }
            )

            print('Epoch:', epoch, 'Step:', step, 'Loss:', err, 'Max nodes:', len(nodes[0]))

            writer.add_summary(summary, step)
            if step % CHECKPOINT_EVERY == 0:
                # save state so we can resume later
                saver.save(sess, os.path.join(checkfile), step)
                print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(step) + ', loss: ' + str(err) + '.')

    saver.save(sess, os.path.join(checkfile), step)


def main():
    # logdir = "bi-tbcnn/bi-tbcnn/logs/20_classes_pku_no_dependency"
    logdir = sys.argv[1]
    inputs = sys.argv[2]
    embeddings = sys.argv[3]
    training = sys.argv[4]
    testing = sys.argv[5]

    train_model(logdir,inputs,embeddings,EPOCHS, training, testing) 

if __name__ == "__main__":
    main()