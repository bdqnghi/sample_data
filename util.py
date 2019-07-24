"""Functions to help with sampling trees."""

import pickle
import numpy as np
import random
from tqdm import *


def build_tree(file_path):
    """Builds an AST from a script."""
   
    with open(file_path, 'rb') as file_handler:
        tree = pickle.load(file_handler, encoding = "latin1")
        # print(tree)
        return tree
    return "error"

def _traverse_tree(root):
    num_nodes = 1
    queue = [root]

    root_json = {
        "node": str(root.kind),

        "children": []
    }
    queue_json = [root_json]
    while queue:
      
        current_node = queue.pop(0)
        num_nodes += 1
        # print (_name(current_node))
        current_node_json = queue_json.pop(0)


        children = [x for x in current_node.child]
        queue.extend(children)
       
        for child in children:
            # print "##################"
            #print child.kind

            child_json = {
                "node": str(child.kind),
                "children": []
            }

            current_node_json['children'].append(child_json)
            queue_json.append(child_json)
            # print current_node_json
  
    return root_json, num_nodes

def gen_samples(all_tree_files, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}
    # print vector_lookup
    # print("ASDasdasd")
    for file in all_tree_files:
        # print(file)
        tree_pb = build_tree(file[0])
        tree, num_nodes = _traverse_tree(tree_pb.element)
        print(tree)
        nodes = []
        children = []
        label = file[1]

        queue = [(tree, -1)]
        # print queue
        while queue:
            # print "############"
            node, parent_ind = queue.pop(0)
            # print node
            # print parent_ind
            node_ind = len(nodes)
            # print "node ind : " + str(node_ind)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            
            n = str(node['node'])
            look_up_vector = vector_lookup[n]
            nodes.append(vectors[int(n)])
        # print "children list length: " + str(len(children))
        # yield (nodes, children, label)
        print(nodes, children, label)

def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels = [], [], []
    samples = 0
    for n, c, l in gen:
        # print n
        # print c
        # print l
        if len(c) > 6000 and len(c) < 20000:
            nodes.append(n)
            children.append(c)
            labels.append(l)
            samples += 1


        if samples >= batch_size:
            yield _pad_batch(nodes, children, labels)
            nodes, children, labels = [], [], []
            samples = 0

    if nodes:
        yield _pad_batch(nodes, children, labels)

def _pad_batch(nodes, children, labels):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, labels

def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]