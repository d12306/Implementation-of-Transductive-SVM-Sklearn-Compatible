# -*- coding: utf-8 -*-
# @Author  : Lin Lan (ryan.linlan@gmail.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow.python.util import nest


def tensor_dict_to_vector(tensors):
    assert isinstance(tensors, dict)
    flat_tensors = nest.flatten(tensors)
    flat_flat_tensors = [tf.reshape(t, (-1,)) for t in flat_tensors]
    vector = tf.concat(flat_flat_tensors, axis=0)
    return vector


def vector_to_tensor_dict(vector, dict_struc):
    # print(vector)
    # tf.print(vector)
    # print(dict_struc)
    flat_dict_struc = nest.flatten(dict_struc)
    pointer = 0
    tensors = []
    for elem in flat_dict_struc:
        size = elem.shape.num_elements()
        tensor = tf.reshape(vector[pointer: pointer + size], elem.shape)
        tensors.append(tensor)
        pointer += size
    tensor_dict = nest.pack_sequence_as(dict_struc, tensors)
    tensor_dict = dict(tensor_dict)
    return tensor_dict


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def construct_tf_sparse_tensor(sp_sparse_matrix):
    if not sp.sparse.issparse(sp_sparse_matrix):
        raise TypeError

    tuple_format = sparse_to_tuple(sp_sparse_matrix)
    tf_sparse_tensor = tf.sparse.SparseTensor(
        indices=tuple_format[0],
        values=tuple_format[1],
        dense_shape=tuple_format[2])
    tf_sparse_tensor = tf.sparse.reorder(tf_sparse_tensor)
    return tf_sparse_tensor


@tf.function
def sparse_gather(sparse_tensor, ids):
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    mask = tf.math.equal(
        indices[:, 0][tf.newaxis, :], ids[:, tf.newaxis])
    new_first_indices = tf.where(mask)[:, 0]
    to_select = tf.where(mask)[:, 1]
    new_second_indices = tf.gather(indices, to_select, axis=0)[:, 1]
    new_indices = tf.stack((new_first_indices, new_second_indices), axis=1)
    new_values = tf.gather(values, to_select, axis=0)
    new_shape = tf.concat(
        (tf.shape(ids, out_type=tf.int64), sparse_tensor.dense_shape[1:]),
        axis=0)
    new_sparse_tensor = tf.sparse.SparseTensor(
        new_indices, new_values, new_shape)
    new_dense_tensor = tf.sparse.to_dense(new_sparse_tensor)
    return new_dense_tensor
