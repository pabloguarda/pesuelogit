import tensorflow as tf
import numpy as np

def error(actual: tf.constant, predicted: tf.constant, mask = None):
    # return tf.boolean_mask(predicted - actual, tf.math.is_finite(predicted - actual))

    if mask is None:
        mask = tf.math.is_finite(predicted - actual)

    return tf.boolean_mask(predicted - actual, mask)


def l1norm(actual: tf.constant, predicted: tf.constant, weight = 1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight*tf.norm(error(actual, predicted), 1)

def sse(actual: tf.constant, predicted: tf.constant, weight = 1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight*tf.reduce_sum(tf.math.pow(error(actual, predicted), 2))

def mse(actual: tf.constant, predicted: tf.constant, weight = 1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight*tf.reduce_mean(tf.math.pow(error(actual, predicted), 2))

def mape(actual: tf.constant, predicted: tf.constant, weight = 1):
    """
    Skip cases where the observed values are equal to zero or nan
    """
    # if weight == 0:
    #     return tf.constant(0, tf.float32)

    mask = tf.cast(tf.math.is_finite(actual),tf.int32) * tf.cast(actual > 0, tf.int32)

    return 100*weight*tf.reduce_mean(tf.abs(error(actual, predicted, mask = mask))/tf.boolean_mask(actual, mask))


def rmse(actual: tf.constant, predicted: tf.constant, weight = 1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight*tf.math.sqrt(mse(actual, predicted))

def nrmse(actual: tf.constant, predicted: tf.constant, weight = 1):
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight*rmse(actual, predicted)/tf.experimental.numpy.nanmean(actual)

def mnrmse(actual: tf.constant, predicted: tf.constant, weight = 1):
    """ Normalized rmse by the maximum observed value"""
    # if weight == 0:
    #     return tf.constant(0, tf.float32)

    return weight*rmse(actual, predicted)/tf.experimental.numpy.max(actual[~tf.experimental.numpy.isnan(actual)])

def btcg_mse(actual: tf.constant, predicted: tf.constant, weight = 1):
    ''' Normalization used by Wu et al. (2018), TRC. This metric has more numerical issues than using MSE'''

    rel_error = tf.math.divide_no_nan(predicted, actual)

    return 1 / 2 * tf.reduce_mean(tf.math.pow(tf.boolean_mask(rel_error, tf.math.is_finite(rel_error)) - 1, 2))
    # return 1 / 2 * tf.reduce_mean(tf.math.pow(error(actual, predicted) /
    #                                           (tf.boolean_mask(actual, tf.math.is_finite(actual)) + epsilon), 2))