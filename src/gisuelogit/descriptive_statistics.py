import tensorflow as tf

def error(actual: tf.constant, predicted: tf.constant):
    # return tf.boolean_mask(predicted - actual, tf.math.is_finite(predicted - actual))
    return tf.boolean_mask(predicted - actual, tf.math.is_finite(predicted - actual))


def l1norm(actual: tf.constant, predicted: tf.constant, weight = 1) -> tf.float64:
    if weight == 0:
        return tf.constant(0, tf.float64)
    return tf.cast(weight*tf.norm(error(actual, predicted), 1),tf.float64)

def sse(actual: tf.constant, predicted: tf.constant, weight = 1) -> tf.float64:
    if weight == 0:
        return tf.constant(0, tf.float64)
    return tf.cast(weight*tf.reduce_sum(tf.math.pow(error(actual, predicted), 2)),tf.float64)

def mse(actual: tf.constant, predicted: tf.constant, weight = 1) -> tf.float64:
    if weight == 0:
        return tf.constant(0, tf.float64)
    return tf.cast(weight*tf.reduce_mean(tf.math.pow(error(actual, predicted), 2)),tf.float64)

# (actual-predicted).numpy().flatten().shape

def rmse(actual: tf.constant, predicted: tf.constant, weight = 1) -> tf.float64:
    if weight == 0:
        return tf.constant(0, tf.float64)
    return tf.cast(weight*tf.math.sqrt(mse(actual, predicted)),tf.float64)

def nrmse(actual: tf.constant, predicted: tf.constant, weight = 1) -> tf.float64:
    if weight == 0:
        return tf.constant(0, tf.float64)
    return weight*rmse(actual, predicted)/tf.experimental.numpy.nanmean(actual)

def mnrmse(actual: tf.constant, predicted: tf.constant, weight = 1) -> tf.float64:
    """ Normalized rmse by the maximum observed value"""
    if weight == 0:
        return tf.constant(0, tf.float64)

    return weight*rmse(actual, predicted)/tf.experimental.numpy.max(actual[~tf.experimental.numpy.isnan(actual)])

def btcg_mse(actual: tf.constant, predicted: tf.constant, weight = 1):
    ''' Normalization used by Wu et al. (2018), TRC. This metric has more numerical issues than using MSE'''

    rel_error = tf.math.divide_no_nan(predicted, actual)

    return 1 / 2 * tf.reduce_mean(tf.math.pow(tf.boolean_mask(rel_error, tf.math.is_finite(rel_error)) - 1, 2))
    # return 1 / 2 * tf.reduce_mean(tf.math.pow(error(actual, predicted) /
    #                                           (tf.boolean_mask(actual, tf.math.is_finite(actual)) + epsilon), 2))