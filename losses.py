"""Loss functions
"""
import tensorflow as tf
import keras.losses as kloss
from concise.utils.helper import get_from_module
import keras.backend as K
import gin
from gin import config
from scipy.stats import pearsonr


@gin.configurable
def ignoreNaNloss(y_true, y_pred):
    bool_finite = tf.is_finite(y_true)
    return K.mean(K.square(tf.boolean_mask(y_pred, bool_finite) - tf.boolean_mask(y_true, bool_finite)), axis=-1)

@gin.configurable
def correlation_loss(y_true, y_pred):
    rp = pearsonr(y_true,y_pred)[0]
    return 1 - (rp**2)

@gin.configurable
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood along the sequence (axis=1)
    and sum the values across all each channels

    Args:
      true_counts: observed count values (batch, seqlen, channels)
      logits: predicted logit values (batch, seqlen, channels)
    """
    # swap axes such that the final axis will be the positional axis
    logits_perm = tf.transpose(logits, (0, 2, 1))
    true_counts_perm = tf.transpose(true_counts, (0, 2, 1))

    counts_per_example = tf.reduce_sum(true_counts_perm, axis=-1)

    dist = tf.contrib.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits_perm)

    # get the sequence length for normalization
    seqlen = tf.to_float(tf.shape(true_counts)[0])

    return -tf.reduce_sum(dist.log_prob(true_counts_perm)) / seqlen


@gin.configurable
class CountsMultinomialNLL:

    def __init__(self, c_task_weight=1):
        self.c_task_weight = c_task_weight

    def __call__(self, true_counts, preds):
        probs = preds / K.sum(preds, axis=-2, keepdims=True)
        logits = K.log(probs / (1 - probs))

        # multinomial loss
        multinomial_loss = multinomial_nll(true_counts, logits)

        mse_loss = kloss.mse(K.log(1 + K.sum(true_counts, axis=(-2, -1))),
                             K.log(1 + K.sum(preds, axis=(-2, -1))))

        return multinomial_loss + self.c_task_weight * mse_loss

    def get_config(self):
        return {"c_task_weight": self.c_task_weight}


@gin.configurable
class PoissonMultinomialNLL:

    def __init__(self, c_task_weight=1):
        self.c_task_weight = c_task_weight

    def __call__(self, true_counts, preds):
        probs = preds / K.sum(preds, axis=-2, keepdims=True)
        logits = K.log(probs / (1 - probs))

        # multinomial loss
        multinomial_loss = multinomial_nll(true_counts, logits)

        poisson_loss = kloss.poisson(K.sum(true_counts, axis=(-2, -1)),
                                     K.sum(preds, axis=(-2, -1)))

        return multinomial_loss + self.c_task_weight * poisson_loss

    def get_config(self):
        return {"c_task_weight": self.c_task_weight}

@gin.configurable
class PearsonCorrelationLoss:
    def __init__(self, c_task_weight=1):
        self.c_task_weight = c_task_weight

    def __call__(self, true_counts, preds):
        probs = correlation_loss(true_counts, preds)
        return probs * self.c_task_weight

AVAILABLE = ["multinomial_nll",
             "CountsMultinomialNLL",
             "PoissonMultinomialNLL",
             "ignoreNaNloss",
             "correlation_loss"]


def get(name):
    try:
        return kloss.get(name)
    except ValueError:
        return get_from_module(name, globals())
