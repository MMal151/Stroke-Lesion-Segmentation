import tensorflow as tf


# Source: https://medium.com/@rekalantar/step-by-step-tutorial-liver-segmentation-on-ct-scans-using-tensorflow-d27bc61fbfe2
def dice_coef_single_label(y_true, y_pred, smooth=1.):
    """
  Dice = (2*|X & Y|)/ (|X|+ |Y|)
        =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
  ref: https://arxiv.org/pdf/1606.04797v1.pdf
  """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
