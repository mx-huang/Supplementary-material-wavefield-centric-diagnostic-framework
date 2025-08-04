import tensorflow as tf
from chamfer_distance import chamfer_distance

# Tversky 损失函数
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, epsilon=1e-6):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 计算真正例 (TP)、假阳性 (FP)、假阴性 (FN)
    true_pos = tf.reduce_sum(y_true * y_pred)
    false_pos = tf.reduce_sum((1 - y_true) * y_pred)
    false_neg = tf.reduce_sum(y_true * (1 - y_pred))

    # 计算 Tversky 指数
    tversky_index = true_pos / (true_pos + alpha * false_neg + beta * false_pos + epsilon)

    # 计算 Tversky 损失
    tversky_loss = 1 - tversky_index

    return tversky_loss

# 定义具名的 Tversky 损失函数
def tversky_loss_fixed(y_true, y_pred):
    return tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, epsilon=1e-6)

