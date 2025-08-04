import tensorflow as tf

def chamfer_distance(y_true, y_pred):
    # 将图像转换为二值点云
    def image_to_point_cloud(img):
        img = tf.squeeze(img, axis=-1)  # 去掉通道维度
        coords = tf.where(tf.greater(img, 0.5))  # 提取非零像素坐标
        return tf.cast(coords, tf.float32)

    # 提取 y_true 和 y_pred 的点云坐标
    y_true_points = image_to_point_cloud(y_true)
    y_pred_points = image_to_point_cloud(y_pred)

    # 扩展维度，计算欧几里得距离
    dist_matrix = tf.norm(tf.expand_dims(y_true_points, 1) - tf.expand_dims(y_pred_points, 0), axis=-1)

    # 计算 Chamfer 距离
    min_dist_true_to_pred = tf.reduce_mean(tf.reduce_min(dist_matrix, axis=1))
    min_dist_pred_to_true = tf.reduce_mean(tf.reduce_min(dist_matrix, axis=0))

    chamfer_loss = min_dist_true_to_pred + min_dist_pred_to_true
    return chamfer_loss
