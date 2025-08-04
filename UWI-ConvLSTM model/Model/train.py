import time  ### 新增：导入Python的时间模块 ###
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger

# 假设这些是您自己的模块, 确保它们存在于您的项目中
from chamfer_distance import chamfer_distance
from networks import convlstm_model

# 配置 GPU 按需分配显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 启用混合精度训练
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class TimingCallback(Callback):
    """一个用于记录每个 epoch 耗时的回调函数"""
    def on_epoch_begin(self, epoch, logs=None):
        # 在每个 epoch 开始时，记录当前时间
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # 在每个 epoch 结束时，计算耗时并添加到日志字典中
        # Keras 的其他回调（如 CSVLogger）会自动读取这个字典
        epoch_duration = time.time() - self.epoch_start_time
        logs['epoch_duration_s'] = round(epoch_duration, 2)  # 单位为秒


# 加载数据
print("加载训练数据...")
X_train = np.load("/root/autodl-tmp/data/X_train.npy")
Y_train = np.load("/root/autodl-tmp/data/Y_train.npy")
X_val = np.load("/root/autodl-tmp/data/X_val.npy")
Y_val = np.load("/root/autodl-tmp/data/Y_val.npy")


# 数据筛选与切分规则
def preprocess_wavefield_data(X, start_frame=101, total_frames=200, step=2):
    X_processed = X[:, start_frame - 1:start_frame - 1 + total_frames:step, :, :, :]
    return X_processed


# 处理训练集和测试集
X_train = preprocess_wavefield_data(X_train)
X_val = preprocess_wavefield_data(X_val)

print(f"处理后的训练数据形状: {X_train.shape}, 标签形状: {Y_train.shape}")
print(f"处理后的测试数据形状: {X_val.shape}, 标签形状: {Y_val.shape}")


# 数据生成器函数
def data_generator(X, y, window_size=25, step=1, batch_size=4):
    num_samples = X.shape[0]
    while True:
        for i in range(num_samples):
            X_windows = []
            y_windows = []
            for j in range(0, X.shape[1] - window_size + 1, step):
                X_windows.append(X[i, j:j + window_size, :, :, :])
                y_windows.append(y[i])

            for k in range(0, len(X_windows), batch_size):
                X_batch = np.array(X_windows[k:k + batch_size])
                y_batch = np.array(y_windows[k:k + batch_size])
                yield X_batch, y_batch


# 自定义可视化回调
class VisualizePrediction(Callback):
    def __init__(self, X_val, y_val, window_size=25, threshold=0.5, save_dir='/root/autodl-tmp/visualizations'):
        super(VisualizePrediction, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.window_size = window_size
        self.threshold = threshold
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        idx = random.randint(0, self.X_val.shape[0] - 1)
        input_sample = self.X_val[idx:idx + 1, :self.window_size, :, :, :]
        true_label = self.y_val[idx]
        predicted_label = self.model.predict(input_sample, verbose=0)[0]
        predicted_label = np.where(predicted_label > self.threshold, 1, 0)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(np.squeeze(input_sample[:, -1, :, :, :]), cmap='gray')
        ax[0].set_title("Input Frame (Last Frame)")
        ax[1].imshow(np.squeeze(true_label), cmap='gray')
        ax[1].set_title("True Label")
        ax[2].imshow(np.squeeze(predicted_label), cmap='gray')
        ax[2].set_title("Predicted Label")
        plt.suptitle(f"Epoch {epoch + 1}")
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch + 1}.png'))
        plt.close(fig)


# Tversky 损失函数
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, epsilon=1e-6):
    """
    计算 Tversky 损失函数
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_pos = tf.reduce_sum(y_true * y_pred)
    false_pos = tf.reduce_sum((1 - y_true) * y_pred)
    false_neg = tf.reduce_sum(y_true * (1 - y_pred))
    tversky_index = true_pos / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    tversky_loss = 1 - tversky_index
    return tversky_loss


# 定义具名的 Tversky 损失函数
def tversky_loss_fixed(y_true, y_pred):
    return tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, epsilon=1e-6)


# 定义文件路径
checkpoint_path = "/root/autodl-tmp/best_model.h5"
log_path = "/root/autodl-tmp/training_log.csv"

# 检查是否存在已保存的模型
if os.path.exists(checkpoint_path):
    print(f"加载已有的模型: {checkpoint_path}")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={
            "chamfer_distance": chamfer_distance,
            "tversky_loss_fixed": tversky_loss_fixed
        }
    )
else:
    print("未找到现有的模型，初始化新模型。")
    input_shape = (25, 400, 400, 1)
    model = convlstm_model(input_shape=input_shape)
    model.compile(
        loss=tversky_loss_fixed,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
        metrics=[chamfer_distance]
    )

# 设置数据生成器
window_size = 25
batch_size = 4
train_gen = data_generator(X_train, Y_train, window_size=window_size, step=1, batch_size=batch_size)
val_gen = data_generator(X_val, Y_val, window_size=window_size, step=1, batch_size=batch_size)

# 计算每个 epoch 的步数
train_steps = (X_train.shape[0] * (X_train.shape[1] - window_size + 1)) // batch_size
val_steps = (X_val.shape[0] * (X_val.shape[1] - window_size + 1)) // batch_size

print("每个 epoch 的训练步数:", train_steps)
print("每个 epoch 的验证步数:", val_steps)

# 定义回调
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', verbose=1),
    VisualizePrediction(X_val, Y_val, save_dir='/root/autodl-tmp/visualizations'),
    CSVLogger(log_path, append=True),
    TimingCallback()
]

# 开始训练
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=100,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks
)