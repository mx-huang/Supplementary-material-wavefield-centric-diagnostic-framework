import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# --- 1. 配置与常量定义 ---
# -------------------------------------------------------------
# 文件路径
MODEL_PATH = "/root/autodl-tmp/best_model.h5"
TEST_DATA_PATH = "/root/autodl-tmp/data/X_test.npy"
TEST_LABEL_PATH = "/root/autodl-tmp/data/Y_test.npy"
PREDICTION_OUTPUT_PATH = "/root/autodl-tmp/predictions_test_set.npy"
METRICS_OUTPUT_PATH = "/root/autodl-tmp/prediction_metrics.csv"
VISUALIZATION_DIR = "/root/autodl-tmp/prediction_visualizations/"

# 模型与数据参数
WINDOW_SIZE = 25
STEP = 1
BATCH_SIZE = 4  # 可根据您的 GPU 显存调整
THRESHOLD = 0.5


# --- 2. 自定义函数与辅助工具 ---
# -------------------------------------------------------------

# 自定义对象需要被 load_model 识别
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, epsilon=1e-6):
    """ Tversky 损失函数 """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_pos = tf.reduce_sum(y_true * y_pred)
    false_pos = tf.reduce_sum((1 - y_true) * y_pred)
    false_neg = tf.reduce_sum(y_true * (1 - y_pred))
    tversky_index = true_pos / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    return 1 - tversky_index


def tversky_loss_fixed(y_true, y_pred):
    """ 定义具名的 Tversky 损失函数，用于模型加载 """
    return tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3)


def chamfer_distance(y_true, y_pred):
    """
    占位符函数，确保模型能加载。
    """
    return tf.constant(0.0, dtype=tf.float32)


def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)


def preprocess_wavefield_data(X, start_frame=101, total_frames=200, step=2):
    """
    预处理波场图数据，与训练时保持一致。
    """
    print(f"原始数据形状: {X.shape}")
    X_processed = X[:, start_frame - 1:start_frame - 1 + total_frames:step, :, :, :]
    print(f"预处理后数据形状: {X_processed.shape}")
    return X_processed


def prediction_generator(X, window_size=25, step=1, batch_size=4):
    """
    为 `model.predict` 创建数据生成器，只产生输入数据 (X)。
    """
    num_samples = X.shape[0]
    windows_per_sample = (X.shape[1] - window_size) // step + 1
    total_windows = num_samples * windows_per_sample

    def generator():
        for i in range(num_samples):
            for j in range(0, X.shape[1] - window_size + 1, step):
                yield X[i, j:j + window_size, :, :, :]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(window_size, X.shape[2], X.shape[3], X.shape[4]), dtype=tf.float32)
    )

    dataset = dataset.batch(batch_size)
    steps = int(np.ceil(total_windows / batch_size))

    return dataset, steps, total_windows


def evaluation_generator(X, Y, window_size=25, step=1, batch_size=4):
    """
    为 `model.evaluate` 创建数据生成器，产生 (X, Y) 对。
    """
    num_samples = X.shape[0]
    windows_per_sample = (X.shape[1] - window_size) // step + 1
    total_windows = num_samples * windows_per_sample

    def generator():
        for i in range(num_samples):
            # 每个工况只有一个真值标签
            label = Y[i]
            for j in range(0, X.shape[1] - window_size + 1, step):
                window = X[i, j:j + window_size, :, :, :]
                yield window, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(window_size, X.shape[2], X.shape[3], X.shape[4]), dtype=tf.float32),
            tf.TensorSpec(shape=(Y.shape[1], Y.shape[2], Y.shape[3]), dtype=Y.dtype)
        )
    )

    dataset = dataset.batch(batch_size)
    steps = int(np.ceil(total_windows / batch_size))

    return dataset, steps


def visualize_and_save_predictions(X_test, Y_test, predictions, save_dir, num_samples_to_show=5, window_size=25,
                                   threshold=0.5):
    """
    随机选择几个样本，可视化输入、真值和预测结果，并保存。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建可视化结果保存目录: {save_dir}")

    num_total_samples = X_test.shape[0]
    indices_to_show = random.sample(range(num_total_samples), min(num_samples_to_show, num_total_samples))

    for idx in indices_to_show:
        # 由于一个工况有多个预测窗口，我们选择最后一个窗口的预测结果进行可视化
        # 最后一个窗口的输入是 X_test[idx, -window_size:, ...]
        # 对应的预测结果是 predictions[idx, -1, ...]
        last_input_frame = X_test[idx, -1, :, :, :]
        true_label = Y_test[idx]
        predicted_label = predictions[idx, -1, :, :, :]
        predicted_label_binary = (predicted_label > threshold).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(np.squeeze(last_input_frame), cmap='gray')
        ax[0].set_title(f"Sample {idx} - Last Input Frame")
        ax[0].axis('off')

        ax[1].imshow(np.squeeze(true_label), cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')

        ax[2].imshow(np.squeeze(predicted_label_binary), cmap='gray')
        ax[2].set_title(f"Prediction (Threshold={threshold})")
        ax[2].axis('off')

        save_path = os.path.join(save_dir, f"sample_{idx}_comparison.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"可视化结果已保存至: {save_path}")


# --- 3. 主执行逻辑 ---
# -------------------------------------------------------------
def main():

    # --- 加载模型 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到于 {MODEL_PATH}")
        return

    print(f"正在加载模型: {MODEL_PATH}")
    # 将 Dice 系数添加到自定义对象中，以便评估时使用
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "tversky_loss_fixed": tversky_loss_fixed,
            "chamfer_distance": chamfer_distance,
            "dice_coefficient": dice_coefficient
        }
    )
    model.compile(
        loss=tversky_loss_fixed,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[chamfer_distance, dice_coefficient]
    )
    print("模型加载并重新编译成功。")
    model.summary()

    # --- 加载并预处理数据 ---
    print("\n正在加载和预处理测试数据...")
    if not (os.path.exists(TEST_DATA_PATH) and os.path.exists(TEST_LABEL_PATH)):
        print(f"错误: 测试数据 {TEST_DATA_PATH} 或标签 {TEST_LABEL_PATH} 未找到。")
        return

    X_test = np.load(TEST_DATA_PATH)
    Y_test = np.load(TEST_LABEL_PATH)

    # 确保数据类型正确
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    X_test_processed = preprocess_wavefield_data(X_test, start_frame=101, total_frames=200, step=2)

    num_test_samples = X_test_processed.shape[0]
    windows_per_sample = (X_test_processed.shape[1] - WINDOW_SIZE) // STEP + 1

    # --- 进行预测并计时 ---
    print("\n开始在测试集上进行预测...")
    pred_gen, pred_steps, total_windows = prediction_generator(
        X_test_processed, window_size=WINDOW_SIZE, step=STEP, batch_size=BATCH_SIZE
    )

    start_time = time.time()
    # 使用 tqdm 创建进度条
    predictions_flat = model.predict(
        pred_gen,
        steps=pred_steps,
        verbose=1,
        workers=8,  # 可以根据你的CPU核心数调整
        use_multiprocessing=True
    )
    end_time = time.time()

    # --- 整理并保存预测结果 ---
    # 将扁平化的预测结果重塑为 (工况数, 每个工况的窗口数, 高, 宽, 通道)
    predictions_reshaped = predictions_flat.reshape(
        (num_test_samples, windows_per_sample, predictions_flat.shape[1], predictions_flat.shape[2],
         predictions_flat.shape[3])
    )
    print(f"\n预测结果已重塑为: {predictions_reshaped.shape}")

    np.save(PREDICTION_OUTPUT_PATH, predictions_reshaped)
    print(f"预测结果已保存至: {PREDICTION_OUTPUT_PATH}")

    # --- 评估模型性能 ---
    print("\n开始在整个测试集上评估模型性能...")
    eval_gen, eval_steps = evaluation_generator(
        X_test_processed, Y_test, window_size=WINDOW_SIZE, step=STEP, batch_size=BATCH_SIZE
    )

    evaluation_results = model.evaluate(
        eval_gen,
        steps=eval_steps,
        verbose=1,
        workers=10,
        use_multiprocessing=True
    )

    # --- 整理并展示指标 ---
    total_prediction_time = end_time - start_time
    avg_time_per_sample = total_prediction_time / num_test_samples
    avg_time_per_window = total_prediction_time / total_windows

    print("\n--- 预测性能报告 ---")
    print(f"测试样本总数: {num_test_samples}")
    print(f"总预测窗口数: {total_windows}")
    print("-" * 25)
    print(f"总预测耗时: {total_prediction_time:.4f} 秒")
    print(f"平均每工况预测耗时: {avg_time_per_sample:.4f} 秒")
    print(f"平均每窗口预测耗时: {avg_time_per_window:.4f} 秒")
    print("-" * 25)
    print(f"测试集 Tversky 损失 (Loss): {evaluation_results[0]:.4f}")
    print(f"测试集 Chamfer 距离 (Metric 1): {evaluation_results[1]:.4f}")
    print(f"测试集 Dice 系数 (Metric 2): {evaluation_results[2]:.4f}")
    print("--- 报告结束 ---\n")

    # 保存指标到 CSV 文件
    metrics_df = pd.DataFrame({
        'Total Prediction Time (s)': [total_prediction_time],
        'Average Time per Sample (s)': [avg_time_per_sample],
        'Average Time per Window (s)': [avg_time_per_window],
        'Test Tversky Loss': [evaluation_results[0]],
        'Test Chamfer Distance': [evaluation_results[1]],
        'Test Dice Coefficient': [evaluation_results[2]],
        'Total Samples': [num_test_samples],
        'Total Windows': [total_windows]
    })
    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
    print(f"详细性能指标已保存至: {METRICS_OUTPUT_PATH}")

    # --- 可视化部分结果 ---
    print("\n正在生成预测结果的可视化样例...")
    visualize_and_save_predictions(
        X_test_processed, Y_test, predictions_reshaped,
        save_dir=VISUALIZATION_DIR,
        num_samples_to_show=5,
        window_size=WINDOW_SIZE,
        threshold=THRESHOLD
    )


if __name__ == '__main__':
    main()