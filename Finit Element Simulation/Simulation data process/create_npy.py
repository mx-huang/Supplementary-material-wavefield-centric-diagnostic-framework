import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_wavefield_single_job(job_dir, label_path, img_size=(400, 400), total_frames=301):
    """
    参数:
        job_dir (str): 单个工况的波场图像文件夹路径。
        label_path (str): 标签图像路径。
        img_size (tuple): 图像大小 (宽, 高)。
        total_frames (int): 每个工况的时间序列帧数。
    返回:
        wavefield (ndarray): 单个工况的波场数据 (301, 400, 400, 1)。
        label (ndarray): 单个工况的标签数据 (400, 400, 1)。
    """
    wavefield = []
    for t in range(total_frames):
        img_path = os.path.join(job_dir, f"t={t * 0.5:.2f} us.png")
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        img = cv2.resize(img, img_size) / 255.0  # 归一化到 [0, 1]
        wavefield.append(img)

    if len(wavefield) == total_frames:
        wavefield = np.expand_dims(np.array(wavefield, dtype="float16"), axis=-1)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, img_size) / 255.0  # 归一化标签
        label = np.expand_dims(label, axis=-1)
        return wavefield, label
    else:
        print(f"工况数据不足: {job_dir}")
        return None, None

def process_and_save_data(base_dir, output_dir, test_size=0.2, img_size=(400, 400), total_frames=301):
    """
    加载所有工况的数据，将数据分割为训练集和测试集，并保存为 .npy 文件。
    参数:
        base_dir (str): 数据集的根目录。
        output_dir (str): 输出 .npy 文件的保存目录。
        test_size (float): 测试集的比例 (0.0 - 1.0)。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化数据存储
    all_wavefields = []
    all_labels = []

    # 遍历所有工况
    for i in range(201, 241):
        print(f"正在处理 Job_{i}...")
        job_dir = os.path.join(base_dir, f"Job_{i}_150kHz", "wavefiled_image")
        label_path = os.path.join(base_dir, f"Job_{i}_150kHz", f"label_{i}.png")

        X, y = load_wavefield_single_job(job_dir, label_path, img_size, total_frames)
        if X is not None and y is not None:
            all_wavefields.append(X)
            all_labels.append(y)
        else:
            print(f"跳过: Job_{i}")

    # 转换为 NumPy 数组
    all_wavefields = np.array(all_wavefields, dtype="float16")
    all_labels = np.array(all_labels, dtype="float16")

    print(f"所有数据形状: 波场数据 {all_wavefields.shape}, 标签数据 {all_labels.shape}")

    X_test_150kHz, Y_test_150kHz = all_wavefields, all_labels

    # 保存数据到 .npy 文件
    np.save(os.path.join(output_dir, " X_test_150kHz.npy"),  X_test_150kHz)
    np.save(os.path.join(output_dir, " Y_test_150kHz.npy"),  Y_test_150kHz)

    print("数据已成功保存为 .npy 文件")
    print(f"测试集形状: X_train { X_test_150kHz.shape}, Y_train {Y_test_150kHz.shape}")


if __name__ == "__main__":
    base_dir = "F:/study_1/Simulation_data"
    output_dir = "F:/study_1/Processed_Data"
    process_and_save_data(base_dir, output_dir)