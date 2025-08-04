import os
import numpy as np
import matplotlib.pyplot as plt

# 加载预测结果
predictions_test = np.load("F:/study_1/train/simulation_test/predictions_test.npy")

# 加载标签数据
labels_test = np.load("F:\\Processed_Data\\Y_test.npy")

# 创建保存预测图的文件夹
save_base_dir = "F:\\study_1\\train\\simulation_test\\predictions_images"
os.makedirs(save_base_dir, exist_ok=True)

# 对每个工况进行处理
for i in range(predictions_test.shape[0]):
    # 获取当前工况的预测结果
    current_predictions = predictions_test[i]

    # 进行 RMS 运算
    rms_prediction = np.sqrt(np.mean(current_predictions ** 2, axis=0))

    # 确保 rms_prediction 是二维数组
    if rms_prediction.ndim == 3 and rms_prediction.shape[2] == 1:
        rms_prediction = np.squeeze(rms_prediction, axis=2)

    # 创建当前工况的保存子文件夹
    sample_dir = os.path.join(save_base_dir, f"sample_{i}_RMS")
    os.makedirs(sample_dir, exist_ok=True)

    # 保存预测图
    plt.imsave(os.path.join(sample_dir, "prediction.png"), rms_prediction, cmap='gray')

    # 获取当前工况的标签图
    current_label = labels_test[i]

    # 确保 current_label 是二维数组
    if current_label.ndim == 3 and current_label.shape[2] == 1:
        current_label = np.squeeze(current_label, axis=2)

    # 保存标签图
    plt.imsave(os.path.join(sample_dir, "label.png"), current_label, cmap='gray')

    print(f"工况 {i} 的预测图和标签图已保存到 {sample_dir}")

    