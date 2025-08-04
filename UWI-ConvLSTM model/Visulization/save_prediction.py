import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 导入预测结果和标签数据
predictions_test = np.load("F:/study_1/train/simulation_test/predictions_test.npy")
y_test = np.load("F:/Processed_Data/Y_test.npy")

print(f"预测结果的形状: {predictions_test.shape}")  # (40, 76, 400, 400, 1)
print(f"标签数据的形状: {y_test.shape}")  # (40, 400, 400, 1)

# 2. 创建文件夹并保存图片
save_base_dir = "F:/study_1/train/simulation_test/predictions_images_1"
os.makedirs(save_base_dir, exist_ok=True)

for sample_idx in range(predictions_test.shape[0]):
    sample_dir = os.path.join(save_base_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # 保存预测结果图片
    for window_idx in range(predictions_test.shape[1]):
        prediction = predictions_test[sample_idx, window_idx, :, :, :]
        prediction = np.squeeze(prediction, axis=-1)
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())

        pred_img_filename = f"pred_window_{window_idx}.png"
        pred_img_filepath = os.path.join(sample_dir, pred_img_filename)
        plt.imsave(pred_img_filepath, prediction, cmap='gray')

    # 保存标签数据图片
    label = y_test[sample_idx, :, :, :]
    label = np.squeeze(label, axis=-1)
    label = (label - label.min()) / (label.max() - label.min())

    label_img_filename = "label.png"
    label_img_filepath = os.path.join(sample_dir, label_img_filename)
    plt.imsave(label_img_filepath, label, cmap='gray')

    print(f"工况 {sample_idx} 的预测结果和标签已保存到 {sample_dir}")