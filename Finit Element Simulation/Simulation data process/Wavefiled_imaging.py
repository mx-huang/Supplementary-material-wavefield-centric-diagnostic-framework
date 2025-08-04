import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filter import filter_iir_bandpass
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import matplotlib

matplotlib.use('TkAgg')

# 设置全局字体
matplotlib.rcParams['font.family'] = 'SimSun'  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 处理负号显示


for i in range(200,240):
    # 读取CSV文件路径
    point_path = fr"F:\study_1\Simulation_data\Job_{i+1}_100kHz\simulation_coordinates.csv"
    displacement_path = fr"F:\study_1\Simulation_data\Job_{i+1}_100kHz\V3_displacement_output.csv"

    # 读取CSV文件
    df_point = pd.read_csv(point_path)
    df_displacement = pd.read_csv(displacement_path)

    # 获取坐标矩阵（假设节点编号在第一列，X坐标在第二列，Y坐标在第三列）
    df_selected_point = df_point.iloc[1:, 1:3]  # 如果CSV数据不同，请调整列索引
    point_data = df_selected_point.values

    # 坐标平移
    min_x = np.min(point_data[:, 0])
    max_x = np.max(point_data[:, 0])
    min_y = np.min(point_data[:, 1])
    max_y = np.max(point_data[:, 1])
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    point_data = point_data - [center_x, center_y]

    # 获取时间矩阵
    df_selected_time = df_displacement.columns[1:]
    time_data = df_selected_time.astype(float).values
    time_data = time_data * 1e6

    # 获取位移矩阵
    df_selected_displacement = df_displacement.iloc[1:, 1:]
    displacement_data = df_selected_displacement.values

    # filter_displacement_data = np.zeros((displacement_data.shape[0], displacement_data.shape[1]))

    # # 滤波
    # for m in range(displacement_data.shape[0]):
    #     filter_displacement_data[m] = filter_iir_bandpass(displacement_data[m])

    filtered_point_data = point_data

    # 可视化波场
    normalized_displacement_data = displacement_data

    # 计算坐标范围
    x_min, x_max = np.min(filtered_point_data[:, 0]), np.max(filtered_point_data[:, 0])
    y_min, y_max = np.min(filtered_point_data[:, 1]), np.max(filtered_point_data[:, 1])

    num_grid_points = 400

    grid_x, grid_y = np.mgrid[x_min:x_max:num_grid_points * 1j, y_min:y_max:num_grid_points * 1j]

    # 修改工作目录为 F: 盘根目录
    os.chdir("F:/")

    # 生成绝对路径
    output_folder = os.path.join("study_1\Simulation_data", f"Job_{i + 1}_100kHz", "wavefiled_image")
    output_folder = os.path.abspath(output_folder)  # 规范路径
    print(f"Creating folder: {output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.axis('off')  # 隐藏坐标轴
    ax.set_position([0, 0, 1, 1])

    cmap = plt.get_cmap('gray')
    norm = Normalize(vmin=-0.02, vmax=0.02)
    heatmap = ax.imshow(np.zeros((num_grid_points, num_grid_points)),
                        origin='lower', cmap=cmap, norm=norm, interpolation='bilinear')

    # 保存每一帧的波场图像
    for frame in range(normalized_displacement_data.shape[1]):
        displacement = normalized_displacement_data[:, frame]
        grid_z = griddata((filtered_point_data[:, 0], filtered_point_data[:, 1]), displacement,
                          (grid_x, grid_y), method='linear')
        if grid_z is None or grid_z.shape != (num_grid_points, num_grid_points):
            grid_z = np.nan * np.zeros((num_grid_points, num_grid_points))

        heatmap.set_array(grid_z.T)

        # 保存图像时输出完整路径
        time_label = f"t={time_data[frame]:.2f} us"
        output_path = os.path.join(output_folder, f"{time_label}.png")
        print(f"Saving image to: {output_path}")  # 输出路径
        plt.savefig(output_path, dpi=100)

    # 关闭图形窗口
    plt.close()

for i in range(200,240):
    # 读取CSV文件路径
    point_path = fr"F:\study_1\Simulation_data\Job_{i+1}_150kHz\simulation_coordinates.csv"
    displacement_path = fr"F:\study_1\Simulation_data\Job_{i+1}_150kHz\V3_displacement_output.csv"

    # 读取CSV文件
    df_point = pd.read_csv(point_path)
    df_displacement = pd.read_csv(displacement_path)

    # 获取坐标矩阵（假设节点编号在第一列，X坐标在第二列，Y坐标在第三列）
    df_selected_point = df_point.iloc[1:, 1:3]  # 如果CSV数据不同，请调整列索引
    point_data = df_selected_point.values

    # 坐标平移
    min_x = np.min(point_data[:, 0])
    max_x = np.max(point_data[:, 0])
    min_y = np.min(point_data[:, 1])
    max_y = np.max(point_data[:, 1])
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    point_data = point_data - [center_x, center_y]

    # 获取时间矩阵
    df_selected_time = df_displacement.columns[1:]
    time_data = df_selected_time.astype(float).values
    time_data = time_data * 1e6

    # 获取位移矩阵
    df_selected_displacement = df_displacement.iloc[1:, 1:]
    displacement_data = df_selected_displacement.values

    # filter_displacement_data = np.zeros((displacement_data.shape[0], displacement_data.shape[1]))

    # # 滤波
    # for m in range(displacement_data.shape[0]):
    #     filter_displacement_data[m] = filter_iir_bandpass(displacement_data[m])

    filtered_point_data = point_data

    # 可视化波场
    normalized_displacement_data = displacement_data

    # 计算坐标范围
    x_min, x_max = np.min(filtered_point_data[:, 0]), np.max(filtered_point_data[:, 0])
    y_min, y_max = np.min(filtered_point_data[:, 1]), np.max(filtered_point_data[:, 1])

    num_grid_points = 400

    grid_x, grid_y = np.mgrid[x_min:x_max:num_grid_points * 1j, y_min:y_max:num_grid_points * 1j]

    # 修改工作目录为 F: 盘根目录
    os.chdir("F:/")

    # 生成绝对路径
    output_folder = os.path.join("study_1\Simulation_data", f"Job_{i + 1}_150kHz", "wavefiled_image")
    output_folder = os.path.abspath(output_folder)  # 规范路径
    print(f"Creating folder: {output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.axis('off')  # 隐藏坐标轴
    ax.set_position([0, 0, 1, 1])

    cmap = plt.get_cmap('gray')
    norm = Normalize(vmin=-0.02, vmax=0.02)
    heatmap = ax.imshow(np.zeros((num_grid_points, num_grid_points)),
                        origin='lower', cmap=cmap, norm=norm, interpolation='bilinear')

    # 保存每一帧的波场图像
    for frame in range(normalized_displacement_data.shape[1]):
        displacement = normalized_displacement_data[:, frame]
        grid_z = griddata((filtered_point_data[:, 0], filtered_point_data[:, 1]), displacement,
                          (grid_x, grid_y), method='linear')
        if grid_z is None or grid_z.shape != (num_grid_points, num_grid_points):
            grid_z = np.nan * np.zeros((num_grid_points, num_grid_points))

        heatmap.set_array(grid_z.T)

        # 保存图像时输出完整路径
        time_label = f"t={time_data[frame]:.2f} us"
        output_path = os.path.join(output_folder, f"{time_label}.png")
        print(f"Saving image to: {output_path}")  # 输出路径
        plt.savefig(output_path, dpi=100)

    # 关闭图形窗口
    plt.close()