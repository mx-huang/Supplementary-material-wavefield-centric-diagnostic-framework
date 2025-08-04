from abaqus import *
from abaqusConstants import *
import odbAccess
import csv
import os
import re

#统计Job数量
# 设置文件夹路径
# folder_path = r"E:\temp"  # 替换为你的文件夹路径
#
# # 使用正则表达式来匹配命名规则为 "Job-n.odb" 的文件，其中 n 为阿拉伯数字
# pattern = re.compile(r"Job-(\d+)\.odb$")
#
# # 初始化计数器
# file_count = 0
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 检查文件是否是 .odb 文件，并且符合 "Job-n.odb" 的命名规则
#     if pattern.match(filename):
#         file_count += 1
#
# # 输出符合规则的文件数量
# print(f"Found {file_count} files matching the pattern 'Job-n.odb' in the folder.")
for i in range(0,240):
    # 打开Odb文件
    odb_path = f"E:/temp/Job-{i+1}.odb"  # 请替换为你的实际odb文件路径
    odb = odbAccess.openOdb(odb_path)

    # 指定输出的节点集名称
    set_name = 'SUR_WAVE'  # 请确保此名称与模型中的名称一致
    # 创建保存路径，确保文件夹存在
    output_folder = f"F:/Simulation_data/Job_{i + 1}"
    os.makedirs(output_folder, exist_ok=True)  # 创建文件夹，如果已经存在则不会抛出错误
    output_file = os.path.join(output_folder, "V3_displacement_output.csv")  # 创建完整的文件路径

    # 获取节点集
    try:
        node_set = odb.rootAssembly.nodeSets[set_name]
    except KeyError:
        print(f"Error: The specified set '{set_name}' does not exist.")
        odb.close()
        raise

    # 获取时间步和所有节点的U3位移
    all_times = []
    node_labels = [node.label for node in node_set.nodes[0]]  # 获取所有节点ID
    v3_displacements = {label: [] for label in node_labels}  # 存储每个节点的U3位移

    for step_name, step in odb.steps.items():
        print(f"Processing step: {step_name}")
        for frame in step.frames:
            time = frame.frameValue
            all_times.append(time)
            print(f"Processing frame at time: {time}")

            # 获取每个节点的U3位移
            if 'V' in frame.fieldOutputs:
                v_field = frame.fieldOutputs['V']
            else:
                print("No displacement field (V) found in this frame.")
                continue

            v_subset = v_field.getSubset(region=node_set)  # 获取节点集的位移数据
            for node_value in v_subset.values:
                node_id = node_value.nodeLabel
                v3_value = node_value.data[2]  # U3 是第三个元素（z方向位移）
                v3_displacements[node_id].append(v3_value)

    # 创建CSV文件并写入数据
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            # 写入时间序列（第一行）
            header = ["NodeID"] + all_times
            writer.writerow(header)

            # 写入每个节点的U3位移数据
            for label in node_labels:
                row = [label] + v3_displacements[label]
                writer.writerow(row)

        print(f"位移数据已成功导出到 {output_file}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")

    # 关闭Odb文件
    odb.close()
