from abaqus import *
from abaqusConstants import *
import odbAccess
import csv
import os
import re

#统计Job数量
# 设置文件夹路径
folder_path = r"E:\temp"  # 替换为你的文件夹路径

for i in range(0,240):
    # 打开Odb文件
    odb_path = f"E:/temp/Job-{i+1}.odb"  # 请替换为你的实际odb文件路径
    odb = odbAccess.openOdb(odb_path)

    # 检查可用的nodeSets
    print("Available node sets:")
    for set_name in odb.rootAssembly.nodeSets.keys():
        print(set_name)

    # 获取set
    try:
        sur_wave_set = odb.rootAssembly.nodeSets['SUR_WAVE']  # 确保此名称正确
    except KeyError:
        print("Error: The specified set 'SUR_WAVE' does not exist.")
        odb.close()
        raise

    # 创建保存路径，确保文件夹存在
    output_folder = f"F:/Simulation_data/Job_{i + 1}"
    os.makedirs(output_folder, exist_ok=True)  # 创建文件夹，如果已经存在则不会抛出错误
    output_file = os.path.join(output_folder, "simulation_coordinates.csv")  # 创建完整的文件路径

    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            # 写入表头
            writer.writerow(['NodeID', 'X', 'Y', 'Z'])

            # 保存节点坐标到CSV文件
            for node in sur_wave_set.nodes[0]:
                node_id = node.label  # 获取节点编号
                coords = node.coordinates  # 获取节点坐标
                writer.writerow([node_id, coords[0], coords[1], coords[2]])

        print(f"SUR_WAVE set的节点信息已保存到 {output_file}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")


    # 关闭Odb文件
    odb.close()
