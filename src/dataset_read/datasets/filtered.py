import os

# 输入和输出路径
input_dir = "/data/pengfei_wang/HQ_pano"
output_file = "/home/pengfei_wang/PanDA/datasets/HQ_pano.txt"

# 初始化一个列表存储符合条件的文件路径s
filtered_files = []

# 遍历目录及其子目录
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".png") and "Structured3D" not in file:
            # 构建完整路径
            file_path = os.path.join(root, file)
            filtered_files.append(file_path)

# 将结果写入文件，每行一个文件路径
with open(output_file, 'w') as out_file:
    for file_path in filtered_files:
        out_file.write(file_path + '\n')

# 打印完成信息
print(f"筛选完成，共保存 {len(filtered_files)} 个文件路径到 {output_file}")