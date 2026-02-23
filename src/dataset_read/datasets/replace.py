import re
import argparse

def replace_strings_in_file_inplace(file_path):
    """
    将文本文件中所有 xx/pano_skybox_color 替换为 mp3d_skybox/xx/matterport_stitched_images。
    
    Args:
        file_path (str): 输入的txt文件路径（原地修改）。
    """
    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # # 使用正则表达式进行替换  Vt2qJdWjCF2/pano_depth
    # pattern = r'(\w+)/pano_skybox_color'
    # replacement = r'mp3d_skybox/\1/matterport_stitched_images'
    # new_content = re.sub(pattern, replacement, content)

    pattern = r'(\w+)/pano/depth'
    replacement = r'\1/depth'
    new_content = re.sub(pattern, replacement, content)
    # pattern = r'Matterport3D_depth/(\w+)/matterport_stitched_images'
    # replacement = r'mp3d_skybox/\1/matterport_stitched_images'
    # new_content = re.sub(pattern, replacement, content)

    # pattern = r'\.jpg'
    # replacement = '.png'
    # new_content = re.sub(pattern, replacement, content)
    
    # 将替换后的内容写回原文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)
    print(f"文件 {file_path} 已成功更新！")

if __name__ == "__main__":
    # 使用 argparse 获取命令行参数
    parser = argparse.ArgumentParser(description="将 xx/pano_skybox_color 替换为 mp3d_skybox/xx/matterport_stitched_images（原地修改文件）")
    parser.add_argument("file_path", type=str, help="需要替换的txt文件路径")
    
    args = parser.parse_args()
    
    # 调用替换函数
    replace_strings_in_file_inplace(args.file_path)