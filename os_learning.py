import os 
from tqdm import tqdm

# 使用os.walk和多个tqdm完成对于目录下所有文件的处理

def use_two_tqdm(root_dir):
    """
    遍历指定目录下的所有文件，并使用两个tqdm进度条来显示目录和文件的处理进度。
    Args:
        root_dir (str): 要遍历的根目录路径。
    """
    folders = [dirpath for dirpath,_,_ in os.walk(root_dir)]
    for dirpath in tqdm(folders, desc="Processing directories",position=0,leave = False):
        files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        for file in tqdm(files, desc="Processing files", position=1):
            file_path = os.path.join(dirpath, file)
            print(f"Processing file: {file_path}")
            pass

use_two_tqdm("/home/yuyue/project/flunet_py")