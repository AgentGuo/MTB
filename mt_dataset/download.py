import os
import pandas as pd

import gdown

# 文件 ID
folder_id = "1hn4jsjJQmZMAPJV3MKMi5tL5ey_9Spq5"
# 输出文件名
output_path = os.path.expanduser("~/.mtb")

def check_and_download_data():
    """
    check and download dataset
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    need_download = False
    for file in ["mtb_orig.csv", "mtb_w.csv", "mtb_wh.csv"]:
        if not os.path.exists(os.path.join(f'{output_path}/', file)):
            need_download = True
    if need_download:
        gdown.download_folder(f'https://drive.google.com/drive/folders/{folder_id}', output=output_path, quiet=False)
    return output_path