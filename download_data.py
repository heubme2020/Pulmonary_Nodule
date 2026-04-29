import os
import time
from datasets import load_dataset
from huggingface_hub import snapshot_download
import hf_transfer
from requests import ConnectionError
import aria2p
import subprocess
import socket
from tcia_utils import nbia

def download_hugging_face_data(data_name, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    while True:
        try:
            local_path = snapshot_download(
                repo_id=data_name,
                repo_type="dataset",
                local_dir=dst_folder,
                local_dir_use_symlinks=False,  # 避免使用符号链接，确保是真实文件
                token=""  # ← 替换为你的 token
            )
            break
        except ConnectionError as e:
            print(f"ConnectionError: {e}. Retrying in 1s...")
            time.sleep(1)

    print(f"\n✅ 数据集已完整下载到: {local_path}")


def download_tcia_data(data_name, dst_folder):
    data = nbia.getSeries(collection=data_name, modality='CT')
    # data = nbia.getSeries(modality="CT", api_url="nlst")
    # data = nbia.downloadSeries("manifest-NLST_allCT.tcia", input_type="manifest")
    print(len(data))
    idx = 0
    while idx < len(data):
        try:
            nbia.downloadSeries(data, number=idx)
            idx = idx + 1
        except:
            time.sleep(3)
            nbia.downloadSeries(data, number=idx)
    os.rename('tciaDownload', dst_folder)


if __name__ == '__main__':
    data_name = 'LIDC-IDRI'
    dst_folder = 'data/' + data_name
    download_tcia_data(data_name, dst_folder)

