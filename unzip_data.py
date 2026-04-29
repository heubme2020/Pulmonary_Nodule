import zipfile
import os


def unzip_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(folder_path, file)
            extract_dir = os.path.join(folder_path, file[:-4])  # 去掉.zip后缀作为文件夹名
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"解压完成: {file} → {extract_dir}")


if __name__ == '__main__':
    # 目标文件夹路径
    folder_path = 'train/LNDb'
    unzip_folder(folder_path)