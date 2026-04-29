import SimpleITK as sitk
import os
import uuid
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import xml.etree.ElementTree as ET


def nii_to_mha(folder):
    nii_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(('.nii', '.nii.gz'))]
    for nii_file in nii_files:
        image = sitk.ReadImage(nii_file)
        random_name = str(uuid.uuid4()) + ".mha"
        output_path = os.path.join(folder, random_name)

        # Write as MHA file
        sitk.WriteImage(image, output_path)
        os.remove(nii_file)


def mhd_to_mha(folder, rename=True):
    mhd_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(('.mhd'))]
    for mhd_file in tqdm(mhd_files):
        image = sitk.ReadImage(mhd_file)
        if rename:
            random_name = str(uuid.uuid4()) + ".mha"
            output_path = os.path.join(folder, random_name)

            # Write as MHA file
            sitk.WriteImage(image, output_path)
            os.remove(mhd_file)
        else:
            output_path = mhd_file.replace('.mhd', '.mha')
            sitk.WriteImage(image, output_path)
            os.remove(mhd_file)


def dicom_to_mha(folder, min_files=31, rename=True):
    # 第一步：找到所有候选系列目录（最终的叶子文件夹）
    series_dirs = []
    for root, dirs, files in os.walk(folder):
        if not dirs and any(f.lower().endswith('.dcm') for f in files):  # 是叶子节点且含DICOM
            series_dirs.append(root)

    print(f"找到 {len(series_dirs)} 个DICOM系列目录")

    # 第二步：处理每个系列
    for series_dir in series_dirs:
        try:
            dicom_files = list(Path(series_dir).glob('*.dcm'))

            if len(dicom_files) < min_files:
                # 删除不足min_files的系列
                shutil.rmtree(series_dir)
                print(f"删除不足文件的系列: {series_dir} ({len(dicom_files)} files)")
            else:
                # 转换为MHA（随机命名）
                try:
                    if rename:
                        random_name = str(uuid.uuid4()) + ".mha"
                        output_path = os.path.join(folder, random_name)
                    else:
                        output_path = series_dir + ".mha"

                    # 读取并转换
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames([str(f) for f in dicom_files])
                    image = reader.Execute()
                    sitk.WriteImage(image, output_path)

                    print(f"转换成功: {series_dir} -> {output_path}")
                    shutil.rmtree(series_dir)
                except Exception as e:
                    print(f"转换失败 {series_dir}: {str(e)}")
        except:
            pass


def mp4_to_mha(folder):
    mp4_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(('.mp4'))]
    for mp4_file in mp4_files:
        frames = []
        # mp4_folder = mp4_file.replace('.mp4', '')
        # os.makedirs(mp4_folder, exist_ok=True)
        cap = cv2.VideoCapture(mp4_file)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 保存为PNG（无损）
            # frame_name = mp4_folder + '/' + str(frame_count) + '.png'
            # cv2.imwrite(frame_name, frame)
            frames.append(frame)
            frame_count += 1

        cap.release()
        print(f"Total frames extracted: {frame_count}")
        print(len(frames))
        volume = np.stack(frames, axis=0)  # 构建3D数组

        # 转换为医学影像（需设置空间参数）
        image = sitk.GetImageFromArray(volume)
        # 3. 必须设置的空间参数（示例值，需根据实际调整）
        image.SetSpacing([1.0, 1.0, 1.0])  # (x,y,z) 间距（单位：mm）
        image.SetOrigin([0, 0, 0])  # 坐标系原点
        image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # 轴向矩阵（RAS坐标系）
        mha_name = mp4_file.replace('.mp4', '.mha')
        sitk.WriteImage(image, mha_name)
        os.remove(mp4_file)


def rename_mha(folder):
    data_folder = os.path.join(folder, 'data')
    mask_folder = os.path.join(folder, 'mask')
    data_files = [os.path.join(root, f) for root, _, files in os.walk(data_folder) for f in files if f.endswith(('.mha'))]
    for data_file in data_files:
        base_name = os.path.basename(data_file)
        mask_file = os.path.join(mask_folder, base_name)
        if os.path.exists(mask_file):
            random_name = str(uuid.uuid4()) + ".mha"
            output_data_name = os.path.join(data_folder, random_name)
            output_mask_name = os.path.join(mask_folder, random_name)
            os.rename(data_file, output_data_name)
            os.rename(mask_file, output_mask_name)
        else:
            os.remove(data_file)


def split_mha(folder):
    mask_folder = folder + '/mask'
    os.makedirs(mask_folder, exist_ok=True)
    mha_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(('_mask.mha'))]
    for mha_file in mha_files:
        base_name = os.path.basename(mha_file)
        base_name = base_name.replace('_mask.mha', '.mha')
        mha_mask = os.path.join(mask_folder, base_name)
        os.rename(mha_file, mha_mask)


# def align_mha(folder):
#     mask_folder = folder + '/mask'
#     os.makedirs(mask_folder, exist_ok=True)
#     mha_files = [os.path.join(root, f) for root, _, files in os.walk(folder) for f in files if f.endswith(('_mask.mha'))]
#     for mha_file in mha_files:
#         base_name = os.path.basename(mha_file)
#         base_name = base_name.replace('_mask.mha', '.mha')
#         mha_mask = os.path.join(mask_folder, base_name)
#         os.rename(mha_file, mha_mask)

def parse_xml_to_dataframe(xml_dir):
    """
    遍历XML目录，提取标注信息并生成DataFrame

    参数:
        xml_dir: 存放XML文件的目录路径

    返回:
        pandas.DataFrame: 包含seriesuid, xmin, ymin, xmax, ymax, label的表格
    """
    data = []
    xml_files = [os.path.join(root, f) for root, _, files in os.walk(xml_dir) for f in files if f.endswith('.xml')]
    # 遍历目录中的所有XML文件
    for xml_file in xml_files:
        try:
            print(xml_file)
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 提取seriesuid（去掉.xml后缀）
            seriesuid = os.path.splitext(os.path.basename(xml_file))[0]

            # 提取每个object的bndbox信息
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    # 添加到数据列表（label固定为128）
                    data.append({
                        'seriesuid': seriesuid,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'label': 128
                    })
        except:
            pass

    # 生成DataFrame
    df = pd.DataFrame(data)
    return df



if __name__ == '__main__':
    folder = 'Lung-PET-CT-Dx'
    data_folder = os.path.join(folder, 'data')
    data_files = [os.path.join(root, f) for root, _, files in os.walk(data_folder) for f in files if f.endswith(('.mha'))]
    for data_file in data_files:
        # base_name = os.path.basename(data_file)
        random_name = str(uuid.uuid4()) + ".mha"
        output_data_name = os.path.join(data_folder, random_name)
        os.rename(data_file, output_data_name)

