import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage.draw import disk
import os
import uuid
import csv
import re
from io import StringIO
import xml.etree.ElementTree as ET
from collections import defaultdict
from scipy.ndimage import distance_transform_edt


def create_nodule_mask(row):
    # """为单个结节生成3D掩码"""
    # # 初始化空白掩码
    # mask = np.zeros((row['slices'], row['rows'], row['cols']), dtype=np.uint8)
    #
    # # 计算结节半径（mm转像素）
    # radius_mm = float(row['Size_CT_Report'].replace('mm', '')) / 2 + 3
    # radius_pixels = int(round(radius_mm / row['spacing_x']))
    #
    # # 获取结节中心坐标
    # z_center = int(row['Slice_nodule_position']) - 1  # DICOM坐标系
    # y_center, x_center = row['pixel_y'], row['pixel_x']
    #
    # # 在每层切片上绘制圆形ROI
    # for z in range(max(0, z_center - 2), min(mask.shape[0], z_center + 3)):
    #     # 计算当前层与中心层的距离权重
    #     distance = abs(z - z_center) * row['spacing_z']
    #     if distance <= radius_mm:
    #         effective_radius = int(round(np.sqrt(radius_mm ** 2 - distance ** 2) / row['spacing_x']))
    #         rr, cc = disk((y_center, x_center), effective_radius, shape=mask.shape[1:])
    #         mask[z, rr, cc] = 1 if row['Pathological_results'] == 'malignant' else 2
    # return mask
    """为单个结节生成3D掩码"""
    # 初始化空白掩码 (slices, rows, cols)
    mask = np.zeros((row['slices'], row['rows'], row['cols']), dtype=np.uint8)

    # 计算结节半径（使用GPT报告的大小，单位mm）
    try:
        radius_mm = float(row['Size_GPT_Report']) / 2
    except:
        radius_mm = 5 / 2  # 默认5mm直径

    # 转换为像素单位（使用x方向间距）
    radius_pixels = int(round(radius_mm / row['spacing_x']))

    # 获取结节中心坐标
    z_center = int(row['Slice_nodule_position']) - 1  # 转换为0-based索引
    y_center, x_center = row['pixel_y'], row['pixel_x']

    # 在3D空间创建球形ROI（简化版，实际应用中可能需要更精确的方法）
    for z in range(max(0, z_center - radius_pixels),
                   min(mask.shape[0], z_center + radius_pixels + 1)):

        # 计算当前层与中心层的距离权重
        z_distance = abs(z - z_center) * row['spacing_z']

        # 计算当前层的有效半径（球形投影）
        if z_distance <= radius_mm:
            effective_radius = int(round(
                np.sqrt(max(0, radius_mm ** 2 - z_distance ** 2)) / row['spacing_x']
            ))

            # 在当前切片绘制圆形
            rr, cc = disk((y_center, x_center),
                          effective_radius,
                          shape=mask.shape[1:])
            mask[z, rr, cc] = 1 if row['Diagnosis'] == 'Malignant' else 2

    return mask


def xlsx_to_mask(file_name, folder):
    # 读取Excel数据
    df = pd.read_excel(file_name)

    # 提取关键字段并清洗
    nodules = df[['Patient ID', 'Video_name', 'Voxel', 'Slice_thickness',
                 'Slice_nodule_position', 'Pixel_nodule_position',
                 'Size_CT_Report', 'Pathological_results']].copy()

    # 解析体素和间距数据
    nodules[['rows', 'cols', 'slices']] = nodules['Voxel'].str.extract(r'(\d+)\*(\d+)\*(\d+)').astype(int)
    nodules[['pixel_x', 'pixel_y']] = nodules['Pixel_nodule_position'].str.extract(r'\((\d+),(\d+)\)').astype(int)
    nodules[['spacing_x', 'spacing_y', 'spacing_z']] = (
        nodules['Slice_thickness'].str.extract(r'([\d.]+)\*([\d.]+)\*([\d.]+)mm').astype(float)
    )
    # print(nodules)
    for idx, row in nodules.iterrows():
        try:
            print(row)
            mask = create_nodule_mask(row)
            mha_name = row['Video_name'].replace('.mp4', '.mha')
            random_name = str(uuid.uuid4()) + ".mha"
            reference_dicom = os.path.join(folder, mha_name)
            reference_dicom_random_name = os.path.join(folder, random_name)
            os.rename(reference_dicom, reference_dicom_random_name)
            reference_dicom = reference_dicom_random_name
            sitk_image = sitk.ReadImage(reference_dicom)

            # 创建NIfTI图像
            mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_image.CopyInformation(sitk_image)

            # 设置标签说明
            mask_image.SetMetaData("Label_1", "Malignant")
            mask_image.SetMetaData("Label_2", "Benign")

            mask_mha_name = reference_dicom.replace('.mha', '_mask.mha')
            # 保存文件
            sitk.WriteImage(mask_image, mask_mha_name)
        except:
            pass


def read_and_parse_csv(csv_content, folder):
    # 读取CSV数据
    df = pd.read_csv(csv_content)
    print(df)
    # 提取关键字段并清洗
    nodules = df[['pid', 'Follow_time', 'Voxel', 'Slice_thickness',
                  'Slice_nodule_position', 'Pixel_nodule_position',
                  'Size_CT_Report', 'Size_GPT_Report',
                  'Malignancy_GPT_report',
                  '100 patients randomly selected based on the positive rate reported by NLST']].copy()

    # 重命名列以保持一致性
    nodules = nodules.rename(columns={
        'pid': 'Patient_ID',
        'Follow_time': 'Scan_date',
        '100 patients randomly selected based on the positive rate reported by NLST': 'Pathology_status'
    })

    # 解析体素和间距数据
    nodules[['rows', 'cols', 'slices']] = (
        nodules['Voxel'].str.extract(r'(\d+)\*(\d+)\*(\d+)').astype(int)
    )

    # 解析像素坐标（处理中英文括号情况）
    pixel_coords = nodules['Pixel_nodule_position'].str.extract(
        r'[\(（](\d+)[,\s]*(\d+)[\)）]'
    )
    nodules[['pixel_x', 'pixel_y']] = pixel_coords.astype(int)

    # 解析切片间距（处理不同分隔符情况）
    spacing_data = nodules['Slice_thickness'].str.extract(
        r'([\d.]+)\*([\d.]+)\*([\d.]+)'
    )
    nodules[['spacing_x', 'spacing_y', 'spacing_z']] = spacing_data.astype(float)

    # 添加诊断结果分类（恶性/良性）
    nodules['Diagnosis'] = nodules['Pathology_status'].map({
        'malignant': 'Malignant',
        'benign': 'Benign'
    })

    # 计算结节体积（简化计算，假设为球形）
    nodules['Volume'] = (4 / 3) * 3.1416 * (nodules['Size_GPT_Report'] / 2) ** 3
    print(nodules)
    # for idx, row in nodules.iterrows():
    #     try:
    #         print(row)
    #         mask = create_nodule_mask(row)
    #         mha_name = row['Patient_ID'] + '.mha'
    #         random_name = str(uuid.uuid4()) + ".mha"
    #         reference_dicom = os.path.join(folder, mha_name)
    #         reference_dicom_random_name = os.path.join(folder, random_name)
    #         os.rename(reference_dicom, reference_dicom_random_name)
    #         reference_dicom = reference_dicom_random_name
    #         sitk_image = sitk.ReadImage(reference_dicom)
    #
    #         # 创建NIfTI图像
    #         mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
    #         mask_image.CopyInformation(sitk_image)
    #
    #         # 设置标签说明
    #         mask_image.SetMetaData("Label_1", "Malignant")
    #         mask_image.SetMetaData("Label_2", "Benign")
    #
    #         mask_mha_name = reference_dicom.replace('.mha', '_mask.mha')
    #         # 保存文件
    #         sitk.WriteImage(mask_image, mask_mha_name)
    #     except:
    #         pass


def create_sphere_mask(shape, center, radius, voxel_spacing):
    """创建一个球形mask"""
    # 创建一个网格
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # 计算每个点到中心的距离
    dist_from_center = np.sqrt(((x - center[0]) * voxel_spacing[0]) ** 2 +
                               ((y - center[1]) * voxel_spacing[1]) ** 2 +
                               ((z - center[2]) * voxel_spacing[2]) ** 2)

    # 创建mask
    mask = dist_from_center <= radius
    return mask


def csv_to_mha_mask(csv_path, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(df)
    #
    # # 读取参考图像获取空间信息
    # reference_image = sitk.ReadImage(reference_image_path)
    # size = reference_image.GetSize()
    # spacing = reference_image.GetSpacing()
    # origin = reference_image.GetOrigin()
    # direction = reference_image.GetDirection()
    #
    # # 创建一个空的numpy数组用于存储mask
    # mask_array = np.zeros(size[::-1], dtype=np.uint8)  # 注意SimpleITK和numpy的维度顺序是相反的

    # # 处理每个标注点
    # for _, row in df.iterrows():
    #     # 获取坐标和直径
    #     coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
    #     diameter = row['diameter_mm']
    #     radius = diameter / 2.0
    #
    #     # 将世界坐标转换为图像坐标
    #     point = reference_image.TransformPhysicalPointToIndex(coord)
    #
    #     # 确保点在图像范围内
    #     if all(0 <= p < s for p, s in zip(point, size)):
    #         # 创建球形mask
    #         sphere_mask = create_sphere_mask(size[::-1], point[::-1], radius, spacing)
    #
    #         # 将球形mask添加到总mask中
    #         mask_array[sphere_mask] = gray_value
    #
    # # 将numpy数组转换为SimpleITK图像
    # mask_image = sitk.GetImageFromArray(mask_array)
    # mask_image.CopyInformation(reference_image)
    #
    # # 保存mask为MHA文件
    # sitk.WriteImage(mask_image, output_path)

# def mark_spherical_roi(mask, center, radius, value):
#     """在 3D 掩码中标记一个球形区域"""
#     z, y, x = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
#     distance = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
#     mask[distance <= radius] = value
#
#
# def csv_to_masks_with_reference(csv_path, reference_image_dir, output_dir):
#     """
#     将 CSV 标注转换为 3D 掩码（需原始图像作为参考）
#
#     参数:
#         csv_path: mask.csv 文件路径
#         reference_image_dir: 存放原始 .mha 图像的目录（文件名格式为 {seriesuid}.mha）
#         output_dir: 输出掩码的目录
#     """
#     # 读取 CSV 文件
#     df = pd.read_csv(csv_path)
#
#     # 按 seriesuid 分组处理
#     grouped = df.groupby('seriesuid')
#
#     for seriesuid, group in grouped:
#         print(f"Processing {seriesuid}...")
#         print(len(group))
#         # 检查原始图像是否存在
#         ref_image_path = os.path.join(reference_image_dir, f"{seriesuid}.mha")
#         if not os.path.exists(ref_image_path):
#             print(f"Warning: Reference image not found for {seriesuid}. Skipping.")
#             continue
#
#         # 读取原始图像
#         ref_image = sitk.ReadImage(ref_image_path)
#         ref_array = sitk.GetArrayFromImage(ref_image)  # 获取 NumPy 数组 (Z, Y, X)
#
#         # 初始化掩码（与原始图像同尺寸）
#         mask = np.zeros_like(ref_array, dtype=np.uint8)
#
#         # 获取原始图像的几何信息
#         origin = ref_image.GetOrigin()  # 物理原点 (X, Y, Z)
#         spacing = ref_image.GetSpacing()  # 体素间距 (mm)
#
#         # 处理当前 seriesuid 的所有标注
#         for _, row in group.iterrows():
#             # 物理坐标 → 体素坐标（注意顺序：X, Y, Z → 列, 行, 切片）
#             x, y, z = row['coordX'], row['coordY'], row['coordZ']
#             voxel_x = int((x - origin[0]) / spacing[0])
#             voxel_y = int((y - origin[1]) / spacing[1])
#             voxel_z = int((z - origin[2]) / spacing[2])
#
#             # 计算球形半径（体素单位）
#             radius_mm = row['diameter_mm'] / 2
#             radius_voxel = int(radius_mm / spacing[0])  # 假设各向同性间距
#
#             # 标记球形区域（跳过超出图像边界的标注）
#             if (0 <= voxel_x < ref_array.shape[2] and
#                     0 <= voxel_y < ref_array.shape[1] and
#                     0 <= voxel_z < ref_array.shape[0]):
#                 mark_spherical_roi(mask, (voxel_z, voxel_y, voxel_x), radius_voxel, row['class'])
#             else:
#                 print(f"Warning: Coordinate ({x}, {y}, {z}) is outside the image for {seriesuid}.")
#
#         # 将 NumPy 数组转回 SimpleITK 图像
#         mask_image = sitk.GetImageFromArray(mask)
#         mask_image.CopyInformation(ref_image)  # 复制几何信息（原点、间距、方向）
#
#         # 保存掩码
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f"{seriesuid}_mask.mha")
#         sitk.WriteImage(mask_image, output_path)
#         print(f"Saved mask to {output_path}")


def mark_ellipsoid_roi(mask, center, radii, value):
    """
    在 3D 掩码中标记一个椭球体区域

    参数:
        mask: 3D NumPy 数组 (Z, Y, X)
        center: 椭球中心坐标 (z, y, x)
        radii: 椭球三个方向的半径 (rz, ry, rx)
        value: 填充的标签值
    """
    z, y, x = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    # 椭球方程: ((z - z0)/rz)^2 + ((y - y0)/ry)^2 + ((x - x0)/rx)^2 <= 1
    distance = np.sqrt(
        ((z - center[0]) / radii[0]) ** 2 +
        ((y - center[1]) / radii[1]) ** 2 +
        ((x - center[2]) / radii[2]) ** 2
    )
    mask[distance <= 1] = value


def csv_to_masks_with_reference(csv_path, reference_image_dir, output_dir):
    """
    将 CSV 标注转换为 3D 掩码，按 seriesuid 生成椭球体形状的立体掩码

    参数:
        csv_path: CSV 文件路径（需包含 coordX/Y/Z 和 diameterX/Y/Z）
        reference_image_dir: 参考 .mha 图像的目录（文件名格式为 {seriesuid}.mha）
        output_dir: 输出掩码的目录
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 按 seriesuid 分组处理
    grouped = df.groupby('seriesuid')

    for seriesuid, group in grouped:
        print(f"Processing {seriesuid}...")

        # 检查参考图像是否存在
        ref_image_path = os.path.join(reference_image_dir, f"{seriesuid}.mha")
        if not os.path.exists(ref_image_path):
            print(f"Warning: Reference image not found for {seriesuid}. Skipping.")
            continue

        # 读取参考图像
        ref_image = sitk.ReadImage(ref_image_path)
        ref_array = sitk.GetArrayFromImage(ref_image)  # 获取 NumPy 数组 (Z, Y, X)

        # 初始化掩码（与参考图像同尺寸）
        mask = np.zeros_like(ref_array, dtype=np.uint8)

        # 获取参考图像的几何信息
        origin = ref_image.GetOrigin()  # 物理原点 (X, Y, Z)
        spacing = ref_image.GetSpacing()  # 体素间距 (mm)

        # 处理当前 seriesuid 的所有标注
        for _, row in group.iterrows():
            # 物理坐标 → 体素坐标（注意顺序：X, Y, Z → 列, 行, 切片）
            x, y, z = row['coordX'], row['coordY'], row['coordZ']
            voxel_x = int((x - origin[0]) / spacing[0])
            voxel_y = int((y - origin[1]) / spacing[1])
            voxel_z = int((z - origin[2]) / spacing[2])

            # 计算椭球三个方向的半径（体素单位）
            radius_x = row['diameterX'] / (2 * spacing[0])
            radius_y = row['diameterY'] / (2 * spacing[1])
            radius_z = row['diameterZ'] / (2 * spacing[2])
            radii = (radius_z, radius_y, radius_x)  # 注意顺序：Z, Y, X

            # # 替换标签值：5→128, 32→255
            label = row['label']
            # if label == 5:
            #     label = 128
            # elif label == 32:
            #     label = 255

            # 标记椭球区域（跳过超出图像边界的标注）
            if (0 <= voxel_x < ref_array.shape[2] and
                    0 <= voxel_y < ref_array.shape[1] and
                    0 <= voxel_z < ref_array.shape[0]):
                mark_ellipsoid_roi(
                    mask,
                    center=(voxel_z, voxel_y, voxel_x),
                    radii=radii,
                    value=label
                )
            else:
                print(f"Warning: Coordinate ({x}, {y}, {z}) is outside the image.")

        # 将 NumPy 数组转回 SimpleITK 图像并复制几何信息
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.CopyInformation(ref_image)

        # 保存掩码
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{seriesuid}_mask.mha")
        sitk.WriteImage(mask_image, output_path)
        print(f"Saved mask to {output_path}")


if __name__ == '__main__':
    csv_name = 'multi/mask.csv'
    reference_folder = 'multi/data'
    out_folder = 'multi/mask'
    csv_to_masks_with_reference(csv_name, reference_folder, out_folder)