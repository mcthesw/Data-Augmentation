import gc
import os
import random
import re
import time
from typing import List

from DataAug import aug_data
from DataObj import ImageData, Patch

# 配置部分
# 注意：此处输入高和长的格式应为(高度, 长度)
MODE = "AUG"  # TODO:根据该项来输出 mode可为AUG,CreatePatch
VAL_RATE = 1 / 10  # 随机产生的VAL列表应当占总文件的比例
AUG = False  # 是否进行数据增强
SPLIT = False  # (384, 512)  # 将图片分割的大小，如果填写0或False则不进行分割

# PATCH 功能配置
PATCH = False  # 是否进行贴图
PATCH_SIZE = (128, 128)  # Patch的长宽
PATCH_AMOUNT = 2  # 一张图上有几个Patch
PATCH_PATH = "Patches\\"

# 基本数据源配置
DataSource = "DataSource\\"  # 数据源
DataTarget = "Target\\"  # 输出路径
assert MODE in ["AUG", "CreatePatch"]
# 配置部分结束

files = os.listdir(DataSource)
print("读取文件成功")

# 获取所有图片
file: str
picFiles = \
    [file for file in files if re.fullmatch(r".*\.(jpg|png)$", file, re.I)]
print("读取所有源图片成功：")
print("\n".join(picFiles))

# 从文件获取所有图像和mask
data_file_list = []  # 需要导出的文件列表
if MODE == "AUG":
    # 创建目标输出文件夹
    try:
        os.mkdir(DataTarget)
    except FileExistsError:
        input("文件夹已存在，请删除Target文件夹，按任意键结束程序")
        exit()
    for i in picFiles:
        print(f"\n\n开始处理图片: {i}")
        cur_data = ImageData.create_from_file(i, DataSource)
        cur_data_list = [cur_data, ]

        if AUG:
            #  print("开始进行图像处理数据增强")
            AUG_list = []
            for j in cur_data_list:
                AUG_list += aug_data(j)
            cur_data_list = AUG_list

        if SPLIT:
            #  print("开始进行图像分割数据增强")
            #  print("注意：如果使用了SPLIT的话只会输出被分割后的图片")
            AUG_list = []
            for j in cur_data_list:
                AUG_list += j.split(SPLIT)
            cur_data_list = AUG_list

        if PATCH:
            print("开始进行贴图数据增强")
            AUG_list = []
            patches = []
            # 初始化patches
            patches = Patch.load_from_folder(PATCH_PATH)
            if len(patches) < PATCH_AMOUNT:
                print("有效Patch数量小于PATCH_AMOUNT，无法执行该项数据增强")
                break
            # 把Patch贴到每一张图上
            # TODO：提供更高可自定义程度的贴图
            # 如果新的贴图产生的ImageData中没有新增某一类型的细胞核，那么会导致重复，解决方法：只使用patched的图片
            for data_file in cur_data_list:
                cur_patches = random.sample(patches, PATCH_AMOUNT)
                for i in cur_patches:
                    data_file = i.apply_to_image_data(data_file)
                AUG_list.append(data_file)
            cur_data_list = AUG_list

        data_file_list += [j.name for j in cur_data_list]
        for j in cur_data_list:
            print(f"正在导出文件:\n{str(j)}")
            j.dump_masks_and_image(DataTarget)
        if len(data_file_list) % 10 == 0:
            gc.collect()

    if int(len(data_file_list) * VAL_RATE) < 1:
        print("样本数量不足，或者VAL_RATE设置太小(该提示不会影响程序运行)")

    print(f"转换全部成功，接下来进行随机选择VAL，你的VAL_RATE为{VAL_RATE}")
    results = os.listdir(DataTarget)
    VALs = random.sample(results, int(len(results) * VAL_RATE))
    with open(os.path.join(DataTarget, "VALs.txt"), mode="w") as val_file:
        val_file.write(str([i for i in VALs]))
    print(f"VALs.txt 已经生成在了 {DataTarget} 下")
    print("所有处理均已完成")


elif MODE == "CreatePatch":
    try:
        os.mkdir(PATCH_PATH)
        print(f"已创建PATCH文件夹在:\n{PATCH_PATH}")
    except FileExistsError:
        print("PATCH文件夹已存在，直接向内追加")

    time = time.strftime("%Y-%m-%d", time.localtime())
    cnt = 0
    for i in picFiles:
        print(f"开始以该图片生成Patch: {i}")
        # noinspection PyTypeChecker
        img: ImageData = ImageData.create_from_file(i, DataSource)
        cur_patches: List[Patch] = \
            Patch.create_from_image_data(img, patch_size=PATCH_SIZE)
        for j in cur_patches:
            cnt += 1
            j.save_to_file(os.path.join(PATCH_PATH, time + f"_{cnt}.patch"))
