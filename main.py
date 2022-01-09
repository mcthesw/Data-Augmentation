import os
import argparse
import re
import random
from DataObj import ImageData

VAL_RATE = 1 / 8


# 通过命令行参数得到数据源路径
parser = argparse.ArgumentParser()
parser.add_argument("DataSource", type=str, help="数据源路径")
args = parser.parse_args()
DataSource = os.path.join(args.DataSource)
DataTarget = "Target\\"
files = os.listdir(DataSource)
print("读取文件成功")

# 获取所有图片
file: str
picFiles = [file for file in files if re.fullmatch(r".*\.(jpg|png)$", file, re.I)]
print("读取所有源图片成功：")
print("\n".join(picFiles))
if int(len(picFiles)*VAL_RATE) < 1:
    print("样本数量不足，或者VAL_RATE设置太小(该提示不会影响程序运行)")

# 创建目标输出文件夹
try:
    os.mkdir(DataTarget)
except FileExistsError:
    input("文件夹已存在，请删除Target文件夹，按任意键结束程序")
    exit()

data_file_list = []
for i in picFiles:
    print(f"正在转换文件:{i}")
    data_file_list.append(ImageData.create_from_file(i, DataSource))
for i in data_file_list:
    i.dump_masks_and_image(DataTarget)

print(f"转换全部成功，接下来进行随机选择VAL，你的VAL_RATE为{VAL_RATE}")
results = os.listdir(DataTarget)
VALs = random.sample(results, int(len(results) * VAL_RATE))
with open(os.path.join(DataTarget, "VALs.txt"), mode="w") as val_file:
    val_file.write(str([i for i in VALs]))
print(f"vals.txt 已经生成在了 {DataTarget} 下")
print("所有处理均已完成")
