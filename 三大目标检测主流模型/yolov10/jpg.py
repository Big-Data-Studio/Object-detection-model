import os

# 指定你的文件夹路径
folder_path = 'C:/Users/86159/Desktop/png问题集/'
i = 2000
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 构建新的文件名
    new_filename = '{}'.format(i)+".jpg"
    i += 1
    # 获取文件的完整路径
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)
    # 重命名文件
    os.rename(old_file, new_file)