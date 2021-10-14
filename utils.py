# 包含一些与网络无关的工具
import glob
import os
import random
import string
import zipfile
import cv2
import numpy
import torch


def get_dataset_list(dataset_path):
    if not os.path.exists(dataset_path + '/dataset_list.txt'):
        all_list = glob.glob(dataset_path + '/labels' + '/*.png')
        random.shuffle(all_list)
        with open(dataset_path + '/dataset_list.txt', 'w', encoding='utf-8') as f:
            for line in all_list:
                f.write(os.path.basename(line.replace('\\', '/')) + '\n')
            print('已生成新的数据list')
        return all_list
    else:
        all_list = open(dataset_path + '/all_dataset_list.txt').readlines()
        return all_list


def zip_dir(dir_path, zip_path):
    """
    压缩文件

    :param dir_path: 目标文件夹路径
    :param zip_path: 压缩后的文件夹路径
    """
    ziper = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    for root, dirnames, filenames in os.walk(dir_path):
        file_path = root.replace(dir_path, '')  # 去掉根路径，只对目标文件夹下的文件及文件夹进行压缩
        # 循环出一个个文件名
        for filename in filenames:
            ziper.write(os.path.join(root, filename), os.path.join(file_path, filename))
    ziper.close()


def ncolors(num_colors):
    """
    生成区别度较大的几种颜色
    copy: https://blog.csdn.net/choumin/article/details/90320297

    :param num_colors: 颜色数
    :return:
    """

    def get_n_hls_colors(num):
        import random
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            li = 50 + random.random() * 10
            _hlsc = [h / 360.0, li / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors

    import colorsys
    rgb_colors = []
    if num_colors < 1:
        return rgb_colors
    for hlsc in get_n_hls_colors(num_colors):
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def visual_label(dataset_path, n_classes):
    """
    将标签可视化

    :param dataset_path: 地址
    :param n_classes: 类别数
    """
    label_path = os.path.join(dataset_path, 'test', 'labels').replace('\\', '/')
    label_image_list = glob.glob(label_path + '/*.png')
    label_image_list.sort()
    from torchvision import transforms
    trans_factory = transforms.ToPILImage()
    if not os.path.exists(dataset_path + '/visual_label'):
        os.mkdir(dataset_path + '/visual_label')
    for index in range(len(label_image_list)):
        label_image = cv2.imread(label_image_list[index], -1)
        name = os.path.basename(label_image_list[index])
        trans_factory(torch.from_numpy(label_image).float() / n_classes).save(
            dataset_path + '/visual_label/' + name,
            quality=95)


def get_ckpt_path(version_nth: int, kth_fold: int):
    if version_nth is None:
        return None
    else:
        version_name = f'version_{version_nth + kth_fold}'
        checkpoints_path = './logs/default/' + version_name + '/checkpoints'
        ckpt_path = glob.glob(checkpoints_path + '/*.ckpt')
        return ckpt_path[0].replace('\\', '/')


def rwxl():
    # 写
    # dataset_xl = xl.Workbook(write_only=True)
    # dataset_sh = dataset_xl.create_sheet('dataset', 0)
    # for row in range(self.x.shape[0]):
    #     for col in range(self.x.shape[1]):
    #         dataset_sh.cell(row + 1, col + 1).value = float(self.x[row, col])
    #     dataset_sh.cell(row + 1, self.x.shape[1] + 1).value = float(self.y[row])
    # dataset_xl.save(dataset_path + '/dataset.xlsx')
    # dataset_xl.close()
    # 读
    # dataset_xl = xl.load_workbook(dataset_path + '/dataset_list.xlsx', read_only=True)
    # dataset_sh = dataset_xl.get_sheet_by_name('dataset_list')
    # temp = [[dataset_sh[row + 1][col].value for col in range(config['dim_in'] + 1)] for row in
    #         range(config['dataset_len'])]
    # dataset_xl.close()
    pass


if __name__ == "__main__":
    get_ckpt_path('version_0')
