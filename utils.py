# 包含一些与网络无关的工具
import glob
import os
import random
import zipfile
import cv2
import torch


def divide_dataset(dataset_path, rate_datasets):
    """
    切分数据集, 划分为训练集,验证集,测试集生成list文件并保存为:
    train_dataset_list、validate_dataset_list、test_dataset_list.
    每个比例必须大于0且保证至少每个数据集中具有一个样本, 验证集可以为0.

    :param dataset_path: 数据集的地址
    :param rate_datasets: 不同数据集[训练集,验证集,测试集]的比例
    """
    # 当不存在总的all_dataset_list文件时, 生成all_dataset_list
    if not os.path.exists(dataset_path + '/all_dataset_list.txt'):
        all_list = glob.glob(dataset_path + '/labels' + '/*.png')
        with open(dataset_path + '/all_dataset_list.txt', 'w', encoding='utf-8') as f:
            for line in all_list:
                f.write(os.path.basename(line.replace('\\', '/')) + '\n')
    path_train_dataset_list = dataset_path + '/train_dataset_list.txt'
    path_validate_dataset_list = dataset_path + '/validate_dataset_list.txt'
    path_test_dataset_list = dataset_path + '/test_dataset_list.txt'
    # 如果验证集的比例为0，则将测试集设置为验证集并取消测试集;
    if rate_datasets[1] == 0:
        # 如果无切分后的list文件, 则生成新的list文件
        if not (os.path.exists(path_train_dataset_list) and
                os.path.exists(path_validate_dataset_list) and
                os.path.exists(path_test_dataset_list)):
            all_list = open(dataset_path + '/all_dataset_list.txt').readlines()
            random.shuffle(all_list)
            train_dataset_list = all_list[0:int(len(all_list) * rate_datasets[0])]
            test_dataset_list = all_list[int(len(all_list) * rate_datasets[0]):]
            with open(path_train_dataset_list, 'w', encoding='utf-8') as f:
                for line in train_dataset_list:
                    f.write(line)
            with open(path_validate_dataset_list, 'w', encoding='utf-8') as f:
                for line in test_dataset_list:
                    f.write(line)
            with open(path_test_dataset_list, 'w', encoding='utf-8') as f:
                for line in test_dataset_list:
                    f.write(line)
            print('已生成新的数据list')
        else:
            # 判断比例是否正确,如果不正确,则重新生成数据集
            all_list = open(dataset_path + '/all_dataset_list.txt').readlines()
            with open(path_train_dataset_list) as f:
                train_dataset_list_exist = f.readlines()
            with open(path_validate_dataset_list) as f:
                test_dataset_list_exist = f.readlines()
            random.shuffle(all_list)
            train_dataset_list = all_list[0:int(len(all_list) * rate_datasets[0])]
            test_dataset_list = all_list[int(len(all_list) * rate_datasets[0]):]
            if not (len(train_dataset_list_exist) == len(train_dataset_list) and
                    len(test_dataset_list_exist) == len(test_dataset_list)):
                with open(path_train_dataset_list, 'w', encoding='utf-8') as f:
                    for line in train_dataset_list:
                        f.write(line)
                with open(path_validate_dataset_list, 'w', encoding='utf-8') as f:
                    for line in test_dataset_list:
                        f.write(line)
                with open(path_test_dataset_list, 'w', encoding='utf-8') as f:
                    for line in test_dataset_list:
                        f.write(line)
                print('已生成新的数据list')
    # 如果验证集比例不为零,则同时存在验证集和测试集
    else:
        # 如果无切分后的list文件, 则生成新的list文件
        if not (os.path.exists(dataset_path + '/train_dataset_list.txt') and
                os.path.exists(dataset_path + '/validate_dataset_list.txt') and
                os.path.exists(dataset_path + '/test_dataset_list.txt')):
            all_list = open(dataset_path + '/all_dataset_list.txt').readlines()
            random.shuffle(all_list)
            train_dataset_list = all_list[0:int(len(all_list) * rate_datasets[0])]
            validate_dataset_list = all_list[int(len(all_list) * rate_datasets[0]):
                                             int(len(all_list) * (rate_datasets[0] + rate_datasets[1]))]
            test_dataset_list = all_list[int(len(all_list) * (rate_datasets[0] + rate_datasets[1])):]
            with open(path_train_dataset_list, 'w', encoding='utf-8') as f:
                for line in train_dataset_list:
                    f.write(line)
            with open(path_validate_dataset_list, 'w', encoding='utf-8') as f:
                for line in validate_dataset_list:
                    f.write(line)
            with open(path_test_dataset_list, 'w', encoding='utf-8') as f:
                for line in test_dataset_list:
                    f.write(line)
            print('已生成新的数据list')
        else:
            # 判断比例是否正确,如果不正确,则重新生成数据集
            all_list = open(dataset_path + '/all_dataset_list.txt').readlines()
            with open(path_train_dataset_list) as f:
                train_dataset_list_exist = f.readlines()
            with open(path_validate_dataset_list) as f:
                validate_dataset_list_exist = f.readlines()
            with open(path_test_dataset_list) as f:
                test_dataset_list_exist = f.readlines()
            random.shuffle(all_list)
            train_dataset_list = all_list[0:int(len(all_list) * rate_datasets[0])]
            validate_dataset_list = all_list[int(len(all_list) * rate_datasets[0]):
                                             int(len(all_list) * (rate_datasets[0] + rate_datasets[1]))]
            test_dataset_list = all_list[int(len(all_list) * (rate_datasets[0] + rate_datasets[1])):]
            if not (len(train_dataset_list_exist) == len(train_dataset_list) and
                    len(validate_dataset_list_exist) == len(validate_dataset_list) and
                    len(test_dataset_list_exist) == len(test_dataset_list)):
                with open(path_train_dataset_list, 'w', encoding='utf-8') as f:
                    for line in train_dataset_list:
                        f.write(line)
                with open(path_validate_dataset_list, 'w', encoding='utf-8') as f:
                    for line in validate_dataset_list:
                        f.write(line)
                with open(path_test_dataset_list, 'w', encoding='utf-8') as f:
                    for line in test_dataset_list:
                        f.write(line)
                print('已生成新的数据list')


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
