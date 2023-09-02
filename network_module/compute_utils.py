import numpy
import torch


def one_hot_encoder(targets, n_classes):
    class_mask = targets.data.new(targets.size(0), n_classes).fill_(0)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    return class_mask


def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    # If num is 0, all numbers are nan. In this case, num is 1 because the denominator cannot be 0
    # num为0表示所有数均为nan, 此时由于分母不能为0, 则设num为1
    if num == 0:
        num = 1
    return value / num


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def norm(array: torch.Tensor, column_index):
    # Extract columns to normalize
    # 提取要归一化的列
    column = array[:, :, column_index]
    # Calculate the minimum and maximum values of a column
    # 计算列的最小值和最大值
    min_value = column.min(1)[0].min(1)[0]
    max_value = column.max(1)[0].max(1)[0]

    max_range = max_value - min_value
    # # When the minimum value is 0, the minimum scale is denoted as 0
    # # 当最小值为0时，最小尺度记为0
    # scale_min = min_value > 0
    # # When the minimum value is 0, the maximum scale is divided by 1,
    # # otherwise the percentage of increment is obtained
    # # 当最小值为0时，最大尺度除1，否则得到增量的百分比
    # scale_max = max_range / (min_value + (min_value == 0))

    # Normalized column data, when the change is constant 0, the denominator is 1, and after normalization,
    # it is constant 0
    # 归一化列数据，当变化量恒为0时，分母为1，归一化后恒为0
    normalized_column = (column - min_value.unsqueeze(dim=1).unsqueeze(dim=1)) / (
            max_range + (max_range == 0)).unsqueeze(dim=1).unsqueeze(dim=1)
    array[:, :, column_index] = normalized_column

    # array = torch.concat([array,
    #                       (torch.ones((array.shape[0], array.shape[1], 1)).type_as(scale_max) * scale_max.unsqueeze(
    #                           dim=1).unsqueeze(
    #                           dim=1)),
    #                       (torch.ones((array.shape[0], array.shape[1], 1)).type_as(scale_min) * scale_min.unsqueeze(
    #                           dim=1).unsqueeze(
    #                           dim=1))], dim=2)
    return array


def concat_zeros(x: torch.Tensor, dim, index, num=1):
    num_dim = len(x.shape)
    code_left = [':'] * num_dim
    code_left[dim] = ':index'
    code_left = ','.join(code_left)
    code_right = [':'] * num_dim
    code_right[dim] = 'index:'
    code_right = ','.join(code_right)
    left = eval(f'x[{code_left}]')
    right = eval(f'x[{code_right}]')
    zeros_shape = list(x.shape)
    zeros_shape[dim] = num
    zeros = torch.zeros(tuple(zeros_shape))
    return torch.concat([left, zeros, right], dim=dim)
