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


def norm(array, column_index):
    # Extract columns to normalize
    # 提取要归一化的列
    column = array[:, column_index]
    min_value = numpy.min(column)
    max_value = numpy.max(column)
    if (max_value - min_value) != 0:
        normalized_column = (column - min_value) / (max_value - min_value)
    else:
        normalized_column = column * 0
    array[:, column_index] = normalized_column
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
