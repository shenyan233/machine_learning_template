import numpy
import torch


def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.long()


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
    normalized_column = (column - min_value) / (max_value - min_value)
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