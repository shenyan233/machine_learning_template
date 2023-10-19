# Includes some tools that are independent of the model
# 包含一些与模型无关的工具
import cProfile
import glob
import pstats

import pandas
import torch


def zip_dir(dir_path, result_path):
    """
    :param dir_path: Destination folder path|目标文件夹路径
    :param result_path: result path|压缩后的文件夹路径
    """
    import zipfile
    import os

    ziper = zipfile.ZipFile(result_path, "w", zipfile.ZIP_DEFLATED)
    for root, dirnames, filenames in os.walk(dir_path):
        # Delete the root path and compress only files and folders in the target folder
        # 去掉根路径，只对目标文件夹下的文件及文件夹进行压缩
        file_path = root.replace(dir_path, '')
        for filename in filenames:
            ziper.write(os.path.join(root, filename), os.path.join(file_path, filename))
    ziper.close()


def unzip_dir(zip_path, dir_path):
    """
    :param zip_path: Destination folder path|目标文件夹路径
    :param dir_path: result path|解压后的文件夹路径
    """
    import zipfile

    zip_obj = zipfile.ZipFile(zip_path, 'r')
    zip_obj.extractall(dir_path)
    zip_obj.close()


def ncolors(num_colors):
    """
    Generate several colors with larger differences
    生成区别度较大的几种颜色
    copy from: https://blog.csdn.net/choumin/article/details/90320297
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
    import os
    from torchvision import transforms
    from PIL import Image
    import torch

    label_path = os.path.join(dataset_path, 'test', 'labels').replace('\\', '/')
    label_image_list = glob.glob(label_path + '/*.png')
    label_image_list.sort()

    trans_factory = transforms.ToPILImage()
    if not os.path.exists(dataset_path + '/visual_label'):
        os.makedirs(dataset_path + '/visual_label')
    for index in range(len(label_image_list)):
        label_image = Image.open(label_image_list[index])
        name = os.path.basename(label_image_list[index])
        trans_factory(torch.from_numpy(label_image).float() / n_classes).save(
            dataset_path + '/visual_label/' + name,
            quality=95)


def get_ckpt_path(version_nth: int, log_name, version_idx=0, path='.'):
    checkpoints_path = f'{path}/logs/{log_name}/version_{version_nth}/checkpoints'
    ckpt_paths = glob.glob(checkpoints_path + '/*.ckpt')
    for ckpt_path in ckpt_paths:
        if f'epoch={version_idx}-step' in ckpt_path:
            return ckpt_path.replace('\\', '/')
    assert False

def ckpt2onnx(version_nth, model_name, log_name, input_size, config, save_path):
    """
    @param version_nth:
    @param log_name:
    @param input_size:
    @param config:
    @param save_path: example: f'./logs/default/version_{version_nth}/version_{version_nth}.onnx'
    @return:
    """
    import importlib
    import torch

    checkpoint_path = get_ckpt_path(version_nth, log_name)

    imported = importlib.import_module(f'network.{model_name}')
    training_module = imported.TrainModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        **{'config': config})

    input_sample = torch.randn(input_size)
    training_module.to_onnx(save_path, input_sample, opset_version=12, export_params=True)


def onnx2tf(version_nth, save_path):
    """
    @param version_nth:
    @param save_path: example: './logs/{log_name}/version_{version_nth}'
    """
    from onnx_tf.backend import prepare
    import onnx

    onnx_path = glob.glob(save_path + '/*.onnx')[0].replace('\\', '/')
    onnx_model = onnx.load(onnx_path)  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(save_path + f"/version_{version_nth}.tf")  # export the model


def tf2tflite(version_nth, save_path):
    """
    @param version_nth:
    @param save_path: example: './logs/default/version_{version_nth}'
    """
    import tensorflow as tf

    tf_path = save_path + f"/version_{version_nth}.tf"
    tflite_path = save_path + f"/version_{version_nth}.tflite"
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True

    tf_lite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tf_lite_model)


def test_onnx(input, model_path):
    """
    @param input:
    @param model_path: example: "./logs/default/version_0/version_0.onnx"
    """
    import onnxruntime as rt
    model = rt.InferenceSession(model_path)
    input_name = model.get_inputs()
    output_name = None
    # If an error ’Invalid Feed Input Name‘ occurs, the Input and output names in the Model can be changed from
    # the Input names observed by the Netron software
    # 如果出现Invalid Feed Input Name: 这一错误，可通过Netron软件观察输入名称，改变model内的输入输出的名称
    result = model.run(output_name, {input_name[0].name: input})
    return result[0]


def test_tflite(input, model_path):
    """
    :@param input: numpy
    :@param model_path: example: "./logs/default/version_0/version_0.tflite"
    """
    import tensorflow as tf
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    index = input_details[0]['index']
    interpreter.set_tensor(index, input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def feature_visualize(features, labels):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_tsne = TSNE().fit_transform(features)
    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE")
    plt.show()


def load_logs_data(log_name, item='Validation acc'):
    from tensorboard.backend.event_processing import event_accumulator
    import os
    import pandas

    list_versions = os.listdir(f'./logs/{log_name}')
    pd = pandas.DataFrame()
    for version in list_versions:
        if 'txt' not in version:
            for file in os.listdir(f'./logs/{log_name}/{version}'):
                if 'events.out' in file:
                    tensor_board = event_accumulator.EventAccumulator(
                        f'./logs/{log_name}/{version}/{file}')
                    tensor_board.Reload()
                    val_acc = tensor_board.scalars.Items(item)

                    list_data = []
                    for i in val_acc:
                        list_data.append(i.value)
                    pd_one = pandas.DataFrame([list_data])
                    pd = pandas.concat([pd, pd_one])
    pd.to_csv(f'{item}.csv', index=False, encoding="utf-8")


def change_csv_colume(augment_colume: list):
    # the func to extract special info from csv as a colume
    import json

    info = pandas.read_csv('./logs/lightning_logs/version_info.csv')
    for line in range(len(info)):
        json_config = json.loads(info.loc[line, 'config'].replace('\'', '\"')
                                 .replace('None', 'null')
                                 .replace('False', 'false')
                                 .replace('True', 'true'))
        for colume in augment_colume:
            if colume in json_config:
                info.loc[line, colume] = json_config[colume]
            else:
                info.loc[line, colume] = ''
    columns = info.columns.to_list().copy()
    columns.remove('config')
    columns.append('config')
    info = info.reindex(columns=columns)
    info.to_csv('./logs/lightning_logs/new_version_info.csv', index=False)
    print('end')


def myprofile(func: str):
    import pandas
    import os

    cProfile.run(func, 'profile.txt')
    stats = pstats.Stats('profile.txt')
    all = {'ncalls1': [],
           'ncalls2': [],
           'tottime1': [],
           'cumtime1': [],
           'ncalls3': [],
           'ncalls4': [],
           'tottime2': [],
           'cumtime2': [],
           'name': [], }
    for i in list(stats.stats.keys()):
        name = str(i)
        all['ncalls1'].append(stats.stats[i][0])
        all['ncalls2'].append(stats.stats[i][1])
        all['tottime1'].append(stats.stats[i][2])
        all['cumtime1'].append(stats.stats[i][3])
        if len(stats.stats[i][4]) != 0:
            all['ncalls3'].append(stats.stats[i][4][list(stats.stats[i][4].keys())[0]][0])
            all['ncalls4'].append(stats.stats[i][4][list(stats.stats[i][4].keys())[0]][1])
            all['tottime2'].append(stats.stats[i][4][list(stats.stats[i][4].keys())[0]][2])
            all['cumtime2'].append(stats.stats[i][4][list(stats.stats[i][4].keys())[0]][3])
        else:
            all['ncalls3'].append('')
            all['ncalls4'].append('')
            all['tottime2'].append('')
            all['cumtime2'].append('')
        all['name'].append(name)
    pandas.DataFrame(all).to_csv('profile.csv', index=False)
    os.remove('profile.txt')


def eval_prob_distribution():
    import matplotlib.pyplot as plt
    import numpy

    valid_prob = pandas.read_csv('valid_prob.csv', header=None)
    right_prob = pandas.read_csv('right_prob.csv', header=None)
    false_prob = pandas.read_csv('false_prob.csv', header=None)

    max = valid_prob.max()[0]
    min = valid_prob.min()[0]

    plt.figure()
    bins = 100
    valid_prob_hist = plt.hist(valid_prob, range=(min, max), bins=bins, label='valid_prob')
    false_prob_hist = plt.hist(false_prob, range=(min, max), bins=bins, label='false_prob')
    right_prob_hist = plt.hist(right_prob, range=(min, max), bins=bins, label='right_prob', rwidth=0.5)
    plt.legend()

    total_valid = numpy.zeros(bins)
    total_right = numpy.zeros(bins)
    for i in range(bins):
        total_valid[i] = valid_prob_hist[0][i:].sum()
        total_right[i] = right_prob_hist[0][i:].sum()

    plt.figure()
    plt.plot(right_prob_hist[0] / valid_prob_hist[0], label='right/valid hist')
    plt.plot(total_right / total_valid, label='right/valid')
    plt.plot([0.6666] * len(right_prob_hist[0]))
    plt.legend()

    plt.show()


def find_usable_cuda_devices(num: int):
    usable_cuda = []
    cuda_num = torch.cuda.device_count()
    for i in range(cuda_num):
        try:
            test = torch.tensor([1.0]).cuda(cuda_num)
            assert test[0] == 1
            usable_cuda.append(i)
            if len(usable_cuda) == num:
                return usable_cuda
        except:
            print(f'cuda: {i} is occupied')
    assert False, 'usable cuda devices is not enough'


if __name__ == "__main__":
    eval_prob_distribution()
    pass
