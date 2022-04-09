# 包含一些与网络无关的工具
import glob


def zip_dir(dir_path, zip_path):
    """
    压缩文件

    :param dir_path: 目标文件夹路径
    :param zip_path: 压缩后的文件夹路径
    """
    import zipfile
    import os

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
    import os
    from torchvision import transforms
    import cv2
    import torch

    label_path = os.path.join(dataset_path, 'test', 'labels').replace('\\', '/')
    label_image_list = glob.glob(label_path + '/*.png')
    label_image_list.sort()

    trans_factory = transforms.ToPILImage()
    if not os.path.exists(dataset_path + '/visual_label'):
        os.makedirs(dataset_path + '/visual_label')
    for index in range(len(label_image_list)):
        label_image = cv2.imread(label_image_list[index], -1)
        name = os.path.basename(label_image_list[index])
        trans_factory(torch.from_numpy(label_image).float() / n_classes).save(
            dataset_path + '/visual_label/' + name,
            quality=95)


def get_ckpt_path(version_nth: int):
    version_name = f'version_{version_nth}'
    checkpoints_path = './logs/default/' + version_name + '/checkpoints'
    ckpt_path = glob.glob(checkpoints_path + '/*.ckpt')
    return ckpt_path[0].replace('\\', '/')


def fill_list(list, n):
    return list[:n] + ['default'] * (n - len(list))


def ckpt2onnx(version_nth, input_size, config):
    from train_model import TrainModule
    import torch

    checkpoint_path = get_ckpt_path(version_nth)
    # 构建网络
    training_module = TrainModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        **{'config': config})
    # 输入参数
    input_sample = torch.randn(input_size)
    training_module.to_onnx(f'./logs/default/version_{version_nth}/version_{version_nth}.onnx', input_sample,
                            opset_version=12, export_params=True)


def onnx2tf(version_nth):
    from onnx_tf.backend import prepare
    import onnx

    save_path = f'./logs/default/version_{version_nth}'
    onnx_path = glob.glob(save_path + '/*.onnx')[0].replace('\\', '/')
    onnx_model = onnx.load(onnx_path)  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(save_path + f"/version_{version_nth}.tf")  # export the model


def tf2tflite(version_nth):
    import tensorflow as tf

    save_path = f'./logs/default/version_{version_nth}'
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


def test_onnx(input):
    import onnxruntime as rt
    model = rt.InferenceSession("./logs/default/version_0/version_0.onnx")
    input_name = model.get_inputs()
    output_name = None
    # 如果出现Invalid Feed Input Name: 这一错误，可通过Netron软件观察输入名称，改变model内的输入输出的名称
    result = model.run(output_name, {input_name[0].name: input})
    return result[0]


# 输入为numpy矩阵
def test_tflite(input):
    import tensorflow as tf
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./logs/default/version_0/version_0.tflite")
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


if __name__ == "__main__":
    pass
