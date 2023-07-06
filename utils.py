# Includes some tools that are independent of the model
# 包含一些与模型无关的工具
import glob


def unzip_dir(zip_path, dir_path):
    """
    :param zip_path: Destination folder path|目标文件夹路径
    :param dir_path: result path|压缩后的文件夹路径
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


def get_ckpt_path(version_nth: int, log_name):
    checkpoints_path = f'./logs/{log_name}/version_{version_nth}/checkpoints'
    ckpt_path = glob.glob(checkpoints_path + '/*.ckpt')
    return ckpt_path[0].replace('\\', '/')


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
    pass


if __name__ == "__main__":
    load_logs_data('logs','Training loss')
    pass
