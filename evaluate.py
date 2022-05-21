import numpy as np
from PIL import Image
from os.path import join

from network_module.classify_eval import get_acc_without_background, get_precision
from network_module.iou import IOU


def evalute(n_classes, dataset_path, verbose=False):
    """
    Evaluate the prediction results under the specified folder, and the evaluation results do not count
    the background class
    评估指定文件夹下的预测结果, 评价结果均不计算背景类
    """
    iou = IOU(n_classes)

    test_list = open(join(dataset_path, 'test_dataset_list.txt').replace('\\', '/')).readlines()
    for ind in range(len(test_list)):
        pred = np.array(Image.open(join(dataset_path, 'prediction', test_list[ind].strip('\n')).replace('\\', '/')))
        label = np.array(Image.open(join(dataset_path, 'labels', test_list[ind].strip('\n')).replace('\\', '/')))
        if len(label.flatten()) != len(pred.flatten()):
            print('跳过{:s}: pred len {:d} != label len {:d},'.format(
                test_list[ind].strip('\n'), len(label.flatten()), len(pred.flatten())))
            continue

        iou.add_data(pred, label)

    # It must be placed before iou.get_miou() because iou.get_miou() clears HIST matrix
    # 必须置于iou.get_miou()前,因为iou.get_miou()会清除hist
    overall_acc, acc = get_acc_without_background(iou.hist), get_precision(iou.hist)
    mIoU, IoUs = iou.get_miou()

    if verbose:
        for ind_class in range(n_classes):
            print('===>' + str(ind_class) + ':\t' + str(IoUs[ind_class].float()))
        print('===> mIoU: ' + str(mIoU))
        print('===> overall accuracy:', overall_acc)
        print('===> accuracy of each class:', acc)


if __name__ == "__main__":
    evalute(9, './dataset/MFNet(RGB-T)-mini', verbose=True)
