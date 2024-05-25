import json
import numpy as np
from os.path import join

# 实现标签检查器
def check_label_acc(A, B, onehotA=False, onehotB=False):
    """
    get correct label percent in all labels 获取所有标签中的正确标签百分比
    :param A: label A
    :param B: label B
    :param onehotA: bool, is label A in onehot?
    :param onehotB: bool, is label B in onehot?
    :return: matched percent in total labels 在总标签中匹配的百分比
    """
    A = np.argmax(A, axis=1) if onehotA else A # 如果onehotA为True，返回沿轴axis最大值的索引，onehotA表示A是否是onehot编码
    B = np.argmax(B, axis=1) if onehotB else B # 如果onehotB为True，返回沿轴axis最大值的索引，onehotB表示B是否是onehot编码

    try:
        assert A.shape == B.shape
    except:
        redux = min(A.shape[0], B.shape[0])
        A = A[:redux]
        B = B[:redux]

    t = np.sum(A == B)
    accu = t / len(A)
    return accu

def check_label_noisy2true(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    new_label = np.argmax(new_label, axis=1) if onehotA else new_label
    clean_label = np.argmax(clean_label, axis=1) if onehotB else clean_label

    try:
        assert new_label.shape == clean_label.shape
    except:
        redux = min(new_label.shape[0], clean_label.shape[0])
        new_label = new_label[:redux]
        clean_label = clean_label[:redux]

    assert new_label.shape == noise_or_not.shape
    assert new_label.shape == clean_label.shape

    # 计算噪声转为真实标签的个数，除以总个数得到准确率
    n2t_num = np.sum((new_label == clean_label).astype(np.int32) * (~noise_or_not).astype(np.int32))
    n2t = n2t_num / clean_label.shape[0]

    return n2t


def check_label_true2noise(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    new_label = np.argmax(new_label, axis=1) if onehotA else new_label
    clean_label = np.argmax(clean_label, axis=1) if onehotB else clean_label

    try:
        assert new_label.shape == clean_label.shape
    except:
        redux = min(new_label.shape[0], clean_label.shape[0])
        new_label = new_label[:redux]
        clean_label = clean_label[:redux]

    assert new_label.shape == noise_or_not.shape
    assert new_label.shape == clean_label.shape

    # 计算真实标签转为噪声标签的个数，除以总个数得到准确率
    t2n_num = np.sum((new_label != clean_label).astype(np.int32) * noise_or_not.astype(np.int32))
    t2n = t2n_num / clean_label.shape[0]

    return t2n

def check_label(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    # 检查标签的准确率，噪声转为真实标签的准确率，真实标签转为噪声标签的准确率
    acc = check_label_acc(new_label, clean_label, onehotA, onehotB)
    n2t = check_label_noisy2true(new_label, clean_label, noise_or_not, onehotA, onehotB)
    t2n = check_label_true2noise(new_label, clean_label, noise_or_not, onehotA, onehotB)
    return acc, n2t, t2n