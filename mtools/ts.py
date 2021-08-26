import numpy as np

from typing import Sequence, List, Iterator, Dict, Any


def selecter_by_window(x_iterator: Sequence, window: int = 3, stride: int = 1) -> List:
    """对序列型数据按照指定stride和窗口(window)大小进行选取, 并生成新的序列, 类似卷积滑动的操作
    Args:
        x_iterator: Sequence, 原始的序列数据
        window: int, 窗口大小设置
        stride: int, 间隔大小

    Returns:
        List[Sequence, ...], 按窗口大小对序列中的元素组合成子序列, 并以List返回
    """
    max_i = len(x_iterator)
    r = []
    for i in range(0, max_i, stride):
        if i + window > max_i:
            break
        r.append(x_iterator[i: i+window])
    return r


def roll_slide_sample(data: np.ndarray, axis: int = 1, train_window: int = 168, predict_window: int = 168,
                      dim_map_index: List = None) -> Iterator:
    """对多维的array按照某axis进行滑动窗口进行样本生成, 生成按照生成器的方式返回每个时间窗口的样本数据
    Args:
        data: ndarray, 多维度的数据集, 一般某个维度为时间维度
        axis: int, 指定滑动窗口的维度或轴
        train_window: int, 设置训练使用的窗口大小
        predict_window: int, 设置预测使用的窗口大小
        dim_map_index: int, 某个维度的index设置, 必须时间维度的index: ['2020-01', '2020-02', '2020-03', ...]

    Returns:
        Iterator[Tuple[index, ndarray]], 生训练窗口和预测窗口为大小的样本数据, 分别输出x_tuple, y_tuple
            x_tuple包含了x在axis轴上的index和数据集
            y_tuple包含了y在axis轴上的index和数据集
    """
    _shape = data.shape
    if dim_map_index is None:
        dim_map_index = [np.arange(length) for length in _shape]
    else:
        if len(dim_map_index) != len(_shape):
            raise ValueError("The input param `dim_map_index` length is not equal data dims.")
    if _shape[axis] < train_window + predict_window:
        print("input data array length < train_window + predict_window for axis, this cannot roll train.")
        return
    index_for_axis = dim_map_index[axis]
    if isinstance(index_for_axis, list):
        index_for_axis = np.ndarray(index_for_axis).flatten()
    for axis_of_index in selecter_by_window(list(range(_shape[axis])), window=train_window + predict_window):
        xy = np.take(data, axis_of_index, axis=axis)  # 获取到序列数据的xy序列数据
        x = np.take(xy, np.arange(train_window).tolist(), axis=axis)
        y = np.take(xy, np.arange(train_window, train_window + predict_window).tolist(), axis=axis)
        yield (index_for_axis[axis_of_index[:train_window]], x), \
              (index_for_axis[axis_of_index[train_window:]], y)
