import tensorflow as tf
from tensorflow.python.platform import build_info

print('\n===== 关键信息 =====')
print('TensorFlow 版本:', tf.__version__)
print('CUDA 可用性:', tf.test.is_built_with_cuda())
print('GPU 设备列表:', tf.config.list_physical_devices('GPU'))
print('CUDA 版本:', build_info.build_info['cuda_version'])
print('cuDNN 版本:', build_info.build_info['cudnn_version'])

print('\n===== 库加载路径 =====')
import ctypes
print('cuBLAS:', ctypes.util.find_library('cublas'))
print('cuDNN:', ctypes.util.find_library('cudnn'))
print('cuFFT:', ctypes.util.find_library('cufft'))

print('\n===== GPU 计算测试 =====')
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print('矩阵乘法结果形状:', c.shape)
else:
    print('警告: 未检测到 GPU 设备')
