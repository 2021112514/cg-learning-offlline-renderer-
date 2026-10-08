# 该文件由DeepSeek生成
import numpy
import cupy
from typing import TYPE_CHECKING


device = 'GPU'

# 类型注解：cupy 未随包提供完整的类型 stub（没有 py.typed），IDE 无法解析
# cupy.ndarray 的成员，且 cupy.ndarray 不支持泛型下标。
# cupy 与 numpy 的 API 一致，因此注解统一使用 numpy.ndarray（仅用于静态检查，不影响运行）。
Array = numpy.ndarray

if TYPE_CHECKING:
    # 静态检查时按 numpy 解析，以获得完整的 API 提示；运行时仍按 device 选择后端
    np = numpy
else:
    np = cupy if device == 'GPU' else numpy


def asnumpy(x: Array) -> numpy.ndarray:
    """把数组转成主机内存的 numpy 数组（GPU 后端会同步，CPU 后端原样返回）"""
    return cupy.asnumpy(x)

