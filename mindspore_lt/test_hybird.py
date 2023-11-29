### Hybird类型的自定义算子开发

import numpy as np
from mindspore import ops
import mindspore as ms
from mindspore.ops import ms_kernel

ms.set_context(device_target="GPU")

# 算子实现，Hybrid DSL
@ms_kernel
def add(a, b):
    c = output_tensor(a.shape, a.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
    return c

if __name__ == "__main__":
    # 定义hybrid类型的自定义算子(Custom的默认模式)
    op = ops.Custom(add)
    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)
