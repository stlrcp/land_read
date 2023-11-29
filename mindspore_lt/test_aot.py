import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="GPU")

if __name__ == "__main__":
    # 定义 aot 类型的自定义算子
    op = ops.Custom("./add.so:CustomAdd", out_shape=lambda x, _: x, out_dtype=lambda x, _: x, func_type="aot")
    x0 = np.array([[0.0, 0.0], [1.0, 1.0]]).astype(np.float32)
    x1 = np.array([[2.0, 2.0], [3.0, 3.0]]).astype(np.float32)
    output = op(ms.Tensor(x0), ms.Tensor(x1))
    print(output)