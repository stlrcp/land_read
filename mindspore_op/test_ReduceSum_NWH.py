import numpy as np
import mindspore as ms

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations._grad_ops as G
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim import Adam
import time

ms.set_seed(1)

class Net(nn.Cell):
    def __init__(self, is_training=True):
        super(Net, self).__init__()
        self.reducesum = P.ReduceSum(keep_dims=True)
    def construct(self, x):
        y = self.reducesum(x, 2)
        return y

def check_result(a, b, atol=0.001, rtol=0.001, total_thresh=0.001):
    diff = np.abs(a-b)
    print(type(diff))
    t_df = np.size(diff)
    comp = atol + rtol * np.abs(b)
    lag = diff - comp # if lag > 0, this is a failed case
    num_failed = np.sum(lag > 0)
    has_passed = (num_failed / t_df) < total_thresh
    if not has_passed:
        n_print = 10
        idx_n = (-lag).argsort(axis=None)[:n_print]
        print(f"Print max_diff top {n_print} values... ")
        for i in range(n_print):
            idx_i = idx_n[i]
            pos_i = np.unravel_index(idx_i, a.shape)
            print(f"{i}: idx = {idx_i}, pos = {pos_i}, cpu = {a[pos_i]:.4f}, gpu = {b[pos_i]:.4f}, diff = {diff[pos_i]:.4f}, comp = {comp[pos_i]:.4f}, lag = {lag[pos_i]:.4f}")
    return has_passed

def test_ReduceSum(is_fp16):
    start = time.time()

    np_dtype = np.float16 if is_fp16 else np.float32

    np_x = np.random.randn(1, 16, 32, 1, 1, 512).astype(np_dtype)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = Net()
    x_cpu = Tensor(np_x)
    values_cpu = net(x_cpu)
    
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = Net()
    x_gpu = Tensor(np_x)
    net = net.set_train()
    values_gpu = net(x_gpu)

    passed = check_result(values_cpu.asnumpy(), values_gpu.asnumpy())
    if passed:
        print("test passed!!!!")
    else:
        print("test failed!!!")


    end = time.time()
    print('Elapsed time: %.4fs', end - start)



if __name__ == "__main__":
    test_ReduceSum(is_fp16=False)
