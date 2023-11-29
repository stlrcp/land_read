import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

np.random.seed(1234)

def igie_conv():
    x = relay.var("x", shape=(1, 56, 56, 64))
    weight = relay.var("weight", shape=(3, 3, 64, 32))
    func = relay.nn.conv2d(x,
                        weight,
                        channels=32,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        data_layout="NHWC",
                        kernel_layout="HWIO",)

    # func = relay.add(x1, x2)

    mod = tvm.IRModule.from_expr(func)

    # run_mod = relay.build(mod, target="iluvatar", params=None)
    run_mod = relay.build(mod, target="llvm", params=None)

    # dev = tvm.iluvatar(0)
    dev = tvm.cpu(0)
    gen_module = graph_executor.GraphModule(run_mod["default"](dev))

    x_data = np.random.rand(1, 56, 56, 64).astype("float32")
    np.savetxt("./input_data.txt", x_data.reshape(56*56*64))
    print(x_data)

    weight_data = np.random.rand(3, 3, 64, 32).astype("float32")
    np.savetxt("./weight_data.txt", weight_data.reshape(9*64*32))
    print(weight_data)

    gen_module.set_input("x", x_data)
    gen_module.set_input("weight", weight_data)
    gen_module.run()

    dshape=(1, 56, 56, 32)
    out = tvm.nd.empty(dshape, device=dev)
    out_data = gen_module.get_output(0, out)
    print(out_data)
    outdata = out_data.asnumpy()
    np.savetxt("./out_data.txt", outdata.reshape(56*56*32))



def torch_conv():
    import torch
    from collections import OrderedDict
    conv = torch.nn.Conv2d(64, 32, 
                           kernel_size=(3, 3), 
                           stride=1, 
                           padding=(1, 1), 
                           dilation=1)
    x_data = np.loadtxt("./test.txt").reshape(1, 56, 56, 64).astype("float32")
    # print(x_data)
    x_tensor = torch.tensor(x_data)
    weight = np.loadtxt("./weight.txt").reshape(3, 3, 64, 32).astype("float32")
    w_tensor = torch.tensor(weight).permute(3, 2, 0, 1)
    b_tensor = torch.zeros(32, dtype=torch.float32)
    in_tensor = x_tensor.permute(0, 3, 1, 2)

    # conv.load_state_dict(w_tensor)
    # model_dict = conv.state_dict()
    d = OrderedDict()
    d['weight'] = w_tensor
    d['bias'] = b_tensor
    # for key, value in model_dict.items():
    #     print("key = ", key)
    #     print("value = ", value.shape)

    print(d)
    conv.load_state_dict(d)

    # print(model_dict)
    out = conv(in_tensor)
    print(out.detach().numpy())
    np.savetxt("./out.txt", out.detach().numpy().reshape(56*56*32))


if __name__ == "__main__":
    torch_conv()
    # igie_conv()

