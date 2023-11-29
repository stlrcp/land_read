import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

np.random.seed(1234)

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

run_mod = relay.build(mod, target="iluvatar", params=None)

dev = tvm.iluvatar(0)
gen_module = graph_executor.GraphModule(run_mod["default"](dev))

x_data = np.random.rand(1, 56, 56, 64).astype("float32")
weight_data = np.random.rand(3, 3, 64, 32).astype("float32")

gen_module.set_input("x", x_data)
gen_module.set_input("weight", weight_data)
gen_module.run()

dshape=(1, 56, 56, 32)
out = tvm.nd.empty(dshape, device=dev)
out = gen_module.get_output(0, out)
print(out)