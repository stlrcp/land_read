import inspect

current_frame = inspect.currentframe()

print(f"===== line : {inspect.getframeinfo(current_frame).lineno} ========")



import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

np.random.seed(1234)

x = relay.var("x", shape=(2, 1024, 1024, 3))
# x = relay.var("x", shape=(2, 3, 1024, 1024))
# weight = relay.var("weight", shape=(3, 3, 3, 6))
weight = relay.var("weight", shape=(6, 3, 3, 3))
func = relay.nn.conv2d(x,
                       weight,
                       channels=6,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       data_layout="NHWC",
                       kernel_layout="OHWI",)

# func = relay.add(x1, x2)

mod = tvm.IRModule.from_expr(func)

# run_mod = relay.build(mod, target="iluvatar", params=None)
# target = tvm.target.iluvatar(model="MR", options="-libs=cudnn,cublas,ixinfer")
target = tvm.target.iluvatar(model="MR", options="-libs=cudnn")
run_mod = relay.build(mod, target=target, params=None)

run_mod.export_library("./conv.so")


dev = tvm.iluvatar(0)
gen_module = graph_executor.GraphModule(run_mod["default"](dev))

# x_data = np.random.rand(2, 1024, 1024, 3).astype("float32")
x_data = np.random.rand(2, 1024, 1024, 3).astype("float32")
np.savetxt("./input_data.txt", x_data.reshape(1024*1024*6))
# weight_data = np.random.rand(3, 3, 3, 6).astype("float32")
weight_data = np.random.rand(6, 3, 3, 3).astype("float32")
np.savetxt("./weight_data.txt", weight_data.reshape(9*6*3))

gen_module.set_input("x", x_data)
gen_module.set_input("weight", weight_data)
gen_module.run()

# dshape=(2, 1024, 1024, 6)
dshape=(2, 1024, 1024, 6)
out = tvm.nd.empty(dshape, device=dev)
out = gen_module.get_output(0, out)
print(out)
outdata = out.asnumpy()
np.savetxt("./out_data.txt", outdata.reshape(12*1024*1024))




