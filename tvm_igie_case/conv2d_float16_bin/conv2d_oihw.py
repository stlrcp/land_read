import inspect

current_frame = inspect.currentframe()

print(f"===== line : {inspect.getframeinfo(current_frame).lineno} ========")



import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

np.random.seed(1234)

x = relay.var("x", shape=(2, 1024, 1024, 3), dtype="float16")

weight = relay.var("weight", shape=(6, 3, 3, 3), dtype="float16")
func = relay.nn.conv2d(x,
                       weight,
                       channels=6,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       data_layout="NHWC",
                       kernel_layout="OIHW",)

mod = tvm.IRModule.from_expr(func)

print(mod)

desired_layout = {'nn.conv2d' : ['NHWC', 'HWIO']}
with tvm.transform.PassContext(opt_level=3):
    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layout)])
    mod = seq(mod)
print(mod)

target = tvm.target.iluvatar(model="MR", options="-libs=cudnn")
run_mod = relay.build(mod, target=target, params=None)

run_mod.export_library("./conv.so")


dev = tvm.iluvatar(0)
gen_module = graph_executor.GraphModule(run_mod["default"](dev))


# x_data = np.random.rand(2, 1024, 1024, 3).astype("float16")
# x_data.tofile("input")
x_data = np.fromfile("x_float16_2x1024x1024x3.bin.in", dtype=np.float16).reshape([2, 1024, 1024, 3])

# weight_data = np.random.rand(6, 3, 3, 3).astype("float16")
# weight_data.tofile("weight")
weight_data = np.fromfile("filter_float16_6x3x3x3.bin.in", dtype=np.float16).reshape([6, 3, 3, 3])

gen_module.set_input("x", x_data)
gen_module.set_input("weight", weight_data)
gen_module.run()

# dshape=(2, 1024, 1024, 6)
dshape=(2, 1024, 1024, 6)
out = tvm.nd.empty(dshape, dtype="float16", device=dev)
out = gen_module.get_output(0, out)
print(out)
outdata = out.asnumpy()
outdata.tofile("out_float16_2x1024x1024x6.bin.in")




