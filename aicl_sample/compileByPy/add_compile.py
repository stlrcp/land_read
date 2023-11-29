
### version one
'''
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.op.contrib import arm_compute_lib
from tvm.contrib import graph_executor
from tvm.contrib import utils

from builtins import print as _print
from sys import _getframe

np.random.seed(1234)

# 自定义print函数
def print(*arg, **kw):
    s = f'{_getframe(1).f_lineno} : '        
    return _print(f"《{__name__}》-{s}", *arg, **kw)

for inputs in [   
    {"a": tvm.nd.array(np.random.uniform(-10, 10, (10,10)).astype("float32")),
     "b": tvm.nd.array(np.random.uniform(-10, 10, (10,10)).astype("float32")),}]:
    print("=================================")
    print(inputs)

tm_puts = {"a": tvm.nd.array(np.random.uniform(-10, 10, (10,10)).astype("float32")), "b": tvm.nd.array(np.random.uniform(-10, 10, (10,10)).astype("float32")),}
tmp_iter = iter(tm_puts)

shape = (10, 10)

a = relay.var(next(tmp_iter), shape=shape, dtype="float32")
b = relay.var(next(tmp_iter), shape=shape, dtype="float32")
print("a = ", a)
print("b = ", b)
func = relay.add(a, b)
# print(func)
mod = tvm.IRModule.from_expr(func)
print(mod)
run_mod = relay.build(mod, target="iluvatar", params=None)
print(run_mod.function_metadata["__tvm_main__"])
run_mod.export_library("./add.so")
print("==============================")

dev = tvm.iluvatar(0)
# dev = tvm.cpu(0)
# gen_module = graph_executor.GraphModule(run_mod["default"](tvm.cpu()))
gen_module = graph_executor.GraphModule(run_mod["default"](dev))
print(gen_module)

print(" *shape = ", *shape)
x_data = np.random.rand(*shape).astype("float32")
y_data = np.random.rand(*shape).astype("float32")

# z_data = np.random.rand(*shape, )
# print(z_data)

# x_data = tvm.nd.array(x_data, tvm.iluvatar())
# y_data = tvm.nd.array(y_data, tvm.iluvatar())

gen_module.set_input("a", x_data)
gen_module.set_input("b", y_data)
gen_module.run()


out = tvm.nd.empty(shape, device=dev)
out = gen_module.get_output(0, out)
print(out)
'''


### version two
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

np.random.seed(1234)

dshape = (10, 10)
x1 = relay.var("x1", shape=dshape)
x2 = relay.var("x2", shape=dshape)

func = relay.add(x1, x2)

mod = tvm.IRModule.from_expr(func)

run_mod = relay.build(mod, target="iluvatar", params=None)

dev = tvm.iluvatar(0)
gen_module = graph_executor.GraphModule(run_mod["default"](dev))

x_data = np.random.rand(*dshape).astype("float32")
y_data = np.random.rand(*dshape).astype("float32")

gen_module.set_input("x1", x_data)
gen_module.set_input("x2", y_data)
gen_module.run()

out = tvm.nd.empty(dshape, device=dev)
out = gen_module.get_output(0, out)
print(out)





