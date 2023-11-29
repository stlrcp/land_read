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