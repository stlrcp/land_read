#include <dmlc/logging.h>
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/topi/cuda/injective.h>

#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/topi/generic/injective.h>

#include <cmath>
#include <string>

using namespace tvm;
using namespace tvm::relay;


int main(){
    using namespace tvm;
    using namespace tvm::te;

    auto n = var("8");
    auto m = var("16");

    LOG(INFO) << n;
    Array<PrimExpr> shape;
    shape.push_back(n);
    LOG(INFO) << shape;
    shape.push_back(m);
    LOG(INFO) << shape;

    auto A = placeholder(shape, DataType::Float(32), "A");
    auto B = placeholder(shape, DataType::Float(32), "B");
    LOG(INFO) << A;
    LOG(INFO) << A.ndim();
    
    auto C = compute(A->shape, [&](PrimExpr i, PrimExpr j){ return A[i][j] + B[i][j]; },"C");

    LOG(INFO) << C;
    LOG(INFO) << C->op;

    auto s = create_schedule({C->op});

    auto cAxis = C->op.as<ComputeOpNode>()->axis;

    IterVar bx, tx;
    s[C].split(cAxis[0], 64, &bx, &tx);

    auto args = Array<Tensor>({A, B, C});
    std::unordered_map<Tensor, Buffer> binds;

    auto target = Target("llvm");
    LOG(INFO) << target;

    auto lowered = LowerSchedule(s, args, "func", binds);
    auto module = build(lowered, target, Target());
    // auto json_f = module.GetFunction("get_graph_json", false);


    const runtime::PackedFunc* graph_executor =
      tvm::runtime::Registry::Get("tvm.graph_executor.create");
    LOG(INFO) << graph_executor;

    int cpu_dev_ty = static_cast<int>(kDLCPU);
    int cpu_dev_id = 0;

    std::string json =
        "{\"nodes\": [{\"op\": \"null\",\"name\": \"A\",\"inputs\": []}, "
        "{\"op\": \"null\",\"name\": \"B\",\"inputs\": []},{\"op\": \"tvm_op\", "
        "\"name\": \"func\",\"attrs\": {\"flatten_data\": \"0\",\"func_name\": \"func\", "
        "\"num_inputs\": \"2\",\"num_outputs\": \"1\"},\"inputs\": [[0,0,0],[1,0,0]]}], "
        "\"arg_nodes\": [0,1],\"node_row_ptr\": [0,1,2,3],\"heads\": [[2,0,0]], "
        "\"attrs\": {\"storage_id\": [\"list_int\",[0,1,2]],\"shape\": [\"list_shape\", "
        " [[8,16],[8,16],[8,16]]],\"device_index\": [\"list_int\",[1,1,1]],\"dltype\": "
        " [\"list_str\",[\"float32\",\"float32\",\"float32\"]]}}";

    LOG(INFO) << json;

    runtime::Module mod =
      (*graph_executor)(json, module, cpu_dev_ty, cpu_dev_id);

    auto a_val = runtime::NDArray::Empty({8, 16}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto b_val = runtime::NDArray::Empty({8, 16}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    LOG(INFO) << a_val;
    LOG(INFO) << b_val;
    auto pa = static_cast<float*>(a_val->data);
    auto pb = static_cast<float*>(b_val->data);

    for (int i = 0; i < 128; ++i) {
        pa[i] = i;
        pb[i] = i + 1;
    }

    auto set_input_f = mod.GetFunction("set_input", false);
    auto run_f = mod.GetFunction("run", false);
    auto get_output_f = mod.GetFunction("get_output", false);

    auto tmp = set_input_f("A", a_val);
    LOG(INFO) << A;
    set_input_f("B", b_val);
    LOG(INFO) << 111111;

    run_f();
    LOG(INFO) << 111111;
    tvm::runtime::NDArray Y = get_output_f(0);
    auto pY = static_cast<float*>(Y->data);
    for (int i = 0; i < 128; ++i) {
        // LOG(INFO) << pY[i];
        ICHECK_LT(fabs(pY[i] - (i + (i + 1))), 1e-4);
    }

    return 0;
}