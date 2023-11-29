#include <tvm/driver/driver_api.h>
#include <tvm/ir/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/generic/injective.h>

using namespace tvm;
using namespace tvm::relay;

TVM_REGISTER_GLOBAL("test.strategy")
    .set_body_typed([](const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type,
                       const Target& target) {
      FTVMCompute fcompute = [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
      };
      FTVMSchedule fschedule = [](const Attrs& attrs, const Array<te::Tensor>& outs,
                                  const Target& target) {
        With<Target> target_scope(target);
        LOG(INFO) << target;
        LOG(INFO) << topi::generic::schedule_injective(target, outs);
        return topi::generic::schedule_injective(target, outs);
      };

      auto n = make_object<OpStrategyNode>();
      auto strategy = tvm::relay::OpStrategy(std::move(n));
      strategy.AddImplementation(fcompute, fschedule, "test.strategy", 10);
      LOG(INFO) << strategy;
      return strategy;
    });

TVM_REGISTER_GLOBAL("relay.backend.lower_call")
    .set_body_typed([](const relay::Call& call, const Array<te::Tensor>& inputs,
                       const Target& target) {
      static auto fstrategy = Op::GetAttrMap<relay::FTVMStrategy>("FTVMStrategy");
      Op op = Downcast<Op>(call->op);
      auto out_type = call->checked_type();
      OpStrategy strategy = fstrategy[op](call->attrs, inputs, out_type, target);
      auto impl = strategy->specializations[0]->implementations[0];
      auto outs = impl.Compute(call->attrs, inputs, out_type);
      auto f = tvm::runtime::Registry::Get("relay.backend._make_LoweredOutput");
      if (!f) {
        LOG(FATAL) << "relay.backend._make_LoweredOutput is not registered";
      }
      return (*f)(outs, impl);
    });

int main(int argc, char* argv[]) {
  LOG(INFO) << "0000";
  auto tensor_type = relay::TensorType({2, 3}, DataType::Float(32));
  auto a = relay::Var("a", tensor_type);
  auto b = relay::Var("b", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto x = relay::Call(add_op, {a, b}, tvm::Attrs(), {});
  auto c = relay::Var("c", tensor_type);
  auto y = relay::Call(add_op, {x, c}, tvm::Attrs(), {});
  LOG(INFO) << 11111;
  auto func = relay::Function(relay::FreeVars(y), y, relay::Type(), {});
  auto A = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto B = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto C = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pA = static_cast<float*>(A->data);
  auto pB = static_cast<float*>(B->data);
  auto pC = static_cast<float*>(C->data);

  // float* pA = new float[6];
  // float* pB = new float[6];
  // float* pC = new float[6];
  for (int i = 0; i < 6; ++i) {
    pA[i] = i;
    pB[i] = i + 1;
    pC[i] = i + 2;
  }
  // A.CopyFromBytes(pA, sizeof(float) * 6);
  // B.CopyFromBytes(pB, sizeof(float) * 6);
  // C.CopyFromBytes(pC, sizeof(float) * 6);

  LOG(INFO) << 11111;
  // get schedule
  auto reg = tvm::runtime::Registry::Get("ir.RegisterOpAttr");
  if (!reg) {
    LOG(INFO) << "no _Register";
  }
  auto reset = tvm::runtime::Registry::Get("ir.OpResetAttr");
  if (!reset) {
    LOG(INFO) << "Reset is not defined.";
  }
  auto fs = tvm::runtime::Registry::Get("test.strategy");

  if (!fs) {
    LOG(INFO) << "No test_strategy registered.";
  }
  auto fgeneric = GenericFunc::Get("test.strategy_generic").set_default(*fs, true);
  (*reset)(add_op, "FTVMStrategy");
  (*reg)("add", "FTVMStrategy", fgeneric, 10);
  Array<Integer> dep;
  dep.push_back(0);
  (*reset)(add_op, "TShapeDataDependent");
  (*reg)("add", "TShapeDataDependent", dep, 10);
  LOG(INFO) << 11111;

  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  Map<tvm::Integer, tvm::Target> targets;
  Target llvm_tgt = Target("llvm");
  targets.Set(0, llvm_tgt);
  LOG(INFO) << 11111;
  auto relay_mod = tvm::IRModule::FromExpr(func);
  ICHECK(relay_mod.defined()) << "Module must be defined";
  build_f(relay_mod, targets, llvm_tgt, Executor::Create("graph"), Runtime::Create("cpp"), "");
  std::string json = json_f();
  LOG(INFO) << json;
  tvm::runtime::Module mod = mod_f();
  LOG(INFO) << 11111;

  // run
  auto dev = A->device;
  auto pfr = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  ICHECK(mod.defined()) << "Module must be defined";
  tvm::runtime::Module run_mod =
      (*pfr)(json, mod, static_cast<int>(dev.device_type), dev.device_id);
  auto set_input_f = run_mod.GetFunction("set_input", false);
  auto run_f = run_mod.GetFunction("run", false);
  auto get_output_f = run_mod.GetFunction("get_output", false);
  LOG(INFO) << 11111;
  set_input_f("a", const_cast<DLTensor*>(A.operator->()));
  set_input_f("b", const_cast<DLTensor*>(B.operator->()));
  set_input_f("c", const_cast<DLTensor*>(C.operator->()));
  run_f();
  tvm::runtime::NDArray Y = get_output_f(0);
  auto pY = static_cast<float*>(Y->data);
  for (int i = 0; i < 6; ++i) {
    ICHECK_LT(fabs(pY[i] - (i + (i + 1) + (i + 2))), 1e-4);
  }



  // mutate the input a bit and run it again
  for (int i = 0; i < 6; ++i) {
    pB[i] = i + 3;
  }
  set_input_f("a", const_cast<DLTensor*>(A.operator->()));
  set_input_f("b", const_cast<DLTensor*>(B.operator->()));
  set_input_f("c", const_cast<DLTensor*>(C.operator->()));
  run_f();
  tvm::runtime::NDArray Y2 = get_output_f(0);
  auto pY2 = static_cast<float*>(Y2->data);
  for (int i = 0; i < 6; ++i) {
    ICHECK_LT(fabs(pY2[i] - (i + (i + 3) + (i + 2))), 1e-4);
  }


  // attach a different input and run it again
  auto C2 = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto pC2 = static_cast<float*>(C2->data);
  for (int i = 0; i < 6; ++i) {
    pC2[i] = i + 4;
  }
  set_input_f("c", const_cast<DLTensor*>(C2.operator->()));
  run_f();
  tvm::runtime::NDArray Y3 = get_output_f(0);
  auto pY3 = static_cast<float*>(Y3->data);
  for (int i = 0; i < 6; ++i) {
    ICHECK_LT(fabs(pY3[i] - (i + (i + 3) + (i + 4))), 1e-4);
  }

}