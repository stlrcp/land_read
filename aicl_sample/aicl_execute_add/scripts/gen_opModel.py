import json
import argparse
import tvm
from tvm import relay

parser = argparse.ArgumentParser(description="argument of gen_op")


def load_data(op_path):
    with open(op_path, "r") as file:
        data = json.load(file)
    file.close()

    return data

def get_funclist(file_path):
    import ast
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    function_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
    return function_names

def gen_opmodel(op_data, out_path):
    print(op_data)
    op_name = op_data[0]["op"].lower()
    in_desc = op_data[0]['input_desc']
    out_desc = op_data[0]['output_desc']
    print(relay.op.tensor.__file__)
    ptensor_file = relay.op.tensor.__file__


    funclist = get_funclist(ptensor_file)
    inp_dir = {}
    funcstr = "relay."
    if op_name in funclist:
        funcstr = funcstr + op_name + "("
        for i, in_d in enumerate(in_desc):
            inp_str = f"input{i}"
            inp_shape = in_d["shape"]
            inp_type = in_d["type"]
            var_str = f"relay.var('{inp_str}', shape={inp_shape}, dtype='{inp_type}')"
            inp_dir[inp_str] = eval(var_str)
            if i < (len(in_desc)-1):
                funcstr = funcstr + f"inp_dir['input{i}']" + ","
            else:
                funcstr = funcstr + f"inp_dir['input{i}']" + ")"
        
    func = eval(funcstr)
    mod = tvm.IRModule.from_expr(func)
    run_mod = relay.build(mod, target="iluvatar", params=None)

    save_path = out_path + f"/{op_name}.so"
    run_mod.export_library(save_path)

    
if __name__ == "__main__":
    parser.add_argument("--singleop", type=str, help="file path of op.json")
    parser.add_argument("--soc", type=str, default="MR", help="type of soc")
    parser.add_argument("--output", type=str, help="save path of output")
    args = parser.parse_args()
    OpPath = args.singleop
    OutPath = args.output
    OpData = load_data(OpPath)
    gen_opmodel(OpData, OutPath)


