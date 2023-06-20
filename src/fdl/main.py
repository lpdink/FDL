from . import __version__
import argparse
import types
import json
import pkg_resources
import re


def check(json_path):
    """
    1. 必须存在task, hparams, objects三个顶层属性
    2. 必须存在task.name, hparams.save_path
    3. objects中的对象，必须存在name, clazz。对象列表的name必须唯一。
    4. objects中的对象是反序构造的，引用的项目必须首先被构造
    """
    with open(json_path, "r") as json_file:
        content = json.load(json_file)
    assert all(item in content.keys() for item in ["task", "hparams", "objects"]), "json must have attrs:'task', 'hparams', 'objects' at top level."
    assert "name" in content["task"].keys() and "save_path" in content["hparams"].keys(), "json must have task.name and hparams.save_path"
    assert isinstance(content["objects"], list), f"objects must be list, got {type(content['objects'])}"
    obj_names = []
    for item in content["objects"]:
        assert isinstance(item, dict), f"item in objects must be dict, got {type(item)}"
        assert "name" in item.keys() and "clazz" in item.keys(), "item in objects must have attr: 'name', 'clazz'."
        assert item["name"] not in obj_names, f"item name in objects repeat:{item['name']}"
        obj_names.append(item["name"])
    # 按照列表的反序构造，检查被引用项必须首先被构造
    for item in content["objects"]:
        name = item["name"]
        name_idx = obj_names.index(name)
        if "args" in item.keys():
            for key, value in item["args"].item():
                match =re.search(r"\${(.+?)}", str(value)) 
                if match:
                    match_str = match.group(1)
                    assert match_str in obj_names, f"reference '{match_str}' not found in object '{name}'"
                    reference_idx = obj_names.index(match_str)
                    assert reference_idx<name_idx, f"objects are constructed  in reverse order, and item '{match_str}' must be constructed before being '{name}' referenced"
    else:
        print(f"json {json_file} check pass.")

"""
fdl -s
loop: classification, regression, GAN, CGAN, 
model: aaa, bbb, ccc, ddd
dataset: aaa, bbb, ccc, ddd, eee
loss:
opt:

"""

def buildjson(module_names):
    base_json_path = pkg_resources.resource_filename("fdl", "jsons/base.json")
    modules = ["loop", "model", "dataset"]
    file_paths = [
        pkg_resources.resource_filename("fdl", f"jsons/{module}.json")
        for module in modules
    ]
    usable_modules = dict()
    for file_path in file_paths:
        with open(file_path, "r") as json_file:
            partly_modules = json.load(json_file)
            usable_modules = {**usable_modules, **partly_modules}
    objects = [usable_modules[name] for name in module_names]
    with open(base_json_path, "r") as json_file:
        content = json.load(json_file)
        content["objects"] = objects
    with open("build_result.json", "w") as rst_file:
        json.dump(content, rst_file)
    print("save to build_result.json")


def getjson(type):
    print(f"getjson:{type}")


def show(type):
    print(f"show:{type}")


def version(show=True):
    if show:
        print(f"fdl {__version__}")


def train(json_path):
    print(f"train:{json_path}")


def infer(json_path):
    print(f"infer:{json_path}")


def deploy(json_path):
    print(f"deploy:{json_path}")


def make_parser():
    parser = argparse.ArgumentParser(
        description="fdl:Fast Deep Learning, train/infer/deploy your deep learning model with only a json file. All In One File!"
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        default=False,
        help="show version of fdl",
    )
    parser.add_argument(
        "-b",
        "--buildjson",
        nargs="*",
        default=None,
        type=str,
        help="build your own json with input modules name.",
    )
    parser.add_argument(
        "-g",
        "--getjson",
        default=None,
        help="copy prepared json to now dir.",
    )
    parser.add_argument(
        "-s",
        "--show",
        choices=["model", "dataset", "loop", "task", "all"],
        default=None,
        help="show usable modules info.",
    )
    parser.add_argument(
        "-c", "--check", default=None, help="check input json file valid."
    )
    parser.add_argument(
        "-t", "--train", default=None, help="train with input json file."
    )
    parser.add_argument(
        "-i", "--infer", default=None, help="infer with input json file."
    )
    parser.add_argument(
        "-d",
        "--deploy",
        default=None,
        help="convert model to onnx/pth/tvm with json file.",
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    # find func and exec with input param.
    for attr in dir(args):
        # must be our function and value is not None/False.
        if not attr.startswith("_") and getattr(args, attr):
            func = globals()[attr]
            if isinstance(func, types.FunctionType):
                func(getattr(args, attr))
                break
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
