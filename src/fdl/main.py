from . import __version__
import argparse
import types
import json
import pkg_resources


def check_valid(json_path):
    """
    1. 必须存在task, hparams, objects三个顶层属性
    2. 必须存在task.name, hparams.save_path
    3. objects中的对象，必须存在name, clazz。对象列表的name必须唯一。
    4. objects中对象的args引用的对象名必须出现在objects中，且不能循环构造.
    """

def buildjson(module_names):
    base_json_path = pkg_resources.resource_filename("fdl", "jsons/base.json")
    modules = ["loop", "model", "dataset"]
    file_paths = [pkg_resources.resource_filename("fdl", f"jsons/{module}.json") for module in modules]
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
    parser = argparse.ArgumentParser(description="fdl:Fast Deep Learning, train/infer/deploy your deep learning model with only a json file. All In One File!")
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
