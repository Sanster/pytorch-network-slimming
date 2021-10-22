import copy
import typing
from collections import defaultdict
from typing import Callable, Dict, List

import torch

from pns.functional import is_depthwise_conv2d

TRACK_ATTR_NAME = "_tracker_attr"
TRACK_ATTR_MODULE_NAME = "_tracker_model_name"


def attach_tracker(
    ctx: "TrackContext", method: Callable, tracker: Callable, method_str
):
    """

    Gets a function that executes PyTorch method_str and TensorRT tracker

    Args:
        ctx:
        method: pytorch method_str
        tracker:
        method_str:

    Returns:

    """

    def wrapper(*args, **kwargs):
        skip = True
        # run original method_str
        outputs = method(*args, **kwargs)

        if not ctx.lock:
            ctx.lock = True
            skip = False

        if not skip:
            ctx.method_args = args
            ctx.method_kwargs = kwargs
            ctx.method_return = outputs
            ctx.method_str = method_str

            tracker(ctx)

            # convert to None so conversion will fail for unsupported layers
            ctx.method_args = None
            ctx.method_kwargs = None
            ctx.method_return = None
            ctx.lock = False

        return outputs

    return wrapper


class ModuleHook:
    """Attaches Track to PyTorch method_str call"""

    def __init__(self, ctx: "TrackContext", method_str: str, tracker: Callable):
        """

        Args:
            ctx:
            method_str: pytorch method name string
            tracker:
        """
        self.ctx = ctx
        self.method_str = method_str
        self.tracker = tracker

    def _set_method(self, method: Callable):
        exec("%s = method" % self.method_str)

    def __enter__(self):
        try:
            self.method_impl = eval(self.method_str)
        except AttributeError:
            self.method_impl = None

        if self.method_impl:
            self._set_method(
                attach_tracker(
                    self.ctx, self.method_impl, self.tracker, self.method_str
                )
            )

    def __exit__(self, type, val, tb):
        if self.method_impl:
            self._set_method(self.method_impl)


TRACKERS = {}


def register_tracker(method: str):
    global TRACKERS

    def _wrapper(tracker):
        TRACKERS[method] = tracker
        return tracker

    return _wrapper


class TrackContext:
    def __init__(
        self,
        module_trackers=TRACKERS,
    ):
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.hooks = [
            ModuleHook(self, method, wrapper)
            for method, wrapper in module_trackers.items()
        ]

        self.module_input_names = defaultdict(list)
        self.module_output_names = defaultdict(list)

        self.shortcuts_group = []
        self.cat_group = []

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)


class ModuleWrapper:
    def __init__(self, name, module):
        self.name = name
        self.module = module

    def is_bn(self):
        return isinstance(self.module, torch.nn.BatchNorm2d)

    def is_cat(self):
        pass

    def is_conv(self):
        return isinstance(self.module, torch.nn.Conv2d)

    def is_fc(self):
        return isinstance(self.module, torch.nn.Linear)


def BFS_find_bn(
    module_names: Dict[str, List], wrappers: Dict[str, ModuleWrapper], name: str
) -> str:
    output_names = module_names[name]
    while len(output_names) != 0:
        output_name = output_names.pop(0)
        if wrappers[output_name].is_bn():
            return wrappers[output_name].name
        else:
            output_names.extend(module_names[output_name])

    return ""


def is_private(method):
    method = method.split(".")[-1]  # remove prefix
    return method[0] == "_" and method[1] is not "_"


def is_function_type(method):
    fntype = eval(method + ".__class__.__name__")
    return (
        fntype == "function"
        or fntype == "builtin_function_or_method"
        or fntype == "method_descriptor"
    )


def get_methods(namespace):
    methods = []
    for method in dir(eval(namespace)):
        full_method = namespace + "." + method
        if not is_private(full_method) and is_function_type(full_method):
            methods.append(full_method)
    return methods


TORCH_METHODS = []
TORCH_METHODS += get_methods("torch")
TORCH_METHODS += get_methods("torch.Tensor")
TORCH_METHODS += get_methods("torch.nn.functional")


def set_outputs_name_attr(outputs, module_name):
    if outputs is None:
        return

    if isinstance(outputs, torch.Tensor):
        setattr(outputs, TRACK_ATTR_NAME, module_name)
        return

    if isinstance(outputs, list) or isinstance(outputs, tuple):
        for output in outputs:
            set_outputs_name_attr(output, module_name)
    elif isinstance(outputs, dict):
        for output in outputs.values():
            set_outputs_name_attr(output, module_name)


def save_input_output_names(ctx: TrackContext):
    inputs = ctx.method_args
    outputs = ctx.method_return
    if len(inputs) == 0:
        return

    module_name = getattr(inputs[0], TRACK_ATTR_MODULE_NAME, None)

    if module_name is not None:
        for input in inputs:
            name = getattr(input, TRACK_ATTR_NAME, None)
            if name is not None:
                if isinstance(name, list):
                    ctx.module_input_names[module_name].extend(name)
                    for it in name:
                        ctx.module_output_names[it].append(module_name)
                else:
                    ctx.module_input_names[module_name].append(name)
                    ctx.module_output_names[name].append(module_name)

        set_outputs_name_attr(outputs, module_name)
        # output_names = getattr(outputs, TRACK_ATTR_NAME, None)
        # a = 0


def pass_input_names(ctx: TrackContext):
    inputs = ctx.method_args
    outputs = ctx.method_return
    if len(inputs) == 0:
        return

    input_names = []
    for input in inputs:
        name = getattr(input, TRACK_ATTR_NAME, None)
        if name is not None:
            if isinstance(name, list):
                input_names.extend(name)
            else:
                input_names.append(name)

    set_outputs_name_attr(outputs, input_names)
    output_names = getattr(outputs, TRACK_ATTR_NAME, None)
    a = 0


for method in TORCH_METHODS:

    @register_tracker(method)
    def one_tracker(ctx: TrackContext):
        # 除了卷积核 BN 之外的层，不需要保存输入输出的名字，只把 input 中的名字传给 output
        pass_input_names(ctx)


@register_tracker("torch.nn.BatchNorm2d.forward")
def track_BatchNorm2d(ctx: TrackContext):
    save_input_output_names(ctx)


@register_tracker("torch.nn.Conv2d.forward")
def track_Conv2d(ctx: TrackContext):
    save_input_output_names(ctx)


@register_tracker("torch.nn.Linear.forward")
def track_Linear(ctx: TrackContext):
    save_input_output_names(ctx)


# TODO: check function inputs, add all func len(inputs) >= 2
@register_tracker("torch.add")
@register_tracker("torch.Tensor.__iadd__")
@register_tracker("torch.Tensor.__add__")
@register_tracker("torch.Tensor.__radd__")
@register_tracker("torch.sub")
@register_tracker("torch.Tensor.__isub__")
@register_tracker("torch.Tensor.__sub__")
@register_tracker("torch.Tensor.__rsub__")
def track_add(ctx: TrackContext):
    inputs = ctx.method_args
    outputs = ctx.method_return
    if len(inputs) == 0:
        return

    input_names = []
    for input in inputs:
        name = getattr(input, TRACK_ATTR_NAME, None)
        if name is None:
            continue

        if isinstance(name, list):
            input_names.extend(name)
            if tuple(name) in ctx.shortcuts_group:
                ctx.shortcuts_group.remove(tuple(name))
        else:
            input_names.append(name)

    if len(input_names) > 1:
        ctx.shortcuts_group.append(tuple(input_names))
        set_outputs_name_attr(outputs, input_names)


@register_tracker("torch.cat")
def track_cat(ctx: TrackContext):
    inputs = ctx.method_args
    outputs = ctx.method_return
    if len(inputs) == 0:
        return

    input_names = []
    for input in inputs[0]:
        name = getattr(input, TRACK_ATTR_NAME, None)
        if name is None:
            continue

        if isinstance(name, list):
            input_names.extend(name)
            if tuple(name) in ctx.cat_group:
                ctx.cat_group.remove(tuple(name))
        else:
            input_names.append(name)

    if len(input_names) > 1:
        ctx.cat_group.append(tuple(input_names))
        set_outputs_name_attr(outputs, input_names)


def gen_pruning_schema(model, *args, **kwargs):
    with TrackContext() as ctx:
        bn_names = []
        depthwise_conv2d_names = []
        for name, module in model.named_modules():
            setattr(module, TRACK_ATTR_MODULE_NAME, name)
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_names.append(name)
            if is_depthwise_conv2d(module):
                depthwise_conv2d_names.append(name)

        model(*args, **kwargs)

        common_names = set(ctx.module_input_names.keys()) | set(
            ctx.module_output_names.keys()
        )
        # print(f"module input names: {ctx.module_input_names}")
        # print(f"module output names: {ctx.module_output_names}")
        # print(f"module common names: {common_names}")
        # print(f"shortcuts: {ctx.shortcuts_group}")

        target_wrappers = {}
        for name, module in model.named_modules():
            if name in common_names:
                target_wrappers[name] = ModuleWrapper(name, module)

        info = {
            "modules": [],
            "shortcuts": [],
            "cats": [],
            "depthwise_conv_adjacent_bn": [],
        }

        module_input_names = copy.deepcopy(ctx.module_input_names)
        module_output_names = copy.deepcopy(ctx.module_output_names)
        for name, wrapper in target_wrappers.items():
            if not (wrapper.is_conv() or wrapper.is_fc()):
                continue

            # module_input_names will be consumed
            prev_bn = BFS_find_bn(module_input_names, target_wrappers, name)
            next_bn = BFS_find_bn(module_output_names, target_wrappers, name)
            m = {"name": name, "prev_bn": prev_bn, "next_bn": next_bn}

            info["modules"].append(m)

            if name in depthwise_conv2d_names:
                if prev_bn and next_bn:
                    info["depthwise_conv_adjacent_bn"].append(
                        {"names": [prev_bn, next_bn], "method": "or"}
                    )

        for shortcuts in ctx.shortcuts_group:
            shortcuts = list(filter(lambda it: it in bn_names, shortcuts))
            if len(shortcuts) <= 1:
                continue
            info["shortcuts"].append({"names": sorted(shortcuts), "method": "or"})

        for cat in ctx.cat_group:
            cat = list(filter(lambda it: it in bn_names, cat))
            if len(cat) <= 1:
                continue
            """
            ctx.module_input_names:
            {
                "conv": ["bn1", "bn2"]
            }
            
            cat: ['bn1', 'bn2']
            
            如果 module_input_names 中 conv 的值和 cat 的值一样，说明该 conv 是 cat 的输出
            """
            # cat 不能 sort，因为顺序会影响 channel index
            cats = {"input_bn_names": cat, "output_conv_names": []}
            for name, values in ctx.module_input_names.items():
                if name not in target_wrappers:
                    continue
                if not target_wrappers[name].is_conv():
                    continue

                if values == cat:
                    cats["output_conv_names"].append(name)

            if len(cats["output_conv_names"]) != 0:
                info["cats"].append(cats)

        return info
