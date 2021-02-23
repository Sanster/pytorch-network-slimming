import os
import sys
from pathlib import Path

from helper import check_gen_schema as _check_schema

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(CURRENT_DIR.parent))
from backbone.build import build_model


def check_schema(net):
    model = build_model(net)
    _check_schema(model)


def test_RepVGG_A0_woid():
    name = "RepVGG-A0-woid"
    check_schema(name)


def test_RepVGG_A0():
    name = "RepVGG-A0"
    check_schema(name)


def test_RepVGG_B2_woid():
    name = "RepVGG-B2-woid"
    check_schema(name)


def test_RepVGG_B2():
    name = "RepVGG-B2"
    check_schema(name)


def test_RepVGG_B3_woid():
    name = "RepVGG-B3-woid"
    check_schema(name)


def test_RepVGG_B3():
    name = "RepVGG-B3"
    check_schema(name)


def test_resnet18():
    name = "resnet18"
    check_schema(name)


def test_resnet34():
    name = "resnet34"
    check_schema(name)


def test_resnet50():
    name = "resnet50"
    check_schema(name)


def test_resnet50():
    name = "resnet50"
    check_schema(name)


def test_vgg11_bn():
    name = "vgg11_bn"
    check_schema(name)
