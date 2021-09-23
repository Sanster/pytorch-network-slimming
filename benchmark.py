import os
import time
from pprint import pprint
from typing import List

import psutil
import numpy as np
import onnxruntime as ort
import typer


def run_one(onnx_model, loop, input_shape):
    model = ort.InferenceSession(onnx_model)
    x = np.ones(input_shape, dtype=np.float32)

    warm_up_count = 6
    output_names = ["output"]
    onnx_input = {"input": x}
    for _ in range(warm_up_count):
        model.run(output_names, onnx_input)

    process = psutil.Process(os.getpid())
    mb_memory = process.memory_info().rss / 1024 / 1024

    times = []
    for _ in range(loop):
        start = time.time()
        model.run(output_names, onnx_input)
        times.append((time.time() - start) * 1000)

    result = {
        "mb_memory": mb_memory,
        "ms_min": np.min(times),
        "ms_max": np.max(times),
        "ms_avg": np.average(times),
    }
    return result


def main(onnx_models: List[str], loop: int = 30, input_shape: str = "1x3x224x224"):
    input_shape = [int(it) for it in input_shape.split("x")]
    print(f"Input shape: {input_shape}")

    for onnx_model in onnx_models:
        data = run_one(onnx_model, loop, input_shape)
        print(onnx_model)
        pprint(data)


if __name__ == "__main__":
    typer.run(main)
