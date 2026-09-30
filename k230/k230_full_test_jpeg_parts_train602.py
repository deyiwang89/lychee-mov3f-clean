import gc
import math
import struct
import sys
import time

import image
import nncase_runtime as nn
import ulab.numpy as np
from libs.AI2D import Ai2d
from libs.AIBase import AIBase


LABELS = [
    "Anthrax_Leaf", "Bituminous_Leaf", "Curl_Leaf", "Deficiency_Leaf",
    "Dry_Leaf", "Felt_Leaf", "Fungal_Leaf_Spot", "Healthy_Leaf",
    "Leaf_Gall", "Leaf_Blight",
]
MODEL_PATH = globals().get("MODEL_PATH_OVERRIDE", "/sdcard/mobilenetv3_fractal_seed43_noclip_w8_a16.kmodel")
TEMP_PATH = "/sdcard/k230_current_test.jpg"
PART_COUNT = 11


class StaticClassifier(AIBase):
    def __init__(self):
        super().__init__(MODEL_PATH, [224, 224], [224, 224], 0)
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.RGB_packed, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([1, 224, 224, 3], [1, 3, 224, 224])


def softmax(values):
    values = values.flatten().tolist() if isinstance(values, np.ndarray) else list(values)
    if values and isinstance(values[0], list):
        values = values[0]
    peak = max(values)
    exps = [math.exp(float(value) - float(peak)) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def main():
    print("K230_JPEG_PARTS_BEGIN")
    app = None
    success = 0
    global_index = 0
    try:
        app = StaticClassifier()
        for part_index in range(PART_COUNT):
            path = "/sdcard/seed43_jpeg_part_%02d.bin" % part_index
            with open(path, "rb") as data_file:
                while True:
                    header = data_file.read(4)
                    if not header:
                        break
                    if len(header) != 4:
                        raise RuntimeError("short length header")
                    length = struct.unpack("<I", header)[0]
                    payload = data_file.read(length)
                    if len(payload) != length:
                        raise RuntimeError("short JPEG record")
                    with open(TEMP_PATH, "wb") as temp_file:
                        temp_file.write(payload)
                    img = image.Image(TEMP_PATH)
                    rgb = img.to_rgb888()
                    input_np = rgb.to_numpy_ref().reshape((1, img.height(), img.width(), 3))
                    if global_index == 0:
                        app.run(input_np)
                    start = time.ticks_ms()
                    app.run(input_np)
                    elapsed = time.ticks_diff(time.ticks_ms(), start)
                    probs = softmax(app.results[0])
                    order = sorted(range(len(probs)), key=lambda item: probs[item], reverse=True)
                    print("K230_JPEG_ROW,%d,%s,%.6f,%d" % (global_index, LABELS[order[0]], probs[order[0]], elapsed))
                    success += 1
                    global_index += 1
                    del input_np, rgb, img, payload
                    if global_index % 50 == 0:
                        gc.collect()
        print("K230_JPEG_PARTS_END,%d" % success)
    except Exception as exc:
        print("K230_JPEG_PARTS_ERROR,%d,%s" % (global_index, exc))
        sys.print_exception(exc)
    finally:
        if app is not None:
            try:
                app.deinit()
            except Exception as exc:
                print("K230_JPEG_PARTS_DEINIT_ERROR,%s" % exc)
        gc.collect()


main()
