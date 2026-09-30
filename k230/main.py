from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from media.sensor import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import utime
import image
import gc
import sys
import math

# 自定义分类推理类，继承自AIBase
class DiseaseClassificationApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, labels, conf_threshold=0.5, rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        # 模型输入分辨率
        self.model_input_size = model_input_size
        self.labels = labels
        # sensor给到AI的图像分辨率
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        # 显示分辨率
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.conf_threshold = conf_threshold
        
        # Ai2d实例，用于实现模型预处理
        self.ai2d = Ai2d(debug_mode)
        # 设置Ai2d的输入输出格式和类型
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    # 配置预处理操作，Ai2d支持crop/shift/pad/resize/affine
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            # 这里直接将原图resize到模型要求的输入尺寸
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def softmax(self, z):
        # 将 z 转换为 list 处理，避免 ulab 中关于 shape 或 axis 的奇怪错误
        # 假设 AIBase 解包后的 z 是一个 float 列表或者 1D numpy array
        if isinstance(z, np.ndarray):
            z_list = z.tolist()
        else:
            z_list = list(z)
            
        # 如果是嵌套列表（2D，即 [1, num_classes]），取第一批次
        if len(z_list) > 0 and isinstance(z_list[0], list):
            z_list = z_list[0]
            
        # 手动计算 softmax
        max_val = max(z_list)
        exp_z = [math.exp(v - max_val) for v in z_list]
        sum_exp_z = sum(exp_z)
        
        return [v / sum_exp_z for v in exp_z]

    # 自定义当前任务的后处理
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            predictions = results[0]
            
            # scores 现在是一个标准的 Python float 列表
            scores = self.softmax(predictions)
                
            max_score = float(max(scores))
            # 找到最大值的索引
            max_id = scores.index(max_score)
            
            if max_score > self.conf_threshold:
                return max_id, max_score
            else:
                return -1, 0.0

    # 绘制结果到OSD
    def draw_result(self, pl, res, fps):
        pl.osd_img.clear()
        with ScopedTiming("display_draw", self.debug_mode > 0):
            ids, score = res
            if ids != -1:
                text = "result: %s score: %.4f" % (self.labels[ids], score)
            else:
                text = "result: Unknown"
            
            # 在OSD层绘制文本
            pl.osd_img.draw_string_advanced(10, 30, 30, text, color=(255, 0, 0))
            pl.osd_img.draw_string_advanced(10, 80, 30, "FPS: %.2f" % fps, color=(0, 255, 0))


if __name__ == "__main__":
    # 显示模式，可以选择"hdmi"、"lcd3_5"(3.5寸mipi屏)和"lcd2_4"(2.4寸mipi屏)
    display = "lcd3_5"
    
    if display == "hdmi":
        display_mode = 'hdmi'
        display_size = [1920, 1080]
        rgb888p_size = [1920, 1080]
    elif display == "lcd3_5":
        display_mode = 'st7701'
        display_size = [800, 480]
        # 优化1：降低摄像头输出分辨率，刚好满足屏幕显示，减少内存拷贝和缩放时间
        rgb888p_size = [800, 480]
    elif display == "lcd2_4":
        display_mode = 'st7701'
        display_size = [640, 480]
        # 优化1：降低摄像头输出分辨率，刚好满足屏幕显示
        rgb888p_size = [640, 480]

    # 模型路径
    kmodel_path = "/sdcard/best_cpu.kmodel"
    model_input_size = [224, 224]
    
    labels = ["Anthrax_Leaf", "Bituminous_Leaf", "Curl_Leaf", "Deficiency_Leaf", 
              "Dry_Leaf", "Felt_Leaf", "Fungal_Leaf_Spot", "Healthy_Leaf", "Leaf_Blight", "Leaf_Gall"]
    conf_threshold = 0.5

    # 初始化PipeLine
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create(Sensor(width=rgb888p_size[0], height=rgb888p_size[1]))  # 创建PipeLine实例

    # 初始化分类实例
    app = DiseaseClassificationApp(kmodel_path, model_input_size=model_input_size, labels=labels, 
                                   conf_threshold=conf_threshold, rgb888p_size=rgb888p_size, 
                                   display_size=display_size, debug_mode=0)
    app.config_preprocess()

    clock = time.clock()

    try:
        # 优化2：控制循环帧率（如果推理非常快，过高的刷新率反而会因为OSD绘制占用总线带宽导致画面卡顿）
        # 如果需要榨干性能，可以不设置sleep，但建议在某些场景下限制在30fps
        while True:
            clock.tick()
            
            img = pl.get_frame() # 获取当前帧数据
            res = app.run(img)   # 经过AI2D预处理并推理当前帧
            fps = clock.fps()    # 获取FPS
            
            app.draw_result(pl, res, fps) # 绘制结果到PipeLine的osd图像
            pl.show_image() # 显示当前的绘制结果
            gc.collect()
            
            # 优化3：如果不需要实时渲染每一帧，可以在这里加个极小的延时，让出 CPU 给底层中断
            # time.sleep_ms(1)

    except Exception as e:
        sys.print_exception(e)
    finally:
        app.deinit()
        pl.destroy()
