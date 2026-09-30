import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        
        # Make sure parent directory exists before creating self.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 修复TensorBoard在Windows上包含中文字符路径的兼容性问题
        # 很多时候FailedPreconditionError是因为路径太深或者包含特殊字符导致tf.io无法创建
        # 我们使用相对短路径，或者直接使用os创建好文件夹，让tf直接写入
        import string
        import random
        # Create a safe ASCII logdir for TensorBoard at the root of the project
        safe_log_dir = os.path.join("tb_logs", os.path.basename(log_dir), "loss_" + str(time_str))
        
        # MUST explicitly create the directory first before passing to SummaryWriter
        # otherwise tensorflow's internal gfile might fail on Windows
        
        # 很多时候FailedPreconditionError是因为路径是相对路径，我们把它转成绝对路径，并且避开中文
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 但是 current_dir 包含 "003-论文" 这种中文，所以我们要尽量把它放进纯英文路径，比如系统临时目录
        import tempfile
        temp_dir = tempfile.gettempdir()
        safe_log_dir = os.path.join(temp_dir, "tb_logs_litchi", os.path.basename(log_dir), "loss_" + str(time_str))
        
        os.makedirs(safe_log_dir, exist_ok=True)
        # also create intermediate dirs just to be safe
        os.makedirs(os.path.join(temp_dir, "tb_logs_litchi", os.path.basename(log_dir)), exist_ok=True)
        
        try:
            self.writer     = SummaryWriter(log_dir=safe_log_dir)
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Warning: Tensorboard initialization failed: {e}")
            self.writer = None
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        if getattr(self, 'writer', None) is not None:
            try:
                self.writer.add_scalar('loss', loss, epoch)
                self.writer.add_scalar('val_loss', val_loss, epoch)
            except:
                pass
            
        if epoch % 10 == 0:
            self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
