import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0
    train_samples   = 0

    val_loss        = 0
    val_accuracy    = 0
    val_samples     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        images, targets, fractal_feats = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                fractal_feats = fractal_feats.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播 (支持双输入)
            #----------------------#
            # 检查模型是否支持 fractal_features 输入
            if hasattr(model_train.module if hasattr(model_train, 'module') else model_train, 'fractal_dim'):
                outputs = model_train(images, fractal_features=fractal_feats)
            else:
                outputs = model_train(images)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                if hasattr(model_train.module if hasattr(model_train, 'module') else model_train, 'fractal_dim'):
                    outputs = model_train(images, fractal_features=fractal_feats)
                else:
                    outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_samples = targets.size(0)
        total_loss += loss_value.item() * batch_samples
        with torch.no_grad():
            total_accuracy += (torch.argmax(outputs, dim=-1) == targets).sum().item()
            train_samples += batch_samples

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / train_samples,
                                'accuracy'  : total_accuracy / train_samples,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, fractal_feats = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                fractal_feats = fractal_feats.cuda(local_rank)

            optimizer.zero_grad()

            if hasattr(model_train.module if hasattr(model_train, 'module') else model_train, 'fractal_dim'):
                outputs = model_train(images, fractal_features=fractal_feats)
            else:
                outputs = model_train(images)
            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            
            batch_samples = targets.size(0)
            correct = (torch.argmax(outputs, dim=-1) == targets).sum().item()
            val_loss       += loss_value.item() * batch_samples
            val_accuracy   += correct
            val_samples    += batch_samples
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / val_samples,
                                'accuracy'  : val_accuracy / val_samples, 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        try:
            print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
            mean_train_loss = total_loss / train_samples
            mean_val_loss = val_loss / val_samples
            print('Total Loss: %.3f || Val Loss: %.3f ' % (mean_train_loss, mean_val_loss))
            
            # Save logs
            if loss_history is not None:
                loss_history.append_loss(epoch + 1, mean_train_loss, mean_val_loss)
        except Exception as e:
            print("Error saving log:", str(e))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, mean_train_loss, mean_val_loss)))

        if len(loss_history.val_loss) <= 1 or mean_val_loss <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
