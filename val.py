from rivagan.dataloader import load_train_val
from rivagan import RivaGAN
from rivagan.utils import mjpeg, psnr, ssim
import gc
import cv2
import torch
from tqdm import tqdm
import numpy as np
import pdb
from rivagan.dense import DenseDecoder, DenseEncoder
from rivagan.noise import Compression, Crop, Scale
from rivagan.utils import mjpeg, psnr, ssim

def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()

class RivaGANval(object):
    def __init__(self):
        self.data_dim = 32
    def val(self):
        seq_len = 1
        batch_size = 12
        dataset = './data/hollywood2'
        train, val = load_train_val(seq_len, batch_size, dataset)
        model = RivaGAN.load('./model.pt')
        model.encoder.eval()
        model.decoder.eval()
        iterator = tqdm(val, ncols=0)
        crop = Crop()
        scale = Scale()
        compress = Compression()
        metrics = {
            "train.loss": [],
            "train.raw_acc": [],
            "train.mjpeg_acc": [],
            "train.adv_loss": [],
            "val.ssim": [],
            "val.psnr": [],
            "val.crop_acc": [],
            "val.scale_acc": [],
            "val.mjpeg_acc": [],
        }
        epoch = 1
        with torch.no_grad():
            for frames in iterator:
                frames = frames.cuda()
                data = torch.zeros((frames.size(0), self.data_dim)).random_(0, 2).cuda()

                wm_frames = model.encoder(frames, data)
                wm_crop_data = model.decoder(mjpeg(crop(wm_frames)))
                wm_scale_data = model.decoder(mjpeg(scale(wm_frames)))
                wm_mjpeg_data = model.decoder(mjpeg(wm_frames))

                metrics["val.ssim"].append(
                    ssim(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                metrics["val.psnr"].append(
                    psnr(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())


                metrics["val.crop_acc"].append(get_acc(data, wm_crop_data))
                metrics["val.scale_acc"].append(get_acc(data, wm_scale_data))
                metrics["val.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))

                iterator.set_description(
                    "%s | SSIM %.3f | PSNR %.3f | Crop %.3f | Scale %.3f | MJPEG %.3f" % (
                        epoch,
                        np.mean(metrics["val.ssim"]),
                        np.mean(metrics["val.psnr"]),
                        np.mean(metrics["val.crop_acc"]),
                        np.mean(metrics["val.scale_acc"]),
                        np.mean(metrics["val.mjpeg_acc"]),))

if __name__ == '__main__':
    val = RivaGANval()
    val.val()
