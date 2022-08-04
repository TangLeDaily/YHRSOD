from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import wandb
import argparse
import os
import random
from data import *
from torch import optim
from util import *

# model name
from rootmodel.R34_base import *
model = YHRSOD()

parser = argparse.ArgumentParser(description="PyTorch Data_Pre")
# wandb and project
parser.add_argument("--use_wandb", default=True, action="store_true")
parser.add_argument("--Project_name", default="YHRSOD_V1", type=str) # wandb Project name
parser.add_argument("--This_name", default="YHRSOD", type=str) # wandb run name & model save name path
parser.add_argument("--wandb_username", default="karledom", type=str)
# dataset 文件夹要以/结尾
parser.add_argument("--train_RGB_root", default='datasets/train/DUTS-TR/DUTS-TR-Image/', type=str)
parser.add_argument("--train_gt_root", default='datasets/train/DUTS-TR/DUTS-TR-Mask/', type=str)
parser.add_argument("--test_RGB_root", default='datasets/test/DUTS-TE/DUTS-TE-Image/', type=str)
parser.add_argument("--test_gt_root", default='datasets/test/DUTS-TE/DUTS-TE-Mask/', type=str)
parser.add_argument("--lowsize", default=256, type=int)
parser.add_argument("--highsize", default=512, type=int)
# train setting
parser.add_argument("--cuda", default=True, action="store_true")
parser.add_argument("--cuda_id", default=1, type=int)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=10000, type=int)
parser.add_argument("--batchSize", default=8, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--threads", default=8, type=int)
# other setting
parser.add_argument("--dataset_pin_memory", default=True, action="store_true")
parser.add_argument("--dataset_drop_last", default=True, action="store_true")
parser.add_argument("--dataset_shuffle", default=True, action="store_true")
parser.add_argument("--test_save_epoch", default=10, type=int)
parser.add_argument("--decay_loss_epoch", default=50, type=int)
parser.add_argument("--decay_loss_ratio", default=0.8, type=float)
opt = parser.parse_args()



def main():
    global model, opt
    if opt.use_wandb:
        wandb.init(project=opt.Project_name, name=opt.This_name, entity=opt.wandb_username)
    print(opt)

    print("===> Find Cuda")
    cuda = opt.cuda
    if cuda:
        torch.cuda.set_device(opt.cuda_id)
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    # 利用显存换取浮点训练加速
    # cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = HRdataset(opt.train_RGB_root, opt.train_gt_root, opt.lowsize, opt.highsize)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, num_workers=opt.threads,
                              pin_memory=opt.dataset_pin_memory, drop_last=opt.dataset_drop_last, shuffle=opt.dataset_shuffle)
    test_set = HRdataset(opt.test_RGB_root, opt.test_gt_root, opt.lowsize, opt.highsize)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1)

    print("===> Setting loss")
    criterion = torch.nn.BCEWithLogitsLoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    print("===> Do Resume Or Skip")
    # model = get_yu(model, "checkpoints/over/TSALSTM_ATD/model_epoch_212_psnr_27.3702.pth")

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.max_epoch + 1):
        # 训练
        train(optimizer, model, criterion, epoch, train_loader)
        # 测试和保存
        if (epoch+1) % opt.test_save_epoch == 0:
            save_fm, save_sm, save_mae = test(model, epoch, test_loader, optimizer.param_groups[0]["lr"])
            save_checkpoint(model, epoch, optimizer.param_groups[0]["lr"], save_fm, save_sm, save_mae)
        # 降低学习率
        if (epoch+1) % opt.decay_loss_epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] *= opt.decay_loss_ratio

def train(optimizer, model, criterion, epoch, train_loader):
    global opt
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    avg_loss = AverageMeter()
    if opt.cuda:
        model = model.cuda()
    for iteration, batch in enumerate(train_loader):
        low, high, gt = batch
        if opt.cuda:
            low = low.cuda()
            high = high.cuda()
            gt = gt.cuda()
        out = model(low, high)
        loss = criterion(out, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())
        if iteration % 50 == 0:
            if opt.use_wandb:
                wandb.log({'epoch': epoch, 'iter_loss': avg_loss.avg})
            print('epoch_iter_{}_loss is {:.10f}'.format(iteration, avg_loss.avg))


def test(model, epoch, test_loader, lr):
    global opt
    print(" -- Start eval --")
    model.eval()
    if not os.path.exists("checkpoints/{}/".format(opt.This_name)):
        os.makedirs("checkpoints/{}/".format(opt.This_name))
    log_write("checkpoints/{}/Test_log.txt".format(opt.This_name), "===> Epoch_{}:".format(epoch))
    save_mae = AverageMeter()
    save_sm = AverageMeter()
    save_fm = AverageMeter()
    with torch.no_grad():
        for iteration, batch in enumerate(test_loader):
            low, high, gt = batch
            if opt.cuda:
                low = low.cuda()
                high = high.cuda()
                gt = gt.cuda()
            out = model(low, high)
            out = out.sigmoid()
            save_fm.update(Fm(out, gt))
            save_sm.update(Sm(out, gt))
            save_mae.update(MAE(out, gt))
    if opt.use_wandb:
        wandb.log({'MAE': save_mae.avg,
                   'Sm': save_sm.avg,
                   'Fm': save_fm.avg,
                   'Epoch': epoch})
        print("===> Epoch:{} lr:{:.8f} Fm:{:.4f} Sm:{:.4f} MAE:{:.4f}".format(epoch, lr, save_fm.avg, save_sm.avg, save_mae.avg))
        log_write("checkpoints/{}/Test_log.txt".format(opt.This_name),
                  "Epoch:{} -- lr:{:.8f} -- Fm:{:.4f} Sm:{:.4f} MAE:{:.4f}".format(epoch, lr, save_fm.avg, save_sm.avg, save_mae.avg))
    return save_fm.avg, save_sm.avg, save_mae.avg

def save_checkpoint(model, epoch, lr, save_fm, save_sm, save_mae):
    global opt
    model_folder = "checkpoints/{}/".format(opt.This_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = model_folder + "EP_{}_LR_{:.8f}_Fm_{:.4f}_Sm_{:.4f}_MAE_{:.4f}.pth".format(epoch, lr, save_fm, save_sm, save_mae)
    torch.save({'model': model.state_dict()}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()