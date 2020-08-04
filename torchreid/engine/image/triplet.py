from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss, TripletLoss, CenterLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageTripletEngine(engine.Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torch
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """   

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, scheduler=None, use_gpu=True,
                 label_smooth=True):
        super(ImageTripletEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_c = CenterLoss(num_classes=751, feat_dim=2048)

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses_t1 = AverageMeter()
        losses_x1 = AverageMeter()
        losses_x2 = AverageMeter()
        accs1 = AverageMeter()
        accs2 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            output1, output2, fea = self.model(imgs)
            b = output1.size(0)
            loss_c1 = self._compute_loss(self.criterion_c, fea[0], pids[:b])
            loss_t1 = self._compute_loss(self.criterion_t, fea[0], pids[:b])
            loss_x1 = self._compute_loss(self.criterion_x, output1, pids[:b])
            loss_x2 = self._compute_loss(self.criterion_x, output2, pids[b:])
            loss1 = (self.weight_x * loss_x1 + self.weight_x * loss_x2) * 0.5
            loss2 = self.weight_t * loss_t1 + 0.0005 * loss_c1
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses_t1.update(loss_t1.item(), b)
            losses_x1.update(loss_x2.item(), b)
            losses_x2.update(loss_x2.item(), b)
          
            accs1.update(metrics.accuracy(output1, pids[:b])[0].item())
            accs2.update(metrics.accuracy(output2, pids[b:])[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_t1 {loss_t1.val:.4f} ({loss_t1.avg:.4f})\t'
                      'Loss_x1 {loss_x1.val:.4f} ({loss_x1.avg:.4f})\t'
                      'Loss_x2 {loss_x2.val:.4f} ({loss_x2.avg:.4f})\t'
                      'Acc1 {acc1.val:.2f} ({acc1.avg:.2f})\t'
                      'Acc2 {acc2.val:.2f} ({acc2.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss_t1=losses_t1,
                      loss_x1=losses_x1,
                      loss_x2=losses_x2,
                      acc1=accs1,
                      acc2=accs2,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                self.writer.add_scalar('Train/Loss_t1', losses_t1.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x1', losses_x1.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x2', losses_x2.avg, n_iter)
                self.writer.add_scalar('Train/Acc1', accs1.avg, n_iter)
                self.writer.add_scalar('Train/Acc2', accs2.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()