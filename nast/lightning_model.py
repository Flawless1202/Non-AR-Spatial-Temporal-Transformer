import time

from torch.utils.data import DataLoader
from mmcv.utils import Config
import torch.optim as optim
import pytorch_lightning as pl

from .models import build_predictor
from .datasets import build_dataset
from .datasets.utils.metrics import calculate_metrics, summarize_metrics
from .datasets.utils.collate import collate
from .datasets.pipelines.box_transforms import *


class LightningModel(pl.LightningModule):

    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.train_cfg = cfg.train_cfg
        self.val_cfg = cfg.val_cfg
        self.optim_cfg = cfg.optimizer_cfg
        self.lr_cfg = cfg.lr_cfg

        self.model = build_predictor(self.model_cfg)
        # self.example_input_array = None

        self.hparams = dict(lr=self.optim_cfg.lr,
                            batch_size=self.data_cfg.batch_size * cfg.batch_size_times,
                            **self.train_cfg)

    def forward(self, data, **kwargs):
        pred = self.model(data, **kwargs)
        return pred

    def training_step(self, batch, batch_idx):
        data, gold = batch
        current = data['hist']['boxes'][:, -1:]
        pred = self(data, **self.train_cfg.forward)
        loss = self.model.loss(pred, current, gold['boxes'], gold['masks'], **self.train_cfg.loss)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, gold = batch

        current = data['hist']['boxes'][:, -1:]
        pred = self(data, **self.val_cfg.forward)
        loss = self.model.loss(pred, current, gold['boxes'], gold['masks'], **self.val_cfg.loss)

        pred_boxes = self.model.predict(data, **self.val_cfg.predict)
        gold_boxes, gold_masks = self.model.get_target_boxes(gold, **self.val_cfg.target)

        logs = calculate_metrics(pred_boxes, gold_boxes, gold_masks)
        logs.update(val_loss=loss)

        return logs

    def validation_epoch_end(self, outputs):
        logs = summarize_metrics(outputs)
        print(logs)
        return {'log': logs}

    def test_step(self, batch, batch_idx):
        data, gold = batch

        now = time.time()
        pred_boxes = self.model.predict(data, **self.val_cfg.predict)

        elapsed = time.time() - now
        print(f"elapsed time: {elapsed}")

        gold_boxes, gold_masks = self.model.get_target_boxes(gold, **self.val_cfg.target)
        logs = calculate_metrics(pred_boxes, gold_boxes, gold_masks)

        return logs

    def test_epoch_end(self, outputs):
        logs = summarize_metrics(outputs)
        print(logs)
        return {'log': logs}

    def prepare_data(self):
        self.train_dataset = build_dataset(self.data_cfg.train)
        self.val_dataset = build_dataset(self.data_cfg.val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.data_cfg.batch_size,
                          shuffle=True,
                          num_workers=self.data_cfg.num_workers,
                          collate_fn=collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=64,
                          shuffle=False,
                          num_workers=self.data_cfg.num_workers,
                          collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4,
                          collate_fn=collate)

    def configure_optimizers(self):
        optim_cfg = self.optim_cfg.copy()
        optim_class = getattr(optim, optim_cfg.pop("type"))
        optimizer = optim_class(self.parameters(), **optim_cfg)

        lr_cfg = self.lr_cfg.copy()
        lr_sheduler_class = getattr(optim.lr_scheduler, lr_cfg.pop("type"))
        scheduler = {
            "scheduler": lr_sheduler_class(optimizer, **lr_cfg),
            "monitor": 'avg_val_loss',
            "interval": "epoch",
            "frequency": 1
        }

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        warm_up_type, warm_up_step = self.cfg.warm_up_cfg.type, self.cfg.warm_up_cfg.step_size
        if warm_up_type == 'Exponential':
            lr_scale = self.model_cfg.encoder.d_model ** -0.5
            lr_scale *= min((self.trainer.global_step + 1) ** (-0.5),
                            (self.trainer.global_step + 1) * warm_up_step ** (-1.5))
        elif warm_up_type == "Linear":
            lr_scale = min(1., float(self.trainer.global_step + 1) / warm_up_step)
        else:
            raise NotImplementedError

        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.hparams.lr

        optimizer.step()
        optimizer.zero_grad()
