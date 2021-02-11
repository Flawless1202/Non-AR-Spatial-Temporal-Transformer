import argparse
import os
import shutil

from mmcv import Config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from nast import LightningModel
from nast.utils import setup_seed, partial_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test a detector.")
    parser.add_argument("config", help="Train config file path.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_seed(cfg.random_seed)

    model = LightningModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        filepath=f"{cfg.checkpoint_path}/{cfg.name}/{cfg.version}/"
                 f"{cfg.name}_{cfg.version}_{{epoch}}_{{avg_val_loss:.3f}}_{{ade:.3f}}_{{fde:.3f}}_{{fiou:.3f}}",
        save_last=True,
        save_top_k=8,
        verbose=True,
        monitor='fiou',
        mode='max',
        prefix=''
    )

    lr_logger_callback = LearningRateLogger(logging_interval='step')

    logger = TensorBoardLogger(save_dir=cfg.log_path, name=cfg.name, version=cfg.version)
    logger.log_hyperparams(model.hparams)

    profiler = SimpleProfiler() if cfg.simple_profiler else AdvancedProfiler()
    check_val_every_n_epoch = cfg.check_val_every_n_epoch if hasattr(cfg, 'check_val_every_n_epoch') else 1

    trainer = pl.Trainer(
        gpus=cfg.num_gpus,
        max_epochs=cfg.max_epochs,
        logger=logger,
        profiler=profiler, # this line won't work in multi-gpu setting.
        weights_summary="top",
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=[lr_logger_callback],
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        accumulate_grad_batches=cfg.batch_size_times,
        check_val_every_n_epoch=check_val_every_n_epoch)

    if (not (args.train or args.test)) or args.train:
        shutil.copy(args.config, os.path.join(cfg.log_path, cfg.name, cfg.version, args.config.split('/')[-1]))

        if cfg.load_from_checkpoint is not None:
            model_ckpt = partial_state_dict(model, cfg.load_from_checkpoint)
            model.load_state_dict(model_ckpt)
        trainer.fit(model)

    if args.test:
        if cfg.test_checkpoint is not None:
            model_ckpt = partial_state_dict(model, cfg.test_checkpoint)
            model.load_state_dict(model_ckpt)
        trainer.test(model)


if __name__ == '__main__':
    main()
