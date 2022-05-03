import torch
from data_ver3 import DataModule
from model_ver2 import LitResnet
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

AVAIL_GPUS = min(1, torch.cuda.device_count())

def main(iter=2):
    
    img_data = DataModule()
    img_model = LitResnet(lr=0.05)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models", 
        filename = "best-checkpoint",
        monitor="val_loss", 
        mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=10,
        # accelerator="cpu",
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("lightning_logs/", name="resnet"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback, 
            early_stopping_callback
        ],
    )

    trainer.fit(img_model, img_data)
    trainer.test(img_model, datamodule=img_data)
    print('Finished Training')
if __name__ == "__main__":
    main()
