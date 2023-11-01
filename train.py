import os
from torch import optim, nn
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
import configargparse
import torchmetrics
from torchmetrics import ROC

from dataset import CatDogDataset

# define the LightningModule
class Classifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.image_output_path = args.image_output_path
        os.makedirs(self.image_output_path, exist_ok=True)

        self.model = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.model.classifier[3] = nn.Linear(1024, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.ROC = ROC(task='binary')
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        prediction = self.softmax(x_hat)[:, 1]
        self.accuracy(prediction, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        prediction = self.softmax(x_hat)[:, 1]
        self.ROC(prediction, y)
        conf_mat = self.confusion_matrix(prediction, y)
        accuracy = (conf_mat[0][0]+conf_mat[1][1]) / (conf_mat[0][0]+conf_mat[0][1]+conf_mat[1][0]+conf_mat[1][1])
        precision = conf_mat[1][1] / (conf_mat[1][1]+conf_mat[0][1])
        recall = conf_mat[1][1] / (conf_mat[1][1]+conf_mat[1][0])
        self.log("val_loss", loss)
        self.log("val_acc", accuracy, on_epoch=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        fig, ax = self.ROC.plot()
        fig.savefig(os.path.join(self.image_output_path, f'{self.current_epoch}_ROC.png'))
        fig1, ax1 = self.confusion_matrix.plot()
        fig1.savefig(os.path.join(self.image_output_path, f'{self.current_epoch}_confusion_matrix.png'))
        self.ROC.reset()
        self.confusion_matrix.reset()
        self.accuracy.reset()
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",  # or 'step
                "frequency": 1,
            },
        }
    )


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--exp-name", type=str, default="exp", help="experiment name")
    parser.add_argument('--accelerator', choices=['auto', 'gpu', 'cpu'], default='auto', help='accelerator to use (default: auto)')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training. ')
    parser.add_argument('--dataset-path', type=str, default='data', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate step gamma (default: 0.7)")
    parser.add_argument('--max-epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--image_output_path', type=str, default='images', help='path to save images')

    parser.add_argument('--eval', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--ckpt-path', type=str, default='./exp/', help='path to the folder of checkpoint or direct path to ckpt file')

    args = parser.parse_args()

    # set seed for reproducibility
    seed_everything(args.seed)

    # init the autoencoder
    autoencoder = Classifier(args)
    # setup data
    dataset = CatDogDataset(args.dataset_path, mode='train')
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # trainer for the model
    checkpoint_callback = ModelCheckpoint(
        f"{args.exp_name}/",
        monitor="val_acc",
        mode="max",
        # save_top_k=1,
        filename="{epoch:02d}-{val_acc:.2f}",
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
    )

    if not args.eval:
        trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.validate(model=autoencoder, dataloaders=val_loader, ckpt_path="best")
    else:
        if os.path.isdir(args.ckpt_path):
            trainer.validate(model=autoencoder, dataloaders=val_loader, ckpt_path=os.path.join(args.ckpt_path, os.listdir(args.ckpt_path)[0]))
        else:
            trainer.validate(model=autoencoder, dataloaders=val_loader, ckpt_path=args.ckpt_path)