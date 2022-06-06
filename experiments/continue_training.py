import os
import sys
import argparse
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(sys.path[0], '..'))

from meTCRs.models.embeddings.mlp import Mlp
from meTCRs.dataloader.data_module import DataModule
from meTCRs.models.distances.euclidean import Euclidean
from meTCRs.models.losses.contrastive_loss import ContrastiveLoss

parser = argparse.ArgumentParser(description="Continue training at a given checkpoint")
parser.add_argument("--model", type=str, default="mlp")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--hparams", type=str)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--encoding", type=str, default="one_hot")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--out", type=str)


def load_data(args):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', args.data)
    data_module = DataModule(data_path, batch_size=args.batch_size, encoding=args.encoding)
    data_module.setup(debug=args.debug, seed=args.seed)
    return data_module


def get_loss(params):
    distance = Euclidean()
    loss = ContrastiveLoss(distance, alpha=params['alpha'], reduction=params['reduction'])
    return loss


def load_hparams(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parser.parse_args()

    data = load_data(args)
    hparams = load_hparams(args.hparams)
    loss = get_loss(hparams['loss'])

    if args.model == "mlp":
        model = Mlp.load_from_checkpoint(args.checkpoint, hparams_file=args.hparams, loss=loss)
    else:
        raise NotImplementedError("No model of type {} implemented".format(args.model))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, every_n_epochs=1)

    trainer = Trainer(max_epochs=args.max_epochs, default_root_dir=args.out, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=data)
