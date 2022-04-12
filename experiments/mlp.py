from pytorch_lightning import Trainer

from meTCRs.models.architectures.mlp import Mlp
from meTCRs.dataloader.VDJdb_data_module import VDJdbDataModule


def main():
    data = VDJdbDataModule('../data/VDJdb-2022-02-25 12_58_09.77.tsv', batch_size=32, classes_per_batch=16)
    data.setup()

    model = Mlp(data.dimension, 128, 64)
    trainer = Trainer(max_epochs=1)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()

