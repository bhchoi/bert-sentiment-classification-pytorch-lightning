import argparse
from typing import Tuple
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import random_split

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl

from preprocess import Preprocessor
from dataset import TextDataset
from model import BertModel


def get_dataloader(args, preprocessor):

    train_dataset = TextDataset(args.train_data, preprocessor, args.max_len)
    test_dataset = TextDataset(args.test_data, preprocessor, args.max_len)

    if not args.val_data:
        train_size = int(len(train_dataset) * 0.8)
        validation_size = len(train_dataset) - train_size
        train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    else:
        validation_dataset = TextDataset(args.val_data, preprocessor, args.max_len)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    return train_dataloader, validation_dataloader, test_dataloader


def main(args):
    preprocessor = Preprocessor(args.model_type, args.max_len)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args, preprocessor)
    bert_finetuner = BertModel(args, train_dataloader, val_dataloader, test_dataloader)

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        version=1,
        name="nsmc-bert"
    )

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=args.checkpoint_path,
        verbose=True,
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        prefix=''
    )

    trainer = pl.Trainer(
        gpus=1,
        # distributed_backend='ddp'
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger
    )

    trainer.fit(bert_finetuner)

    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--train_data", type=str, default="data/ratings_train.txt")
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--test_data", type=str, default="data/ratings_test.txt")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--checkpoint_path", type=str, default='checkpoint')
    args = parser.parse_args()
    main(args)