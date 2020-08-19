import pytorch_lightning as pl
from transformers import BertForSequenceClassification, BertConfig, AdamW
import torch
from sklearn.metrics import accuracy_score


class BertModel(pl.LightningModule):
    def __init__(self, args, nsmc_train_dataloader, nsmc_val_dataloader, nsmc_test_dataloader):
        super().__init__()
        self.args = args
        self.nsmc_train_dataloader = nsmc_train_dataloader
        self.nsmc_val_dataloader = nsmc_val_dataloader
        self.nsmc_test_dataloader = nsmc_test_dataloader

        self.config = BertConfig.from_pretrained(self.args.model_type, num_labels=2)
        self.model = BertForSequenceClassification.from_pretrained(self.args.model_type, config=self.config)


    def forward(self, input_ids, input_mask, labels):
        output = self.model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        return output

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, labels = batch
        loss = self(input_ids, attention_mask, labels)
        loss = loss[0]
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs[0]
        y_hat = outputs[1]

        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), labels.cpu())
        val_acc = torch.tensor(val_acc)
        loss = torch.tensor(loss)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_val_loss, 'val_acc': avg_val_acc}

        return {'val_loss': avg_val_loss, 'val_acc': avg_val_acc, 'progress_bar': tensorboard_logs, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, labels = batch
        loss = self(input_ids, attention_mask, None)

        a, y_hat = torch.max(loss, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), labels.cpu())

        return {'test_acc': torch.tensor(test_acc)}


    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.nsmc_train_dataloader

    def val_dataloader(self):
        return self.nsmc_val_dataloader

    def test_dataloader(self):
        return self.nsmc_test_dataloader
