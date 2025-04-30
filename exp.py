import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from utils.eval_utils import cal_accuracy, cal_F1, cal_AUC_AP
from utils.data_utils import load_data
from utils.train_utils import EarlyStopping, act_fn
from logger import create_logger
from utils.config import list2str
import time
import os

from modules.models import BoundaryGCN
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from torch_geometric.utils import degree


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger = create_logger(configs.log_path)

    def load_model(self, dataset):
        nc_model = BoundaryGCN(n_layers=self.configs.n_layers,
                       in_dim=dataset.num_features, hid_dim=self.configs.hid_dim,
                               embed_dim=self.configs.embed_dim, out_dim=dataset.num_classes,
                               bias=False, act=act_fn(self.configs.act),
                               drop=self.configs.dropout).to(self.device)
        return nc_model

    def load_data(self):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        data = dataset[0]
        data.degree = degree(data.edge_index[0], data.num_nodes)
        return dataset, data

    def train(self):
        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        self.logger.info("--------------------------Training Start-------------------------")
        for t in range(self.configs.exp_iters):
            dataset, data = self.load_data()
            nc_model = self.load_model(dataset)
            nc_model.train()
            optimizer = Adam(nc_model.parameters(), lr=self.configs.lr_nc,
                             weight_decay=self.configs.weight_decay_nc)
            early_stop = EarlyStopping(self.configs.patience_nc)
            for epoch in range(self.configs.epochs_nc):
                epoch_loss = []

                data = data.to(self.device)
                loss, pred, true = self.train_step(nc_model, data, optimizer)
                epoch_loss.append(loss)

                train_loss = np.mean(epoch_loss)
                train_acc = cal_accuracy(pred, true)

                self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

                if epoch % self.configs.val_every == 0:
                    val_loss, val_acc, val_weighted_f1, val_macro_f1 = self.val(nc_model, data)
                    self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, "
                                     f"val_acc={val_acc * 100: .2f}%,"
                                     f"val_weighted_f1={val_weighted_f1 * 100: .2f},"
                                     f"val_macro_f1={val_macro_f1 * 100: .2f}%")
                    early_stop(-val_acc, nc_model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        print("---------Early stopping--------")
                        break
            test_acc, weighted_f1, macro_f1 = self.test(nc_model, data)
            self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                             f"weighted_f1={weighted_f1 * 100: .2f},"
                             f"macro_f1={macro_f1 * 100: .2f}%")
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        self.logger.info(f"Best ACCs: {[round(acc * 100, 2) for acc in total_test_acc]}")
        self.logger.info(f"Evaluation Acc is {mean * 100: .2f}% \u00B1 {std * 100: .2f}%")
        mean, std = np.mean(total_test_weighted_f1), np.std(total_test_weighted_f1)
        self.logger.info(f"Evaluation weighted F1 is {mean * 100: .2f}% \u00B1 {std * 100: .2f}%")
        mean, std = np.mean(total_test_macro_f1), np.std(total_test_macro_f1)
        self.logger.info(f"Evaluation macro F1 is {mean * 100: .2f}% \u00B1 {std * 100: .2f}%")

    def val(self, nc_model, data):
        nc_model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = nc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data.val_mask)
        acc = cal_accuracy(pred, true)
        weighted_f1, macro_f1 = cal_F1(pred, true)
        nc_model.train()
        return loss, acc, weighted_f1, macro_f1

    def test(self, nc_model, data=None):
        data = self.load_data()[1] if data is None else data
        nc_model.eval()
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        nc_model.load_state_dict(torch.load(path))
        with torch.no_grad():
            data = data.to(self.device)
            out = nc_model(data)
            loss, pred, true = self.cal_loss(out, data.y, data.test_mask)
        test_acc = cal_accuracy(pred, true)
        weighted_f1, macro_f1 = cal_F1(pred, true)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                         f"weighted_f1={weighted_f1 * 100: .2f},"
                         f"macro_f1={macro_f1 * 100: .2f}%")
        return test_acc, weighted_f1, macro_f1

    def train_step(self, nc_model, data, optimizer):
        optimizer.zero_grad()
        out = nc_model(data)
        loss, pred, true = self.cal_loss(out, data.y, data.train_mask)
        loss.backward()
        optimizer.step()
        return loss.item(), pred, true

    @staticmethod
    def cal_loss(output, label, mask):
        out = output[mask]
        y = label[mask].reshape(-1)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=-1).detach().cpu().numpy()
        return loss, pred, y.detach().cpu().numpy()