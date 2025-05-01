import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.data_utils import load_data
from utils.train_utils import EarlyStopping
from logger import create_logger
import os
from torch_geometric.loader import DataLoader
from modules.models import BoundaryGCN
from torch_geometric.utils import degree


class GraphTransferExp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger = create_logger(configs.log_path)

    def load_model(self, distance):
        nc_model = BoundaryGCN(n_layers=self.configs.additional_layers + distance,
                       in_dim=1, hid_dim=self.configs.hid_dim,
                               embed_dim=self.configs.embed_dim, out_dim=1,
                               bias=self.configs.bias, act=self.configs.act, input_act=self.configs.input_act,
                               drop=self.configs.dropout, norm=self.configs.norm,
                               add_self_loop=self.configs.add_self_loop).to(self.device)
        return nc_model

    def load_data(self, split='train', distance=50):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset,
                            split=split, distance=distance)
        loader = DataLoader(dataset, batch_size=self.configs.batch_size)
        return dataset, loader

    def train(self):
        total_mse = {}
        self.logger.info("--------------------------Training Start-------------------------")
        for dist in self.configs.distance_list:
            train_set, train_loader = self.load_data('train', dist)
            val_set, val_loader = self.load_data('val', dist)
            test_set, test_loader = self.load_data('test', dist)
            model = self.load_model(dist)
            model.train()
            optimizer = Adam(model.parameters(), lr=self.configs.lr_trans,
                             weight_decay=self.configs.weight_decay_trans)
            early_stop = EarlyStopping(self.configs.patience_trans)
            for epoch in range(self.configs.epochs_trans):
                train_loss = 0
                for data in train_loader:
                    data.degree = degree(data.edge_index[0], data.num_nodes)
                    data = data.to(self.device)
                    train_loss += self.train_step(model, data, optimizer)

                train_loss = train_loss / len(train_loader)

                self.logger.info(f"Epoch {epoch}: train_loss={train_loss}")

                if epoch % self.configs.val_every == 0:
                    val_loss = self.val(model, val_loader)
                    self.logger.info(f"Epoch {epoch}: val_mse={val_loss}")
                    early_stop(val_loss, model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        print("---------Early stopping--------")
                        break
            test_mse = self.test(model, test_loader, distance=dist)
            self.logger.info(f"test_mse={test_mse}")
            total_mse[dist] = test_mse

    def val(self, model, val_loader):
        loss = 0
        for data in val_loader:
            loss += self.test_step(model, data)
        loss = loss / len(val_loader)
        model.train()
        return loss

    def test(self, model, test_loader, distance):
        test_loader = self.load_data('test', distance) if test_loader is None else test_loader
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        model.load_state_dict(torch.load(path))

        loss = 0
        for data in test_loader:
            data.degree = degree(data.edge_index[0], data.num_nodes)
            data = data.to(self.device)
            loss += self.test_step(model, data)
        loss = loss / len(test_loader)
        self.logger.info(f"test_mse={loss}%")
        return loss

    @staticmethod
    def train_step(model, data, optimizer):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def test_step(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            loss = F.mse_loss(out, data.y)
        return loss.item()