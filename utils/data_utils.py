from torch_geometric.datasets import Amazon, Coauthor, WebKB, WikipediaNetwork, WikiCS, HeterophilousGraphDataset
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents, Compose
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(root: str, data_name: str, num_splits=10):
    if data_name in ["Texas", "Wisconsin", "Cornell"]:
        dataset = WebKB(root, name=data_name, transform=Compose([LargestConnectedComponents(),
                                                                 get_split(num_splits=num_splits, num_val=0.25, num_test=0.25)]))
    elif data_name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root, name=data_name, transform=get_split(num_splits=num_splits, num_val=0.25, num_test=0.25))
    elif data_name in ["Amazon-ratings", "Roman-empire"]:
        dataset = HeterophilousGraphDataset(root, data_name, transform=get_split(num_splits=num_splits, num_val=0.25, num_test=0.25))
    elif data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name, transform=get_split(num_splits=num_splits, num_val=0.2, num_test=0.2))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name, transform=get_split(num_splits=num_splits, num_val=0.2, num_test=0.2))
    elif data_name == "WikiCS":
        dataset = WikiCS(os.path.join(root, data_name), transform=get_split(num_splits=num_splits, num_val=0.2, num_test=0.2))
    else:
        raise NotImplementedError
    return dataset


def get_split(num_splits, num_val=0.2, num_test=0.2):
    return RandomNodeSplit(num_splits=num_splits, num_val=num_val, num_test=num_test)