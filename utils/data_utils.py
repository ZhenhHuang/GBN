import torch
from torch_geometric.datasets import Amazon, Coauthor, WebKB, Planetoid, WikipediaNetwork, GitHub, WikiCS, HeterophilousGraphDataset
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents, Compose
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(root: str, data_name: str,
              num_val=0.1, num_test=0.2, num_per_class=None):
    if num_per_class is None:
        transform = RandomNodeSplit(num_val=num_val, num_test=num_test)
    else:
        transform = RandomNodeSplit(num_val=num_val, num_test=num_test, split="test_rest", num_train_per_class=num_per_class)

    if data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name, transform=transform)
    elif data_name in ["Texas", "Wisconsin", "Cornell"]:
        dataset = WebKB(root, name=data_name,
                        transform=Compose([LargestConnectedComponents()
                                        ,RandomNodeSplit(num_val=0.25, num_test=0.25)]))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name, transform=transform)
    elif data_name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root, name=data_name, transform=transform)
    elif data_name in ["Amazon-ratings", "Roman-empire"]:
        dataset = HeterophilousGraphDataset(root, data_name, transform)
    elif data_name in ['Cora', 'Citeseer', 'PubMed']:
        if num_per_class is None:
            num_per_class = 20
            split = "public"
        else:
            split = "random"
        dataset = Planetoid(root, name=data_name, split=split, num_train_per_class=num_per_class)
    # elif data_name == 'ogbn-arxiv':
    #     dataset = PygNodePropPredDataset(name=data_name, root=root,
    #                                      transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name == 'GitHub':
        dataset = GitHub(os.path.join(root, "GitHub"), transform=transform)
    elif data_name == "WikiCS":
        dataset = WikiCS(os.path.join(root, data_name), transform=transform)
    else:
        raise NotImplementedError
    return dataset