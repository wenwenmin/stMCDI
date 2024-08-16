import pickle
import yaml
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
# from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
import anndata
import scanpy as sc
warnings.filterwarnings("ignore")

def process_func(datapath: str, locationpath: str, aug_rate=1, missing_ratio=0.1):
    adata = sc.read(datapath)
    data = adata.X
    if hasattr(data, 'toarray'):
        data = data.toarray()
    data = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    location = pd.read_csv(locationpath, header=None, sep='\t')
    # print(data.head())


    # data.replace("?", np.nan, inplace=True)
    data = data.astype(float)
    location = location.astype(float)


    location_standardized = (location - location.mean(axis=0)) / location.std(axis=0)
    data_standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    data_aug = pd.concat([data_standardized] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.copy()
    # for each column, mask `missing_ratio` % of observed values.
    for col in range(observed_values.shape[1]):  # col #
        obs_indices = np.where(masks[:, col])[0]
        miss_indices = np.random.choice(
            obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices, col] = False

    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)


    observed_masks = observed_masks.astype(int)  # "float32"
    gt_masks = gt_masks.astype("float32")
    if observed_values.shape[1] == 0:
        raise ValueError("The sequence length (L) is 0. Please check the input data.")

    return observed_values, observed_masks, gt_masks, location_standardized


class st_dataset(Dataset):
    def __init__(
        self,datapath, coordinate, eval_length=1000, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0
    ):
        np.random.seed(seed)

        dataset_path = datapath
        location_path = coordinate
        self.observed_values, self.observed_masks, self.gt_masks, self.location = process_func(
            dataset_path, location_path, aug_rate=aug_rate, missing_ratio=missing_ratio
        )
        self.eval_length = self.observed_values.shape[1]
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]

        if index >= len(self.observed_values):
            raise IndexError(
                f"Index {index} is out of bounds for observed_values with size {len(self.observed_values)}")

        if index >= len(self.location):
            raise IndexError(f"Index {index} is out of bounds for location DataFrame.")

        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "location": self.location.iloc[index].values  # 访问特定样本的xy坐标
        }
        return s

    def __len__(self):
        return len(self.use_index_list)




def get_dataloader(datapath, coordinate, seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
    dataset = st_dataset(datapath, coordinate, missing_ratio=missing_ratio, seed=seed)
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * len(dataset) * tmp_ratio)
    end = (int)(nfold * len(dataset) * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)
    num_train = (int)(len(remain_index) * 1)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    # # Here we perform max-min normalization.
    # processed_data_path_norm = (
    #     f"./data/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    # )
    # if not os.path.isfile(processed_data_path_norm):
    #     # Data transformation after train-test split.
    #     col_num = dataset.observed_values.shape[1]
    #     max_arr = np.zeros(col_num)
    #     min_arr = np.zeros(col_num)
    #     mean_arr = np.zeros(col_num)
    #     for k in range(col_num):
    #         # Using observed_mask to avoid counting missing values (now represented as 0)
    #         obs_ind = dataset.observed_masks[train_index, k].astype(bool)
    #         temp = dataset.observed_values[train_index, k]
    #         max_arr[k] = max(temp[obs_ind])
    #         min_arr[k] = min(temp[obs_ind])
    #     # print(f"--------------Max-value for each column {max_arr}--------------")
    #     # print(f"--------------Min-value for each column {min_arr}--------------")
    #
    #     dataset.observed_values = (
    #         (dataset.observed_values - (min_arr - 1)) / (max_arr - min_arr + 1)
    #     ) * dataset.observed_masks


    # Create datasets and corresponding data loaders objects.
    train_dataset = st_dataset(
        datapath, coordinate ,use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    print("--------Training Dataset created--------")
    print(f"Training dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=0)

    valid_dataset = st_dataset(
        datapath, coordinate, use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    print("--------Validation Dataset created--------")
    print(f"Validation dataset size: {len(valid_dataset)}")
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = st_dataset(
        datapath, coordinate, use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    print("--------Test Dataset created--------")
    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)


    return train_loader, valid_loader, test_loader


