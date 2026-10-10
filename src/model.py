import numpy as np
import torch
import torch.nn as nn
import itertools
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Union

ArrayLike = Union[np.ndarray, list, tuple]

PROTEIN_EMBEDDING_DIM = 5120
MORGAN_EMBEDDING_DIM = 1024
CC_EMBEDDING_DIM = 1280
CC_FULL_EMBEDDING_DIM = 3200
CHEMBERTA_EMBEDDING_DIM = 384
BACNET_INPUT_DIM = 7808



def cc_id_reference():
    # The processing is based on the fact that each of the 10 spaces (A1, A2, ..., B5) has 128 dimensions, and they are concatenated in order.
    alphas = ["A","B","C","D","E"]
    nums = [1,2,3,4,5]
    i = 0
    cc_idx = {}
    for alpha in alphas:
        for num in nums:
            alpha_num = alpha + str(num)
            range_tmp = list(range(i*128,(1+i)*128))
            tmp = {alpha_num: range_tmp}
            i += 1
            cc_idx = dict(cc_idx, **tmp)
    return cc_idx

    
    
class CustomDataset_Search_Drug(Dataset):
    def __init__(self, target, library, cc_type):
        self.target = target
        self.library = library
        self.cc_type = cc_type
        self.cc_idx = cc_id_reference()

        list_target_name = target.keys()
        first_step = True
        for key in library.keys():
            tmp = list(library[key].keys())
            if first_step:
                list_chemical_name = tmp
                first_step = False
            else:
                list_chemical_name = list(set(list_chemical_name) & set(tmp))

        self.links = list(itertools.product(list_target_name, list_chemical_name))
        print(f"# links pairs: {len(self.links)}")

    def __len__(self):
        return len(self.links)
    
    def __getitem__(self, idx):
        # Create the vector that combines the information of the protein and chemical
        protein_name, compound_id = self.links[idx]
        out_vec = torch.as_tensor(self.target[protein_name], dtype=torch.float32).flatten()
        if out_vec.numel() != PROTEIN_EMBEDDING_DIM:
            raise ValueError(
                f"Invalid ESM-2 dimension for {protein_name}: "
                f"expected {PROTEIN_EMBEDDING_DIM}, got {out_vec.numel()}"
            )
        if not torch.isfinite(out_vec).all():
            raise ValueError(f"Non-finite ESM-2 values found for {protein_name}")

        for key, tmp_vec in self.library.items():
            if key == "cc":
                cc_idx_tmp = [self.cc_idx[k] for k in self.cc_type]
                cc_idx_tmp = list(itertools.chain.from_iterable(cc_idx_tmp))
                cc_vector = torch.as_tensor(tmp_vec[compound_id], dtype=torch.float32).flatten()
                if cc_vector.numel() == CC_EMBEDDING_DIM:
                    cc_selected = cc_vector
                elif cc_vector.numel() == CC_FULL_EMBEDDING_DIM:
                    cc_selected = cc_vector[cc_idx_tmp]
                else:
                    raise ValueError(
                        f"Invalid Chemical Checker dimension for {compound_id}: expected "
                        f"{CC_EMBEDDING_DIM} or {CC_FULL_EMBEDDING_DIM}, got {cc_vector.numel()}"
                    )
                if not torch.isfinite(cc_selected).all():
                    raise ValueError(f"Non-finite Chemical Checker values found for {compound_id}")
                out_vec = torch.cat([out_vec, cc_selected])
            else:
                embedding = torch.as_tensor(tmp_vec[compound_id], dtype=torch.float32).flatten()
                expected_dim = MORGAN_EMBEDDING_DIM if key == "mf" else CHEMBERTA_EMBEDDING_DIM
                if embedding.numel() != expected_dim:
                    raise ValueError(
                        f"Invalid {key} dimension for {compound_id}: "
                        f"expected {expected_dim}, got {embedding.numel()}"
                    )
                if not torch.isfinite(embedding).all():
                    raise ValueError(f"Non-finite {key} values found for {compound_id}")
                out_vec = torch.cat([out_vec, embedding])

        if out_vec.numel() != BACNET_INPUT_DIM:
            raise ValueError(
                f"Invalid concatenated BaCNet input dimension for "
                f"{protein_name}/{compound_id}: expected {BACNET_INPUT_DIM}, got {out_vec.numel()}"
            )

        """
        return: output vector, (protein name, drug name)
        """
        return out_vec, self.links[idx]



@dataclass
class FrozenECDF:
    ref_sorted: Optional[np.ndarray] = None
    n: int = 0
    eps: float = 0.0
    version: str = "ecdf_v1"

    def fit(self, ref_scores: ArrayLike, eps_mode: str = "N+1") -> "FrozenECDF":
        ref = np.asarray(ref_scores, dtype=float)
        if ref.ndim != 1:
            raise ValueError("ref_scores must be 1-D.")
        if len(ref) < 10:
            raise ValueError("Ref size is too small (<10). Provide more reference scores.")
        self.ref_sorted = np.sort(ref)
        self.n = len(self.ref_sorted)
        if eps_mode == "N+1":
            self.eps = 0.5 / (self.n + 1)
        elif eps_mode == "N":
            self.eps = 0.5 / self.n
        else:
            self.eps = float(eps_mode)
        return self

    def transform(self, scores: ArrayLike) -> np.ndarray:
        if self.ref_sorted is None:
            raise RuntimeError("Call fit() first.")
        s = np.asarray(scores, dtype=float)
        grid = np.linspace(0.0, 1.0, self.n, endpoint=True)
        u = np.interp(s, self.ref_sorted, grid, left=0.0, right=1.0)
        return np.clip(u, self.eps, 1.0 - self.eps)


    @classmethod
    def load(cls, path: str) -> "FrozenECDF":
        data = np.load(path, allow_pickle=False)
        obj = cls()
        obj.ref_sorted = data["ref_sorted"]
        obj.n = int(data["n"])
        obj.eps = float(data["eps"])
        obj.version = str(data["version"])
        return obj



def create_dataloader_search_drugs(target, library):
    """
    """

    dataset = CustomDataset_Search_Drug(target=target,
                                        library=library,
                                        cc_type=['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5'])
    data_loader = DataLoader(dataset = dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)
    return data_loader



class BaCNet(nn.Module):
    """
    simple three layer perceptron
    input_dim:
        dimension size of input data
    num_features:
        no. of nodes each layer.
        designation no. of nodes by list which contains 3 elements
        ex) [1024,128,64]
    """
    def __init__(self, input_dim, num_features, dropout_rate=0.01):
        super(BaCNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, num_features[0]),
            nn.BatchNorm1d(num_features[0]),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(num_features[0], num_features[1]),
            nn.BatchNorm1d(num_features[1]),
            nn.ReLU(),
        )
        self.l3 = nn.Sequential(
            nn.Linear(num_features[1], num_features[2]),
            nn.BatchNorm1d(num_features[2]),
            nn.ReLU(),
        )
        self.output = nn.Linear(num_features[2], 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        x = self.l1(input)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.l3(x)
        x = self.output(x)
        return x



def create_BaCNet(dropout_rate=0.01):
    model = BaCNet(
        input_dim=BACNET_INPUT_DIM,
        num_features=[1024, 256, 32],
        dropout_rate=dropout_rate,
    )
    return model

        

def search_drug(model, device, dataloader, ecdf_path):
    model.eval()
    memory = {}
    ecdf = FrozenECDF.load(ecdf_path)
    with torch.no_grad():
        for data in dataloader:
            x, pair = data
            output = model(x.to(device)).to('cpu').detach().numpy().copy()
            output =  ecdf.transform(output)
            
            pair_keys = tuple(zip(pair[0], pair[1]))
            memory_tmp = dict(zip(pair_keys,output))
            for k, v in memory_tmp.items():
                memory[k] = v.item()

        print(f"# screened compounds: {len(memory.keys())}")
    return memory
