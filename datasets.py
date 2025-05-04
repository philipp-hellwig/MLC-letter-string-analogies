import string
from collections import defaultdict
from copy import copy

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "SOS" # start of sentence
EOS_token = "EOS" # end of sentence
PAD_token = "PAD" # padding symbol
IO_SEP = 'IO' # separator '->' between input/outputs in support examples
ITEM_SEP  = SOS_token # separator '|' between support examples in input sequence


class Lang:
    def __init__(self, alphabet: list, in_out="IO"):
        n = len(alphabet)
        self.alphabet = alphabet # list of non-special symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token, n+2: PAD_token, n+3: in_out}
        self.symbol2index = {SOS_token : n, EOS_token : n+1, PAD_token : n+2, in_out: n+3}
        for idx, letter in enumerate(alphabet):
            self.index2symbol[idx] = letter
            self.symbol2index[letter] = idx
        self.n_symbols = len(self.index2symbol)
        self.SOS_idx = n
        self.EOS_idx = n+1
        self.PAD_idx = self.symbol2index[PAD_token]
        self.IN_OUT_idx = n+3
        self.PAD_token = PAD_token
        
    def symbols_to_tensor(self, symbols: list, add_eos=True) -> torch.LongTensor:
        """Convert a list of token strings to a tensor of symbol indices. Adds EOS token at end by default

        Args:
            symbols (list): list of m symbols as strings
            add_eos (bool, optional): Add EOS token at the end of the sequence? Defaults to True.

        Returns:
            torch.LongTensor: LongTensor of length [m or m+1(in case `add_eos`=True) ] which contains the token index (integer) for each symbol (plus EOS if appropriate).
        """
        symbols = copy(symbols)
        if add_eos: symbols.append(EOS_token)
        indices = [self.symbol2index[s] for s in symbols]
        output = torch.tensor(indices, dtype=torch.int64)
        return output

    def tensor_to_symbols(self, indices) -> list:
        """Convert tensor of token index to token strings, breaking where we get a EOS token.
        The EOS token is not included at the end in the result string list.

        Args:
            indices: list of symbol indices or tensor of symbols

        Returns:
            list: A list of symbols (str) that correspond to the sequence of symbol `indices`. 
        """
        if torch.is_tensor(indices):
            assert indices.dim()==1
            indices = indices.tolist()
        assert isinstance(indices, list)
        symbols = []
        for x in indices:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            symbols.append(s)
        return symbols


class LetterStringDataset(Dataset):
    def __init__(self, mode: str, data_dir: str, alphabet: list=list(string.ascii_lowercase)):
        """Initialize a LetterStringDataset from a .csv file located at `data_dir`/`mode`.csv.

        Args:
            mode (str): Either "train" or "val"
            data_dir (str): The directory the data is stored in.
            alphabet (list): The unique letters that occur in this alphabet.
        """
        assert mode in ['train','val']        
        self.mode = mode
        self.train = mode == 'train'
        self.langs = {'input' : Lang(alphabet), 'output': Lang(alphabet)}
        # load data:
        self.data = pd.read_csv(f"{data_dir}/{mode}.csv")
        self.transformation_types = list(self.data.transformation.unique())
        # preprocess data:
        # split problem into query xq and target yq
        self.data[["xq","yq"]] = self.data["problem"].str.split(">", expand=True)
        self.data["xq"] = self.data["xq"].str.strip()
        # create context (alphabet, study example, query):
        sos = self.langs["input"].index2symbol[self.langs["input"].SOS_idx]
        io = self.langs["input"].index2symbol[self.langs["input"].IN_OUT_idx]
        # prepare context
        self.data["xq_context"] = self.data["alphabet"] + " " + sos + " " + self.data["study"] + " " + sos + " " + self.data["xq"]
        self.data["xq_context"] = self.data["xq_context"].str.replace(">", io)
        self.data["xq_context"] = self.data["xq_context"].str.split(" ")

        # convert problem (xq=query and yq=solution) to lists
        self.data["xq"] = self.data["xq"].str.split(" ")
        self.data["yq"] = self.data["yq"].str.strip().str.split(" ")
        # get yq lengths
        self.data["yq_lengths"] = self.data["yq"].apply(lambda x: len(x))
        self.yq_max = self.data["yq_lengths"].max()
        # create tensors
        self.data["xq_context_tensor"] = self.data["xq_context"].apply(lambda x: self.langs["input"].symbols_to_tensor(x))
        self.data["yq_tensor"] = self.data["yq"].apply(lambda x: self.langs["output"].symbols_to_tensor(x))
        # yq shifted right (starting with sos token)
        self.data["yq_sos_tensor"] = self.data["yq"].apply(lambda x: [sos] + x).apply(lambda x: self.langs["output"].symbols_to_tensor(x, add_eos=False))
        # convert to list of dicts for easier retrieval by iterator
        self.data = self.data.to_dict("records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int=None):
        """Return letter-string problem by idx. If idx is not given, choose random problem instead."""
        if idx is None:
            idx = np.random.randint(len(self))
        return self.data[idx]

    def collate_fn(self, problems: list[dict]) -> dict:
        """Prepares a batch of problems for passing through MLC model. Passed to DataLoader as argument `collate_fn`.

        Args:
            problems (list[dict]): list of letter-string problems obtained via `__getitem__()`

        Returns:
            dict: Padded tensors for MLC forward pass ("xq_context_padded", "yq_padded", "yq_sos_padded") & meta data about each problem in the batch. 
        """
        batch = defaultdict(list)
        for d in problems:
            for key, value in d.items():
                batch[key].append(value)
        batch.default_factory = None
        # pad tensors:
        batch["xq_context_padded"] = pad_sequence(batch["xq_context_tensor"], batch_first=True, padding_value=self.langs["input"].PAD_idx)
        batch["yq_padded"] = pad_sequence(batch["yq_tensor"], batch_first=True, padding_value=self.langs["output"].PAD_idx)
        batch["yq_sos_padded"] = pad_sequence(batch["yq_sos_tensor"], batch_first=True, padding_value=self.langs["output"].PAD_idx)
        return set_batch_to_device(batch)


def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if '_padded' in k]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE, non_blocking=True)
    return batch


if __name__ == "__main__":
    # example for creating Dataset and DataLoader objects:
    from torch.utils.data import DataLoader
    D_val = LetterStringDataset(data_dir="data/no_pred", mode="val")
    item = D_val.__getitem__(0)
    print("Dataset item:")
    print(item)

    val_dataloader = DataLoader(D_val, batch_size=25, collate_fn=D_val.collate_fn)
    batch = next(iter(val_dataloader))
    print("\nDataloader batch:")
    print(f"with keys: {batch.keys()}\n")
    print(batch)
