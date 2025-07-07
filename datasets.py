from collections import defaultdict
from copy import copy
import math
import random
import string

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

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
        if add_eos: 
            symbols.append(EOS_token)
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


class BatchSampler(Sampler):
    """Creates a sampler that batches problems by the filter set by `batch_by` in `LetterStringDataset`. 
    E.g., if `batch_by=="transformation"` and `dataset.current_filter == "succ"`, the sampler will return a batch of only successor problems
    """
    def __init__(self, dataset, batch_size: int, reshuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.reshuffle = reshuffle
        self.all_indices = None
        self.filter = dataset.filter

    def __iter__(self):
        if (self.reshuffle 
            or self.all_indices is None
            or self.filter.keys() != self.dataset.filter.keys()):

            self.all_indices = []
            # go through all filters
            for f in self.dataset.filter.keys():
                # Get indices for the current filter f
                indices = self.dataset.filter[f]
                indices = indices.copy()
                random.shuffle(indices)
                for i in range(0, len(indices), self.batch_size):
                    self.all_indices.append(indices[i:i+self.batch_size])
            random.shuffle(self.all_indices)
        
        for batch in self.all_indices:
            yield batch

    def __len__(self):
        return math.ceil(len(self.dataset) // self.batch_size)


class LetterStringDataset(Dataset):
    r"""Initialize a LetterStringDataset from a .csv file located at `data_dir`/`mode`.csv.

    Args:
        mode (str): Either "train", "val", or "test"
        data_dir (str): The directory the data is stored in.
        alphabet (list, optional): The unique letters that occur in this dataset. Default: standard unpermuted alphabet (a,b,c,...,z).
        batching_method (str, optional): How to group batches. Options are: \
            "random"- Construct batch with random subset of the problems. \
            "alphabet"- Construct batch with problems from the same alphabet. \
            "transformation"- Construct batch with problems from the same transformation type. \
            "transformation_alphabet"- Construct batch with problems from the same transformation type and from the same alphabet. \
            "study_alphabet"- Construct batch with alphabet and study example held constant. \
            Default: "random".
        query_first (str, optional): Whether to put the query infront of the study example(s) and alphabet or after the alphabet and study example(s). Default: True.
    """
    def __init__(
            self,
            mode: str, 
            data_dir: str, 
            alphabet: list=list(string.ascii_lowercase), 
            batch_size: int=25,
            batching_method: str="random", 
            query_first: bool=False
        ):
        assert mode in ['train','val','test']        
        self.mode = mode
        self.train = mode == 'train'
        self.langs = {'input' : Lang(alphabet), 'output': Lang(alphabet)}
        # load data:
        data = pd.read_csv(f"{data_dir}/{mode}.csv")
        self.batching_method = batching_method
        self.transformation_types = list(data.transformation.unique())
        self.unique_alphabets = list(data.alphabet.unique())
        self.unique_study = list(data.study.unique())
        self.query_first = query_first
        # split problem into query xq and target yq
        data[["xq","yq"]] = data["problem"].str.split(">", expand=True)
        # convert problem (xq=query and yq=solution) to lists
        data["xq"] = data["xq"].str.strip()
        data["yq"] = data["yq"].str.strip().str.split(" ")
        # get yq lengths
        data["yq_lengths"] = data["yq"].apply(lambda x: len(x))
        self.yq_max = data["yq_lengths"].max()

        sos = self.langs["input"].index2symbol[self.langs["input"].SOS_idx]
        io = self.langs["input"].index2symbol[self.langs["input"].IN_OUT_idx]
        data["study"] = data["study"].str.replace("|", sos)

        if query_first:
            # concatenate context (query, study, alphabet)
            data["xq_context"] = data["xq"] + " " + sos + " " + data["study"] + " " + sos + " " + data["alphabet"]
        else:
            # concatenate context (alphabet, study, query)
            data["xq_context"] = data["alphabet"] + " " + sos + " " + data["study"] + " " + sos + " " + data["xq"]
        
        data["xq_context"] = data["xq_context"].str.replace(">", io)
        data["xq_context"] = data["xq_context"].str.split(" ")
        # create tensors
        data["xq_context_tensor"] = data["xq_context"].apply(lambda x: self.langs["input"].symbols_to_tensor(x))
        data["yq_tensor"] = data["yq"].apply(lambda x: self.langs["output"].symbols_to_tensor(x))
        # yq shifted right (starting with io token)
        data["yq_io_tensor"] = data["yq"].apply(lambda x: [io] + x).apply(lambda x: self.langs["output"].symbols_to_tensor(x, add_eos=False))
        # convert to list of dicts for easier retrieval by iterator
        self.data = data.to_dict("records")
        
        # initialize sampling method for obtaining batches from the dataset:
        self.set_filter(batching_method)
        self.sampler = BatchSampler(self, batch_size=batch_size, reshuffle=self.train)

    def set_filter(self, batching_method: str, include=None):
        match batching_method:
            case "transformation":
                # accumulate example ids by transformation type:
                if include is None:
                    self.filter = {trans: [] for trans in self.transformation_types}
                else:
                    self.filter = {trans: [] for trans in include}
                for i, example in enumerate(self.data):
                    if example["transformation"] in self.filter.keys():
                        self.filter[example["transformation"]].append(i)
            case "alphabet":
                # accumulate example ids by alphabet:
                self.filter = {alph: [] for alph in self.unique_alphabets}
                for i, example in enumerate(self.data):
                    self.filter[example["alphabet"]].append(i)
            case "transformation_alphabet":
                # accumulate example ids by transformation type and alphabet:
                if include is None:
                    self.trans_alph_combinations = [" | ".join([trans, alph]) for trans in self.transformation_types for alph in self.unique_alphabets]
                    self.filter = {comb: [] for comb in self.trans_alph_combinations}
                else:
                    self.filter = {trans: [] for trans in include}
                for i, example in enumerate(self.data):
                    if " | ".join([example["transformation"], example["alphabet"]]) in self.filter.keys():
                        self.filter[" | ".join([example["transformation"], example["alphabet"]])].append(i)
            case "study_alphabet":
                # accumulate example ids by transformation type and alphabet:
                self.study_alph_combinations = [" | ".join([study, alph]) for study in self.unique_study for alph in self.unique_alphabets]
                self.filter = {comb: [] for comb in self.study_alph_combinations}
                for i, example in enumerate(self.data):
                    self.filter[" | ".join([example["study"], example["alphabet"]])].append(i)
            case "random":
                self.filter = {"all": list(range(len(self.data)))}
            case _ :
                raise NotImplementedError(f"{batching_method} is an unknown value for the argument batching_method.")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int=None):
        """Return letter-string problem by `idx`. If `idx` is not given, return a random problem instead."""
        return self.data[idx]

    def collate_fn(self, problems: list[dict]) -> defaultdict:
        """Prepares a batch of problems for passing through MLC model. Passed to DataLoader as argument `collate_fn`.

        Args:
            problems (list[dict]): list of letter-string problems obtained via `__getitem__()`

        Returns:
            dict: Padded tensors for MLC forward pass ("xq_context_padded", "yq_padded", "yq_io_padded") & meta data about each problem in the batch. 
        """
        batch = defaultdict(list)
        for d in problems:
            for key, value in d.items():
                batch[key].append(value)
        batch.default_factory = None
        # pad tensors:
        batch["xq_context_padded"] = pad_sequence(batch["xq_context_tensor"], batch_first=True, padding_value=self.langs["input"].PAD_idx)
        batch["yq_padded"] = pad_sequence(batch["yq_tensor"], batch_first=True, padding_value=self.langs["output"].PAD_idx)
        batch["yq_io_padded"] = pad_sequence(batch["yq_io_tensor"], batch_first=True, padding_value=self.langs["output"].PAD_idx)
        return set_batch_to_device(batch)

    def __str__(self):
        return "\n\t".join([
            f"LetterStringDataset({self.mode}):",
            f"{len(self):,} letter-string analogy problems.",
            f"{len(self.transformation_types)} transformation types: {", ".join(self.transformation_types)}",
            f"{len(self.unique_alphabets)} permuted alphabets.",
            f"Batch by: {self.batching_method}.",
            f"Query first: {self.query_first}."
        ])


def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if '_padded' in k]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE, non_blocking=True)
    return batch


class LetterStringDataLoader(DataLoader):
    def __init__(
        self,
        mode: str, 
        data_dir: str, 
        alphabet: list=list(string.ascii_lowercase), 
        batch_size: int=25,
        batching_method: str="random", 
        query_first: bool=False
        ):
        dataset = LetterStringDataset(
            mode,
            data_dir,
            alphabet,
            batch_size,
            batching_method,
            query_first
        )
        super().__init__(dataset, batch_sampler=dataset.sampler, collate_fn=dataset.collate_fn)


if __name__ == "__main__":
    # example for creating Dataset and DataLoader objects:
    from torch.utils.data import DataLoader
    D_val = LetterStringDataset(data_dir="data/no_pred", mode="val")
    item = D_val.__getitem__()
    print("Dataset item:")
    print(item)

    val_dataloader = DataLoader(D_val, batch_size=25, collate_fn=D_val.collate_fn)
    batch = next(iter(val_dataloader))
    print("\nDataloader batch:")
    print(f"with keys: {list(batch.keys())}\n")
    print(batch)