
import string
from copy import copy

import torch
from torch.utils.data import Dataset
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
        
    def symbols_to_tensor(self, mylist, add_eos=True):
        # Convert a list of token strings to token index (adding a EOS token at end)
        # 
        # Input
        #  mylist  : list of m symbols as strings
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] token index for each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos: mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices) # keep on CPU since this occurs inside Dataset getitem..
        return output

    def tensor_to_symbols(self, v):
        # Convert tensor of token index to token strings, breaking where we get a EOS token.
        #   The EOS token is not included at the end in the result string list.
        # 
        # Input
        #  v : python list of m indices, or 1D tensor
        #   
        # Output
        #  mylist : list of symbols (excluding EOS)
        if torch.is_tensor(v):
            assert v.dim()==1
            v = v.tolist()
        assert isinstance(v, list)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

class LetterStringDataset(Dataset):
    # dataset version where data is loaded from one large .csv file rather than many small .csv files

    def __init__(self, mode: str, data_dir: str):
        # Input
        # mode : 'train' or 'val' data
        # data_dir : directory where data is stored

        assert mode in ['train','val']        
        self.placeholder_length = 100_000 # placeholder number of episodes in epoch
        self.mode = mode
        self.train = mode == 'train'
        self.randomize_order = True
        alphabet = list(string.ascii_lowercase)
        self.langs = {'input' : Lang(alphabet), 'output': Lang(alphabet)}

        # load data:
        self.data = pd.read_csv(f"{data_dir}/{mode}.csv")
        
        # preprocess data:
        # split query into query xq and target yq
        self.data[["xq","yq"]] = self.data["query"].str.split(">", expand=True)
        self.data["xq"] = self.data["xq"].str.strip()
        self.data["yq"] = self.data["yq"].str.strip().str.split(" ")
        # create context:
        sos = " " + self.langs["input"].index2symbol[self.langs["input"].SOS_idx] + " "
        io = self.langs["input"].index2symbol[self.langs["input"].IN_OUT_idx]
        self.data["xq_context"] = self.data["alphabet"] + sos + self.data["study"] + sos + self.data["xq"]
        self.data["xq_context"] = self.data["xq_context"].str.replace(">", io)
        self.data["xq_context"] = self.data["xq_context"].str.split(" ")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int=0):
        # choose index randomly for training dataset:
        if self.train:
            idx = np.random.randint(low=0, high=len(self)) # random index for training
        xq_context = self.data.loc[idx,"xq_context"]
        yq = self.data.loc[idx,"yq"]
        return {"xq_context": xq_context, "yq": yq}

    def pprint(self, sample):
        # Pretty print the episode
        print("\nProblems:")
        for problem, solution in zip(sample["xq_context"], sample["yq"]):
            print("".join(problem).replace("IO", " -> ").replace("SOS", "\n"), "-> ?", f"({"".join(solution)})")

    def collate_fn(self, batch):
        return get_mlc_batch(batch, self.langs)

def pad_seq(seq, max_length):
    # Pad token string sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols (as strings)
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += (max_length - len(seq)) * [PAD_token]
    return seq


def build_padded_tensor(list_seq, lang, add_eos=True, add_sos=False):
    # Transform list of python lists to a padded torch tensors
    # 
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of token srings)
    #  lang : language object for translation of token string to token index
    #  add_eos : add end-of-sequence token at the end?
    #  add_sos : add start-of-sequence token at the beginning?
    #
    # Output
    #  z_padded : LongTensor (n x max_len)
    #  z_lengths : python list of sequence lengths (n-length list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = list_seq
    if add_sos: 
        z_eos = [[SOS_token]+z for z in z_eos]
    if add_eos:
        z_eos = [z+[EOS_token] for z in z_eos]    
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths) # maximum length in this episode
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.symbols_to_tensor(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded, dim=0) # n x max_len
    return z_padded, z_lengths


def get_mlc_batch(samples, langs):
    # Combine individual samples into a batch
    xq_contexts = [sample["xq_context"] for sample in samples]
    yqs = [sample["yq"] for sample in samples]
    xq_context_padded, xq_context_lengths = build_padded_tensor(xq_contexts, langs['input'])
    yq_padded, yq_lengths = build_padded_tensor(yqs, langs['output'])
    yq_sos_padded, yq_sos_lengths = build_padded_tensor(yqs, langs['output'], add_eos=False, add_sos=True)  
    return {
        "xq_context": xq_contexts,
        "yq": yqs,
        "xq_context_padded": xq_context_padded,
        "xq_context_lengths": xq_context_lengths,
        "yq_padded": yq_padded,
        "yq_lengths": yq_lengths,
        "yq_sos_padded": yq_sos_padded,
        "yq_sos_lengths": yq_sos_lengths
        }


def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if '_padded' in k]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE)
    return batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    val_data = LetterStringDataset(data_dir="data", mode="val")
    item = val_data.__getitem__(0)
    print(item)
    val_dataloader = DataLoader(val_data, batch_size=2)
    batch = next(iter(val_dataloader))
    print(batch)
