
import string
from copy import copy
import torch
from torch.utils.data import Dataset
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

        # convert to a list of dicts for faster processing
        self.samples = []
        for _, row in self.data.iterrows():
            self.samples.append({
                "xq_context": row["xq_context"],
                "yq": row["yq"],
                "xq_context_tensor": self.langs["input"].symbols_to_tensor(row["xq_context"]),
                "yq_tensor": self.langs["output"].symbols_to_tensor(row["yq"]),
                "yq_sos_tensor": self.langs["output"].symbols_to_tensor([sos.strip()] + row["yq"], add_eos=False),
                "transformation": row["transformation"],
                "n_perm": row["n_perm"]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int=0):
        return self.samples[idx]

    def pprint(self, sample):
        # Pretty print each episode
        print("\nProblems:")
        for problem, solution in zip(sample["xq_context"], sample["yq"]):
            print("".join(problem).replace("IO", " -> ").replace("SOS", "\n"), "-> ?", f"({"".join(solution)})")

    def collate_fn(self, samples):
        # context and target as symbols:
        xq_context = [s["xq_context"] for s in samples]
        yq = [s["yq"] for s in samples]

        # extract tensors:
        xq_tensors = [s["xq_context_tensor"] for s in samples]
        yq_tensors = [s["yq_tensor"] for s in samples]
        yq_sos_tensors = [s["yq_sos_tensor"] for s in samples]

        # pad sequences
        xq_padded = pad_sequence(xq_tensors, batch_first=True, padding_value=samples[0]["xq_context_tensor"].new_tensor([self.langs["input"].PAD_idx]).item())
        yq_padded = pad_sequence(yq_tensors, batch_first=True, padding_value=samples[0]["yq_tensor"].new_tensor([self.langs["output"].PAD_idx]).item())
        yq_sos_padded = pad_sequence(yq_sos_tensors, batch_first=True, padding_value=samples[0]["yq_sos_tensor"].new_tensor([self.langs["output"].PAD_idx]).item())

        # lengths
        xq_lengths = [len(t) for t in xq_tensors]
        yq_lengths = [len(t) for t in yq_tensors]
        yq_sos_lengths = [len(t) for t in yq_sos_tensors]

        return {
            "xq_context": xq_context,
            "yq": yq,
            "xq_context_padded": xq_padded,
            "xq_context_lengths": xq_lengths,
            "yq_padded": yq_padded,
            "yq_lengths": yq_lengths,
            "yq_sos_padded": yq_sos_padded,
            "yq_sos_lengths": yq_sos_lengths,
            "transformation": [s["transformation"] for s in samples],
            "n_perm": [s["n_perm"] for s in samples],
        }


def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if '_padded' in k]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE, non_blocking=True)
    return batch


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # example for creating Dataset and DataLoader objects:
    D_val = LetterStringDataset(data_dir="data", mode="val")
    item = D_val.__getitem__(0)
    print("Dataset item:")
    print(item)

    val_dataloader = DataLoader(D_val, batch_size=25, collate_fn=D_val.collate_fn, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    batch = next(iter(val_dataloader))
    print("\nDataloader batch:")
    print(f"with keys: {batch.keys()}\n")
    print(batch)
