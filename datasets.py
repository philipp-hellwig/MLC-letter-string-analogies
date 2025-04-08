
import os
import torch
import random
import glob
from copy import copy
from torch.utils.data import Dataset
import pandas as pd
import string

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
    
    def __init__(self, mode: str, data_dir: str):
        # Input
        # mode : 'train' or 'val' data
        # data_dir : directory where data is stored

        assert mode in ['train','val']        
        self.placeholder_length = 100_000 # placeholder number of episodes in epoch
        self.mode = mode
        self.train = mode == 'train'
        self.randomize_order = True
        self.dir_items = os.path.join(data_dir,self.mode)
        self.list_items = glob.glob(self.dir_items+"/*.csv") # all episode files

        alphabet = list(string.ascii_lowercase)
        self.langs = {'input' : Lang(alphabet), 'output': Lang(alphabet)}

    def __len__(self):
        if self.train:
            return self.placeholder_length
        else:
            return len(self.list_items)

    def __getitem__(self, idx: int=0):
        if self.train:
            S = read_file(random.choice(self.list_items), self.randomize_order)
        else:
            S = read_file(self.list_items[idx]) # if we truly want to iterate over files
        
        return bundle_ls_episode(S['xs'],S['ys'],S['xq'],S['yq'], S['alphabet'])

    def pprint(self, sample):
        # Pretty print the episode
        print("\nProblems:")
        for problem, solution in zip(sample["xq_context"], sample["yq"]):
            print("".join(problem).replace("IO", " -> ").replace("SOS", "\n"), "-> ?", f"({"".join(solution)})")


def read_file(fn_in: str, randomize_order: bool=False) -> dict:
    """Read batch from csv file.
    Args:
        fn_in (str): Path to the csv file.
        randomize_order (bool): Shuffle the order of samples or leave as is.
    """
    data = pd.read_csv(fn_in)
    if randomize_order:
        data = data.sample(frac=1).reset_index(drop=True)
    
    x_query, y_query = split_problems(data["query"])
    x_study, y_study = split_problems(data["study"])
    alphs = data["alphabet"].apply(lambda x: list(x))
    return {'xs':x_study, 'ys':y_study, 'xq':x_query, 'yq':y_query, 'alphabet': alphs}


def split_problems(problems):
    xs, ys = [], []
    for problem in problems:
        x, y = problem.split("->")
        xs.append(list(x))
        ys.append(list(y))
    return xs,ys


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
    return z_padded,z_lengths


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

def get_mlc_batch(samples, langs):
    # Batch episodes into a series of padded input and target tensors
    # 
    # Input
    #  samples : list of dicts from bundle_biml_episode
    #  langs : input and output version of Lang class
    assert isinstance(samples,list)
    m = len(samples)
    mybatch = {}
    mybatch['list_samples'] = samples
    mybatch['batch_size'] = m
    mybatch['xq_context'] = [] # list of source sequences (as lists) across all episodes
    mybatch['xq'] = []  # list of queries (as lists) across all episodes
    mybatch['yq'] = [] # list of query outputs (as lists) across all episodes
    mybatch['q_idx'] = [] # index of which episode each query belongs to
    for idx in range(m): # each episode
        sample = samples[idx]
        nq = len(sample['xq'])
        assert(nq == len(sample['yq']))
        mybatch['xq_context'] += sample['xq_context']
        mybatch['xq'] += sample['xq']
        mybatch['yq'] += sample['yq']
        mybatch['q_idx'] += [idx*torch.ones(nq, dtype=torch.int)]
    mybatch['q_idx'] = torch.cat(mybatch['q_idx'], dim=0)
    mybatch['xq_context_padded'],mybatch['xq_context_lengths'] = build_padded_tensor(mybatch['xq_context'], langs['input'])
    mybatch['yq_padded'],mybatch['yq_lengths'] = build_padded_tensor(mybatch['yq'], langs['output'])
    mybatch['yq_sos_padded'],mybatch['yq_sos_lengths'] = build_padded_tensor(mybatch['yq'],langs['output'],add_eos=False,add_sos=True)
    return mybatch

def bundle_ls_episode(x_support,y_support,x_query,y_query,alphabet):
    # Bundle components for an episode suitable for optimizing BIML
    # 
    # Input
    #  x_support [length ns list of lists] : input sequences (each a python list of words/symbols)
    #  y_support [length ns list of lists] : output sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : input sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : output sequences (each a python list of words/symbols)
    #  myhash : unique string identifier for this episode (should be order invariant for examples)
    #  aux [dict] : any misc information that we want to pass along with the episode
    #
    # Output
    #  sample : dict that stores episode information
    ns = len(x_support)
    x_query_context = [ [ITEM_SEP] + alphabet[j] + [ITEM_SEP] + x_support[j] + [IO_SEP] + y_support[j] + [ITEM_SEP] + x_query[j] for j in range(ns)] # Create the combined source sequence for every support example
    sample = {}
    sample['identifier'] = alphabet # unique identifying string for this episode (order invariant)
    sample['xs'] = x_support # support 
    sample['ys'] = y_support
    sample['xq'] = x_query # query
    sample['yq'] = y_query
    sample['xq_context'] = x_query_context
    return sample

if __name__ == "__main__":
    import string
    from torch.utils.data import DataLoader
    alphabet = list(string.ascii_lowercase)
    D_train = LetterStringDataset(alphabet=alphabet, mydir="data", mode="train")
    train_dataloader = DataLoader(D_train,batch_size=5,collate_fn=lambda x:get_mlc_batch(x,D_train.langs),
                                        shuffle=True)
    print(next(iter(train_dataloader)))