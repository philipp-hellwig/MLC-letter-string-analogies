from dataclasses import dataclass, asdict

import torch
from torch.utils.data import DataLoader
import torch.optim
import pandas as pd

from model import MLC, MLCConfig
from datasets import LetterStringDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TrainConfig:
    filename_model: str
    dir_model: str
    dir_data: str
    batch_size: int
    batching_method: str
    query_first: bool
    nepochs: int
    lr: float
    lr_end_factor: int
    lr_warmup: bool
    save_best: bool
    save_best_skip: float


class CheckPoint:
    def __init__(self, mlc_config: MLCConfig, train_config: TrainConfig):
        self.train_config = train_config
        self.mlc_config = mlc_config
        self.best_val_loss = float('inf')
        self.loss_hist = []
        self.val_acc_hist = []
        self.save_path = f"{train_config.dir_model}/{train_config.filename_model}"
        self.epoch = 1
        self.step = 0
        self.net_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_epoch_state_dict = None

    def check_compatibility(self, current_mlc_config, current_train_config):
        if current_mlc_config == self.mlc_config and current_train_config == self.train_config:
            return True
        else:
            mismatches = []
            saved_train = asdict(self.train_config)
            current_train = asdict(current_train_config)
            for k in saved_train:
                if saved_train[k] != current_train.get(k):
                    mismatches.append(f"TrainConfig: {k}: checkpoint={saved_train[k]}, current={current_train.get(k)}")
            saved_mlc = asdict(self.mlc_config)
            current_mlc = asdict(current_mlc_config)
            for k in saved_mlc:
                if saved_mlc[k] != current_mlc.get(k):
                    mismatches.append(f"MLCConfig: {k}: checkpoint={saved_mlc[k]}, current={current_mlc.get(k)}")
            msg = "Config mismatches found:\n" + "\n".join("  " + m for m in mismatches)
            raise AssertionError(msg)

    def load_dataloaders(self, data_dir="data", verbose: bool = True, batch_size: int=None, use_datasets=["train", "val"]) -> tuple:
        """Load and return train, validation, and test DataLoaders"""
        
        dataloaders = []
        for dataset in use_datasets:
            ds = LetterStringDataset(
                data_dir=data_dir,
                mode=dataset,
                batching_method=self.train_config.batching_method,
                batch_size=self.train_config.batch_size if batch_size is None else batch_size,
                query_first=self.train_config.query_first,
            )
            dataloader = DataLoader(ds, batch_sampler=ds.sampler, collate_fn=ds.collate_fn)           
            dataloaders.append(dataloader)
        if verbose:
            print("Loaded following dataloaders:")
            print(",\n".join([f"{dataset} dataloader (n={len(dataloader.dataset):,})" for dataset, dataloader in zip(use_datasets, dataloaders)]))
        return dataloaders

    def load_optimizer(self, model, optimizer_class=torch.optim.AdamW):
        optimizer = optimizer_class(
            model.parameters(),
            lr=self.train_config.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        optimizer.load_state_dict(self.optimizer_state_dict)
        return optimizer

    def load_scheduler(self, optimizer, scheduler_class=torch.optim.lr_scheduler.LinearLR):
        scheduler = scheduler_class(
            optimizer,
            start_factor=1.0,
            end_factor=self.train_config.lr_end_factor,
            total_iters=self.train_config.nepochs - 1,
        )
        scheduler.load_state_dict(self.scheduler_epoch_state_dict)
        return scheduler

    def load_model(self, verbose: bool = True) -> MLC:
        """Load MLC model"""
        # Initialize model architecture:
        net = MLC(**vars(self.mlc_config))
        # load trained parameters:
        net.load_state_dict(self.net_state_dict)
        net = net.to(device=DEVICE)
        if verbose:
            print(f"Loading model that has completed {self.epoch} of {self.train_config.nepochs} epochs")
            print(f"\tbatch size: {self.train_config.batch_size}")
            print(f"\tnumber of steps: {self.step:,}")
            print(f"\tbest val loss achieved: {self.best_val_loss:.4f}")
            print(net)
        return net

    def resume_training(self, mlc_config, train_config):
        if self.check_compatibility(mlc_config, train_config):
            model = self.load_model(DEVICE)
            optimizer = self.load_optimizer(model)
            scheduler = self.load_scheduler(optimizer)
            self.epoch += 1
            return model, optimizer, scheduler

    def save(self, model, optimizer, scheduler_epoch, is_best=False, save_path:str=None):
        self.net_state_dict = model.state_dict()
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_epoch_state_dict = scheduler_epoch.state_dict()
        if save_path is None:
            if is_best:
                s = self.save_path.rsplit(".", 1)  # split off extension
                save_path = s[0] + "_best." + s[1]
                print("> Saving new *best* model as", save_path, end="")
            else:
                save_path = self.save_path
        if not is_best:
            print("> Saving model as", save_path, end="")
        state = vars(self).copy()
        state["train_config"] = asdict(self.train_config)
        state["mlc_config"] = asdict(self.mlc_config)
        torch.save(state, save_path)
        print(" < Done. >")

    @classmethod
    def from_pt(cls, path: str):
        state = torch.load(path, DEVICE, weights_only=False)
        saved_train_config = TrainConfig(**state["train_config"])
        saved_mlc_config = MLCConfig(**state["mlc_config"])
        cp = CheckPoint(saved_mlc_config, saved_train_config)
        filtered_state = {k: v for k, v in state.items() if k not in ("train_config", "mlc_config")}
        cp.__dict__.update(filtered_state)
        return cp

    @classmethod
    def update_version(cls, current_path: str, data_dir: str, save_path: str):
        """Updates version of .pt file to be compatible with new CheckPoint implementation.

        Args:
            current_path (str): the current_path to the .pt file
            data_dir (str): the path to the data directory the model was trained on
            save_path (str): the path to which the model should be saved
            device (str): _description_
        """
        cp_state = torch.load(current_path, map_location=DEVICE, weights_only=False)
        print(cp_state)
        if "args" in cp_state.keys():
            args = cp_state["args"]
            if args.sampling_method:
                args.batching_method = args.sampling_method
        
            train_config = TrainConfig(
                args.filename_model,
                args.dir_model,
                args.dir_data,
                args.batch_size,
                args.batching_method,
                args.query_first,
                args.nepochs,
                args.lr,
                args.lr_end_factor,
                args.lr_warmup,
                args.save_best,
                args.save_best_skip,
            )
            D_train = LetterStringDataset("train", data_dir=data_dir)
            # setup model:
            mlc_config = MLCConfig(
                hidden_size=args.emb_size, 
                input_size=D_train.langs['input'].n_symbols, 
                output_size=D_train.langs['output'].n_symbols,
                PAD_idx_input=D_train.langs['input'].PAD_idx, 
                PAD_idx_output=D_train.langs['output'].PAD_idx,
                nlayers_encoder=args.nlayers_encoder, 
                nlayers_decoder=args.nlayers_decoder,
                nhead=args.nheads,
                dropout_p=args.dropout,
                activation= args.act,
                ff_mult=args.ff_mult
            )
            cp = CheckPoint(mlc_config, train_config)

            cp.best_val_loss = cp_state["best_val_loss"]
            cp.loss_hist = cp_state["train_tracker"]
            cp.val_acc_hist = cp_state["val_accuracy"]
            cp.step = cp_state["step"]
            cp.epoch = cp_state["epoch"]
            cp.net_state_dict = cp_state["nets_state_dict"]
            cp.optimizer_state_dict = cp_state["optimizer_state_dict"]
            cp.scheduler_epoch_state_dict = cp_state["scheduler_epoch_state_dict"]

            model, optimizer, scheduler_epoch = cp.resume_training(mlc_config, train_config)
            cp.save(model, optimizer, scheduler_epoch, save_path=save_path)
        else:
            cp = CheckPoint.from_pt(current_path)
            if cp.train_config.batching_method == "unstructured":
                cp.train_config.batching_method = "random"
            model, optimizer, scheduler_epoch = cp.resume_training(cp.mlc_config, cp.train_config)
            cp.save(model, optimizer, scheduler_epoch, save_path=save_path)

    def __str__(self):
        val_acc = pd.DataFrame(self.val_acc_hist)
        val_acc = pd.melt(val_acc, id_vars=['epoch'], value_vars=["in", "out-of", "overall"], var_name="distribution", value_name="accuracy")
        max_overall_acc = val_acc.loc[val_acc["distribution"] =="overall","accuracy"].max()
        max_id_acc = val_acc.loc[val_acc["distribution"] =="in","accuracy"].max()
        max_ood_acc = val_acc.loc[val_acc["distribution"] =="out-of","accuracy"].max()
        return (
            "### MLC Config\n\n"
            + "```\n"
            + "\n".join([f"{key}: {item}" for key, item in self.mlc_config.__dict__.items()])
            + "\n```\n\n"
            + "### Train Config\n\n"
            + "```\n"
            + "\n".join([f"{key}: {item}" for key, item in self.train_config.__dict__.items()])
            + "\n```"
            + "\n\n"
            + "### Training Run\n\n"
            + f"Trained for {self.epoch-1} out of {self.train_config.nepochs} epochs."
            + "\n\n"
            + f"Best Val. Loss: {self.best_val_loss:.3f}\n\n"
            + f"Highest Val. Accuracy: {max_overall_acc:.3f}, in-distribution: {max_id_acc:.3f}, out-of-distribution: {max_ood_acc:.3f}"
        )