import pandas as pd
import argparse

def main(dataset:str, reference_dataset: str, seed: int):
    for ds_type in ["train","val", "test"]:
        ref_dataset = pd.read_csv(f"{reference_dataset}/{ds_type}.csv")
        num_tasks_to_keep = ref_dataset.shape[0]
        ds = pd.read_csv(f"{dataset}/{ds_type}.csv")
        ds = ds.sample(n=num_tasks_to_keep, random_state=seed).reset_index(drop=True)
        ds.to_csv(f"{dataset}/{ds_type}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='directory of the dataset to be shrinked. Must be larger than the reference dataset.')
    parser.add_argument('--reference_dataset', help='directory of the dataset which size should be matched by dataset.')
    parser.add_argument('--seed', default=42, type=int, help="random seed for data generation. Default: 42")
    args = parser.parse_args()
    
    main(
        dataset=args.dataset, 
        reference_dataset=args.reference_dataset, 
        seed=args.seed
    )