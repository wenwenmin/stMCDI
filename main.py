import argparse
import torch
import datetime
import json
import yaml
import os
import warnings
import random
from src.main_model_st import stMCDI
from src.utils_st import train, evaluate
from dataset import get_dataloader
import numpy as np
import warnings

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning)
parser = argparse.ArgumentParser(description="stMCDI")
parser.add_argument("--config", type=str, default="st_config.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1000)

parser.add_argument("--dataname", type=str, default="DLPFC")
parser.add_argument("--subfolder", type=str, default="151507")

args = parser.parse_args()
print(args)



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(args.seed)

dataname = args.dataname
subfolder = args.subfolder
if dataname == "DLPFC":
    dataset_path = f"data/DLPFC/{subfolder}/{subfolder}.h5ad"
    location_path = f"data/DLPFC/{subfolder}/Location.txt"
elif dataname == "Benchmark":
    dataset_path = f"data/Benchmark/{subfolder}/{subfolder}.h5ad"
    location_path = f"data/Benchmark/{subfolder}/Location.txt"
else:
    raise ValueError("Invalid dataname")

path = "./config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))
# print(config)
# current_time = datetime.datetime.now().strftime("%Y%m")
# foldername = "./save/st_fold" + str(args.nfold) + "_" + str(args.dataname) + "_" + current_time + "/"
foldername = f"./save/{dataname}_{subfolder}"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "/config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    datapath=dataset_path,
    coordinate=location_path,
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

num_features = train_loader.dataset.observed_values.shape[1]
model = stMCDI(config, args.device, num_features).to(args.device)

model_path = foldername+"model.pth"

if not os.path.exists(model_path):
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )


else:
    model.load_state_dict(torch.load(model_path))
print("---------------Start testing---------------")
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)


