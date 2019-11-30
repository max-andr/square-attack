# -*- coding: utf-8 -*-

import torch
import argparse
import robustml
import numpy as np
from foolbox.models import PyTorchModel
from robustml_portal import attacks as atk
from robustml_portal import postAveragedModels as pamdl


# argument parsing
parser = argparse.ArgumentParser(description="robustml evaluation on ImageNet")
parser.add_argument("datasetPath", help="directory containing 'val.txt' and 'val/' folder")
parser.add_argument("--start", type=int, default=0, help="inclusive starting index for data. default: 0")
parser.add_argument("--end", type=int, help="exclusive ending index for data. default: dataset size")
parser.add_argument("--attack", choices=["pgd", "fgsm", "df", "cw", "none"], default="pgd", help="attack method to be used. default: pgd")
parser.add_argument("--device", help="compuation device to be used. 'cpu' or 'cuda:<index>'")
args = parser.parse_args()

if args.device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

# setup test model
model = pamdl.pa_resnet152_config1()
model.to(device)
model.eval()

# setup attacker
nClasses = 1000
victim_model = PyTorchModel(model.model, (0,1), nClasses, device=device, preprocessing=(np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)), np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))))
if args.attack == "pgd":
    attack = atk.pgdAttack(victim_model)
elif args.attack == "fgsm":
    attack = atk.fgsmAttack(victim_model)
elif args.attack == "df":
    attack = atk.dfAttack(victim_model)
elif args.attack == "cw":
    attack = atk.cwAttack(victim_model)
else:
    attack = atk.NullAttack()

# setup data provider
prov = robustml.provider.ImageNet(args.datasetPath, (224, 224, 3))

# evaluate performance
if args.end is None:
    args.end = len(prov)
atk_success_rate = robustml.evaluate.evaluate(model, attack, prov, start=args.start, end=args.end)
print('Overall attack success rate: %.4f' % atk_success_rate)