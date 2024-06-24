import argparse
import os
import pdb
import random
import re
import sys
import time
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm
from utils.dataloader import into_dataset

from models.GregNet2D import GregNet2D
from models.GregNet2D_nobatch import GregNet2D_nobatch
from models.GregNet2DModified import GregNet2DModified
from models.GregNet2DModified_nobatch import GregNet2DModified_nobatch
from models.GWaveNet2D import GWaveNet2D
from models.GWaveNet2D_newpad import GWaveNet2D_newpad
from models.GWaveNet2D_constantpad import GWaveNet2D_constantpad
from models.GWaveNet2D_diffskip import GWaveNet2D_diffskip
from models.GWaveNet2D_newpad_nobatch import GWaveNet2D_newpad_nobatch
from models.GWaveNet2DModified import GWaveNet2DModified
from models.GWaveNet2DModified_nobatch import GWaveNet2DModified_nobatch
from models.GWaveNet2DModified_newpad import GWaveNet2DModified_newpad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    default="./dataset/",
    help="Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation",
)
parser.add_argument(
    "--model_dir",
    default="./models/",
    help="Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth",
)
parser.add_argument(
    "--config_file", default=None, help="quantization configuration file"
)
parser.add_argument(
    "--subset_len",
    default=None,
    type=int,
    help="subset_len to evaluate model, using the whole validation dataset if it is not set",
)
parser.add_argument(
    "--batch_size", default=25, type=int, help="input data batch size to evaluate model"
)
parser.add_argument(
    "--quant_mode",
    default="calib",
    choices=["float", "calib", "test"],
    help="quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model",
)
parser.add_argument(
    "--fast_finetune",
    dest="fast_finetune",
    action="store_true",
    help="fast finetune model before calibration",
)
parser.add_argument(
    "--deploy", dest="deploy", action="store_true", help="export xmodel for deployment"
)
parser.add_argument(
    "--inspect", dest="inspect", action="store_true", help="inspect model"
)

parser.add_argument(
    "--target", dest="target", nargs="?", const="", help="specify target device"
)
parser.add_argument(
    "--decision_threshold",
    dest="decision_threshold",
    type=float,
    default=0.5,
    help="decision threshold for binary classification",
)
parser.add_argument(
    "--model_name", dest="model_name", default="GWaveNet2D", help="model name"
)

args, _ = parser.parse_known_args()


def load_data(
    train=True,
    data_dir="dataset",
    batch_size=25,
    subset_len=None,
    sample_method="random",
    distributed=False,
    model_name="GWaveNet2D",
    batch_n=0,
    quant_mode="calib",
    **kwargs,
):

    train_sampler = None
    if train:
        datasets = []
        for folder in range(0, 5):
            datasets.append(into_dataset(batch="train", current_folder=folder))
        dataset = torch.utils.data.ConcatDataset(datasets)
        # dataset = into_dataset(batch="train", current_folder=0)
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == "random":
                dataset = torch.utils.data.Subset(
                    dataset, random.sample(range(0, len(dataset)), subset_len)
                )
            else:
                dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **kwargs,
        )
    else:
        datasets = []
        if quant_mode == "calib":
            for folder in range(0, 5):
                datasets.append(
                    into_dataset(batch="validation", current_folder=folder, batch_n=0)
                )
        else:
            for folder in range(0, 5):
                datasets.append(
                    into_dataset(batch="test", current_folder=folder, batch_n=batch_n)
                )
        dataset = torch.utils.data.ConcatDataset(datasets)
        # dataset = into_dataset(batch="validation", current_folder=0)
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == "random":
                dataset = torch.utils.data.Subset(
                    dataset, random.sample(range(0, len(dataset)), subset_len)
                )
            else:
                dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, **kwargs
        )
    return data_loader, train_sampler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, decision_threshold):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    with torch.no_grad():
        probabilities = torch.sigmoid(output)
        pred = []

        for probability in probabilities:
            if probability > decision_threshold:
                pred.append(1)
            else:
                pred.append(0)

        batch_size = target.size(0)

        # pred = pred.t() # transposes pred
        correct = []
        for i in range(len(pred)):
            if pred[i] == target[i]:
                correct.append(1)
            else:
                correct.append(0)
        # correct = pred.eq(target)

        # correct = correct.flatten().float().sum(0, keepdim=True)
        correct = sum(correct)
        # res = correct.mul_(100.0 / batch_size)
        res = correct * 100.0 / batch_size
        return res


def evaluate(model, val_loader, loss_fn, decision_threshold):

    model.eval()
    model = model.to(device)
    top1 = AverageMeter("Acc@1", ":6.2f")
    # top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    Loss = 0
    for iteraction, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
        x, y = x.to(device), y.to(device)
        x, y = x.float(), y.float()

        outputs = model(x)
        # squeeze outputs from (32, 1) to (32)
        outputs = outputs.squeeze(1)
        loss = loss_fn(outputs, y)
        Loss += loss.item()
        total += x.size(0)
        acc = accuracy(outputs, y, decision_threshold)
        top1.update(acc, x.size(0))
    return top1.avg, Loss / total


def quantization(title="optimize", model_name="GWaveNet2D", file_path="./models/"):

    data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    inspect = args.inspect
    config_file = args.config_file
    target = args.target
    decision_threshold = args.decision_threshold
    model_name = args.model_name

    if quant_mode != "test" and deploy:
        deploy = False
        print(
            r"Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!"
        )
    if deploy and (batch_size != 1 or subset_len != 1):
        print(
            r"Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!"
        )
        batch_size = 1
        subset_len = 1

    if model_name == "GregNet2D":
        model = GregNet2D("pc").to(device)
    elif model_name == "GregNet2D_nobatch":
        model = GregNet2D_nobatch("pc").to(device)
    elif model_name == "GregNet2DModified":
        model = GregNet2DModified().to(device)
    elif model_name == "GregNet2DModified_nobatch":
        model = GregNet2DModified_nobatch().to(device)
    elif model_name == "GWaveNet2D":
        model = GWaveNet2D(
            in_channels=3, dilation=2, out_channels=64, kernel_size=16, stride=2
        ).to(device)
    elif model_name == "GWaveNet2D_newpad":
        model = GWaveNet2D_newpad(
            in_channels=3, dilation=2, out_channels=64, kernel_size=16, stride=2
        ).to(device)
    elif model_name == "GWaveNet2D_constantpad":
        model = GWaveNet2D_constantpad(
            in_channels=3, dilation=2, out_channels=64, kernel_size=16, stride=2
        ).to(device)
    elif model_name == "GWaveNet2D_diffskip":
        model = GWaveNet2D_diffskip(
            in_channels=3, dilation=2, out_channels=64, kernel_size=16, stride=2
        ).to(device)
    elif model_name == "GWaveNet2D_newpad_nobatch":
        model = GWaveNet2D_newpad_nobatch(
            in_channels=3, dilation=2, out_channels=64, kernel_size=16, stride=2
        ).to(device)
    elif model_name == "GWaveNet2DModified":
        model = GWaveNet2DModified(
            in_channels=3, dilation=2, out_channels=64, kernel_size=2, stride=2
        ).to(device)
    elif model_name == "GWaveNet2DModified_newpad":
        model = GWaveNet2DModified_newpad(
            in_channels=3, dilation=2, out_channels=64, kernel_size=2, stride=2
        ).to(device)
    elif model_name == "GWaveNet2DModified_nobatch":
        model = GWaveNet2DModified_nobatch(
            in_channels=3, dilation=2, out_channels=64, kernel_size=2, stride=2
        ).to(device)

    checkpoint = torch.load(file_path, map_location=device)["model_state_dict"]

    # Adjust the state_dict keys (handle DataParallel)
    state_dict = OrderedDict()
    for key in checkpoint.keys():
        new_key = key.split("module.")[-1]
        state_dict[new_key] = checkpoint[key]

    # Check for missing keys and mismatched sizes
    model_state_dict = model.state_dict()
    missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    size_mismatches = {
        key: (state_dict[key].size(), model_state_dict[key].size())
        for key in state_dict.keys() & model_state_dict.keys()
        if state_dict[key].size() != model_state_dict[key].size()
    }

    # Debug print statements
    if missing_keys:
        print(f"Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys in state_dict: {unexpected_keys}")
    if size_mismatches:
        print(f"Size mismatches: {size_mismatches}")

    # Load the state_dict
    model.load_state_dict(
        state_dict, strict=False
    )  # Set strict=False to ignore missing keys

    # Debug: Check model loading
    print("Model loaded successfully.")

    input = torch.randn([batch_size, 3, 155648])
    if quant_mode == "float":
        quant_model = model
        if inspect:
            if not target:
                raise RuntimeError("A target should be specified for inspector.")
            import sys

            from pytorch_nndct.apis import Inspector

            # create inspector
            inspector = Inspector(target)  # by name
            # start to inspect
            inspector.inspect(quant_model, (input,), device=device)
            sys.exit()

    else:
        ## new api
        ####################################################################################
        quantizer = torch_quantizer(
            quant_mode,
            model,
            (input),
            device=device,
            quant_config_file=config_file,
            target=target,
        )

        quant_model = quantizer.quant_model
        #####################################################################################

    # Debug: Check quantizer and quantized model
    print("Quantizer initialized successfully.")
    print("Quantized model created successfully.")

    # to get loss value after evaluation
    if (
        model_name == "GregNet2D"
        or model_name == "GregNet2DModified"
        or model_name == "GregNet2D_nobatch"
        or model_name == "GregNet2DModified_nobatch"
    ):
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.4)).to(device)
    elif (
        model_name == "GWaveNet2D"
        or model_name == "GWaveNet2DModified"
        or model_name == "GWaveNet2DModified_nobatch"
        or model_name == "GWaveNet2D_newpad"
        or model_name == "GWaveNet2D_newpad_nobatch"
        or model_name == "GWaveNet2DModified_newpad"
        or model_name == "GWaveNet2D_constantpad"
        or model_name == "GWaveNet2D_diffskip"
    ):
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1)).to(device)

    if quant_mode == "calib":
        val_loader, _ = load_data(
            subset_len=subset_len,
            train=False,
            batch_size=batch_size,
            sample_method="random",
            data_dir=data_dir,
            model_name=model_name,
            quant_mode=quant_mode,
        )
        acc1_gen, loss_gen = evaluate(
            quant_model, val_loader, loss_fn, decision_threshold
        )
    elif quant_mode == "test" and deploy:
        val_loader, _ = load_data(
            subset_len=subset_len,
            train=False,
            batch_size=batch_size,
            sample_method="random",
            data_dir=data_dir,
            model_name=model_name,
            batch_n=0,
            quant_mode=quant_mode,
        )
        acc1_gen, loss_gen = evaluate(
            quant_model, val_loader, loss_fn, decision_threshold
        )
    elif quant_mode == "test" or quant_mode == "float":
        acc1_gen_arr = []
        loss_gen_arr = []
        for batch_n in range(5):
            val_loader, _ = load_data(
                subset_len=subset_len,
                train=False,
                batch_size=batch_size,
                sample_method="random",
                data_dir=data_dir,
                model_name=model_name,
                batch_n=batch_n,
                quant_mode=quant_mode,
            )
            acc1_gen, loss_gen = evaluate(
                quant_model, val_loader, loss_fn, decision_threshold
            )
            acc1_gen_arr.append(acc1_gen)
            loss_gen_arr.append(loss_gen)
        acc1_gen = sum(acc1_gen_arr) / len(acc1_gen_arr)
        loss_gen = sum(loss_gen_arr) / len(loss_gen_arr)

    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        ft_loader, _ = load_data(
            subset_len=5120,
            train=False,
            batch_size=batch_size,
            sample_method="random",
            data_dir=data_dir,
            model_name=model_name,
            quant_mode=quant_mode,
        )
        if quant_mode == "calib":
            quantizer.fast_finetune(
                evaluate, (quant_model, ft_loader, loss_fn, decision_threshold)
            )
        elif quant_mode == "test":
            quantizer.load_ft_param()

    # record  modules float model accuracy
    # add modules float model accuracy here
    acc_org1 = 0.0
    loss_org = 0.0

    # logging accuracy
    print("loss: %g" % (loss_gen))
    print(f"top-1 accuracy: {round(acc1_gen, 2)}%")

    # handle quantization result
    if quant_mode == "calib":
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)


if __name__ == "__main__":
    model_name = args.model_name
    file_path = os.path.join(args.model_dir, model_name + ".pt")

    feature_test = " float model evaluation"
    if args.quant_mode != "float":
        feature_test = " quantization"
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += " with optimization"
    else:
        feature_test = " float model evaluation"
    title = model_name + feature_test

    print("Command line options:")
    print(" --data_dir           : ", args.data_dir)
    print(" --model_dir          : ", args.model_dir)
    print(" --config_file        : ", args.config_file)
    print(" --subset_len         : ", args.subset_len)
    print(" --batch_size         : ", args.batch_size)
    print(" --quant_mode         : ", args.quant_mode)
    print(" --fast_finetune      : ", args.fast_finetune)
    print(" --deploy             : ", args.deploy)
    print(" --inspect            : ", args.inspect)
    print(" --target             : ", args.target)
    print(" --decision_threshold : ", args.decision_threshold)
    print(" --model_name         : ", args.model_name)

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(title=title, model_name=model_name, file_path=file_path)

    print("-------- End of {} test ".format(model_name))
