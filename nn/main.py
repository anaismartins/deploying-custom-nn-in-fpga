# my modules
import gc
import os
import random
import shutil
import warnings

import colored_traceback
import matplotlib.pyplot as plt
import numpy as np
import psutil

# torch
import torch
import torch.optim.lr_scheduler as lr_scheduler
from fvcore.nn import FlopCountAnalysis
from torch import dropout, nn
from torchsummary import summary

from models.globals import (
    DATA_DIR,
    DETECTORS,
    DILATION,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FILENAME_ADD,
    FILTERS,
    FOLDER,
    KERNEL_SIZE_CONVOLUTION,
    LEARNING_RATE,
    MINIBATCH_SIZE,
    MODEL,
    OPTIMIZER_NAME,
    OUTPUT_DIR,
    STRIDE,
)
from models.GregNet import GregNet
from models.GregNet2D import GregNet2D
from models.GregNet2DModified import GregNet2DModified
from models.GWaveNet import GWaveNet
from models.GWaveNet2D import GWaveNet2D
from models.GWaveNet2DModified import GWaveNet2DModified
from nn.generate_outputs import (
    plot_flops,
    save_flop_text,
    save_text_output,
)
from nn.train_model import train_model
from nn.utils import print_or_output

# initial parameters for running the program
# colored errors
colored_traceback.add_hook()
# get more detailed errors for UserWarning
warnings.simplefilter("error", UserWarning)
random.seed(42)
cmap = "RdPu"

if os.path.exists(DATA_DIR):
    d = "cluster"
    filters = FILTERS
else:
    d = "pc"
    filters = 4

# use cuda if available
if d == "pc":
    device = torch.device("cpu")
elif d == "cluster":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")


# length of the input
Lin = 155648

# loss function
if MODEL == "GWaveNet" or MODEL == "GWaveNet2D" or MODEL == "GWaveNet2DModified":
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1))
elif MODEL == "GregNet" or MODEL == "GregNet2D" or MODEL == "GregNet2DModified":
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.4))
l = "BCEWithLogitsLoss"

if d == "pc":
    num_epochs = 3
elif d == "cluster":
    num_epochs = EPOCHS

# filename = f"{m}_{layers}layers_{filters}filters_{len(DETECTORS)}detectors_{l}_{o}_{LEARNING_RATE}lr_{num_epochs}epochs_{CLIP_VALUE}clip_{MINIBATCH_SIZE}minibatch"
if (
    MODEL == "PaperNet"
    or MODEL == "ModifiedPaperNet"
    or MODEL == "GregNet"
    or MODEL == "GregNet2D"
    or MODEL == "GregNet2DModified"
):
    filename = f"{MODEL}_{num_epochs}epochs_{LEARNING_RATE}lr_{MINIBATCH_SIZE}_minibatch_{OPTIMIZER_NAME}optimizer"  # _{LR_DECAY_FACTOR}decayfactor_{LR_DECAY_PATIENCE}patience_{EARLY_STOPPING_MIN_DELTA}early_stopping_delta_{EARLY_STOPPING_PATIENCE}patience"
elif MODEL == "PaperNetModified":
    filename = f"{MODEL}_{num_epochs}epochs_{LEARNING_RATE}lr_{DROPOUT}dropout_{MINIBATCH_SIZE}_minibatch_{OPTIMIZER_NAME}optimizer"  # _{LR_DECAY_FACTOR}decayfactor_{LR_DECAY_PATIENCE}patience_{EARLY_STOPPING_MIN_DELTA}early_stopping_delta_{EARLY_STOPPING_PATIENCE}patience"
elif MODEL == "WavenetSimple":
    filename = f"{MODEL}_{num_epochs}epochs_{LEARNING_RATE}lr_{filters}filters_{KERNEL_SIZE_CONVOLUTION}conv_{DILATION}dilation_{STRIDE}stride_{DROPOUT}dropout_{ACTIVATION}activation_{MINIBATCH_SIZE}minibatch_{OPTIMIZER_NAME}optimizer"  # _{EARLY_STOPPING_MIN_DELTA}early_stopping_delta_{EARLY_STOPPING_PATIENCE}patience"
elif MODEL == "GWaveNet" or MODEL == "GWaveNet2D" or MODEL == "GWaveNet2DModified":
    filename = f"{MODEL}_{num_epochs}epochs_{LEARNING_RATE}lr_{filters}filters_{KERNEL_SIZE_CONVOLUTION}conv_{DILATION}dilation_{MINIBATCH_SIZE}minibatch_{OPTIMIZER_NAME}optimizer"  # _{EARLY_STOPPING_MIN_DELTA}early_stopping_delta_{EARLY_STOPPING_PATIENCE}patience"
else:
    filename = f"{MODEL}_{num_epochs}epochs_{LEARNING_RATE}lr_{filters}filters_{KERNEL_SIZE_CONVOLUTION}conv_{KERNEL_SIZE_MAXPOOL}maxpool_{DILATION}dilation_{PADDING}padding_{STRIDE}stride_{DROPOUT}dropout_{ACTIVATION}activation_{MINIBATCH_SIZE}minibatch_{OPTIMIZER_NAME}optimizer"  # _{LR_DECAY_FACTOR}decayfactor_{LR_DECAY_PATIENCE}patience_{EARLY_STOPPING_MIN_DELTA}early_stopping_delta_{EARLY_STOPPING_PATIENCE}patience"

filename = f"{filename}{FILENAME_ADD}"

if FOLDER == 4:
    filename = f"curriculum_{filename}"
elif FOLDER == 0:
    filename = f"no_curriculum_{filename}"

# set output directory
if d == "pc":
    output_dir = "./../output/"
    out_file = ""
    print(f"Using device: {device}")
elif d == "cluster":
    output_dir = OUTPUT_DIR
    out_file = f"{output_dir}prints/{filename}.txt"
    with open(out_file, "w") as f:
        f.write(f"Using device: {device}\n")

print_or_output(
    f"Used RAM at the beginning: {(psutil.virtual_memory().used * 1e-9):.2} Gb",
    d,
    out_file,
)

if device == "cuda:0":
    print_or_output(f"Using {torch.cuda.get_device_name(0)}", d, out_file)

print_or_output("Training Model", d, out_file)


if MODEL == "MySecondCNN":
    model = MySecondCNN(Lin=Lin, n_input=len(DETECTORS))
elif MODEL == "Wavenet":
    model = Wavenet(
        input_channel=len(DETECTORS),
        dilation=DILATION,
        filters=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
        padding=PADDING,
        d=d,
        out_file=out_file,
        Lin=Lin,
    )
    initial_model = Wavenet(
        input_channel=len(DETECTORS),
        dilation=DILATION,
        filters=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
        padding=PADDING,
        d=d,
        out_file=out_file,
        Lin=Lin,
    )
elif MODEL == "BackToBasics":
    model = BackToBasics(
        d=d,
        out_file=out_file,
        Lin=Lin,
        n_channels=filters,
        kernel_size_conv=KERNEL_SIZE_CONVOLUTION,
        kernel_size_maxpool=KERNEL_SIZE_MAXPOOL,
        dilation=DILATION,
        padding=PADDING,
        stride=STRIDE,
        activation=ACTIVATION,
        weight_init=WEIGHT_INIT,
    )
    initial_model = BackToBasics(
        d=d,
        out_file=out_file,
        Lin=Lin,
        n_channels=filters,
        kernel_size_conv=KERNEL_SIZE_CONVOLUTION,
        kernel_size_maxpool=KERNEL_SIZE_MAXPOOL,
        dilation=DILATION,
        padding=PADDING,
        stride=STRIDE,
        activation=ACTIVATION,
        weight_init=WEIGHT_INIT,
    )
elif MODEL == "BackToBasics3":
    model = BackToBasics3(
        d=d,
        out_file=out_file,
        Lin=Lin,
        n_channels=filters,
        kernel_size_conv=KERNEL_SIZE_CONVOLUTION,
        kernel_size_maxpool=KERNEL_SIZE_MAXPOOL,
        dilation=DILATION,
        padding=PADDING,
        stride=STRIDE,
        activation=ACTIVATION,
        weight_init=WEIGHT_INIT,
    )
    initial_model = BackToBasics3(
        d=d,
        out_file=out_file,
        Lin=Lin,
        n_channels=filters,
        kernel_size_conv=KERNEL_SIZE_CONVOLUTION,
        kernel_size_maxpool=KERNEL_SIZE_MAXPOOL,
        dilation=DILATION,
        padding=PADDING,
        stride=STRIDE,
        activation=ACTIVATION,
        weight_init=WEIGHT_INIT,
    )
elif MODEL == "WavenetSimple":
    model = WavenetSimple(
        input_channel=len(DETECTORS),
        dilation=DILATION,
        filters=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
        padding=PADDING,
        dropout=DROPOUT,
        d=d,
        out_file=out_file,
        Lin=Lin,
    )
    initial_model = WavenetSimple(
        input_channel=len(DETECTORS),
        dilation=DILATION,
        filters=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
        padding=PADDING,
        d=d,
        out_file=out_file,
        Lin=Lin,
    )
elif MODEL == "PaperNet":
    model = PaperNet(d=d)
    initial_model = PaperNet(d=d)
elif MODEL == "PaperNetModified":
    model = PaperNetModified(d=d, dropout=DROPOUT)
    initial_model = PaperNetModified(d=d, dropout=DROPOUT)
elif MODEL == "ModifiedPaperNet":
    model = ModifiedPaperNet(d=d)
    initial_model = ModifiedPaperNet(d=d)
elif MODEL == "GregNet":
    model = GregNet()
    initial_model = GregNet()
elif MODEL == "GregNet2D":
    model = GregNet2D()
    initial_model = GregNet2D()
elif MODEL == "GregNet2DModified":
    model = GregNet2DModified()
    initial_model = GregNet2DModified()
elif MODEL == "GWaveNet":
    model = GWaveNet(
        in_channels=len(DETECTORS),
        dilation=DILATION,
        out_channels=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
    )
    initial_model = GWaveNet(
        in_channels=len(DETECTORS),
        dilation=DILATION,
        out_channels=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
    )
elif MODEL == "GWaveNet2D":
    model = GWaveNet2D(
        in_channels=len(DETECTORS),
        dilation=DILATION,
        out_channels=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
    )
    initial_model = GWaveNet2D(
        in_channels=len(DETECTORS),
        dilation=DILATION,
        out_channels=filters,
        kernel_size=KERNEL_SIZE_CONVOLUTION,
        stride=STRIDE,
    )
elif MODEL == "GWaveNet2DModified":
    kernel_size_convolution = 2
    model = GWaveNet2DModified(
        in_channels=len(DETECTORS),
        dilation=DILATION,
        out_channels=filters,
        kernel_size=kernel_size_convolution,
        stride=STRIDE,
    )
    initial_model = GWaveNet2DModified(
        in_channels=len(DETECTORS),
        dilation=DILATION,
        out_channels=filters,
        kernel_size=kernel_size_convolution,
        stride=STRIDE,
    )

# model = nn.DataParallel(model)
# initial_model = nn.DataParallel(initial_model)
model = model.to(device)
initial_model = initial_model.to(device)

summary(model, (len(DETECTORS), Lin))

input = torch.randn(1, 3, Lin).to(device)
flops = FlopCountAnalysis(model, input)
n_flops = flops.total()
print(f"Flops: {n_flops}")

# optimzer
if OPTIMIZER_NAME == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER_NAME == "Adamax":
    optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER_NAME == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER_NAME == "AdamaxGreg":
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )

print_or_output(
    f"Used RAM before training: {(psutil.virtual_memory().used * 1e-9):.2} Gb",
    d,
    out_file,
)

(
    train_accuracies,
    train_accuracies_stdev,
    train_losses,
    train_losses_stdev,
    validation_accuracies,
    validation_accuracies_stdev,
    validation_losses,
    validation_losses_stdev,
    test_accuracy,
    final_epoch,
    cfm,
    best_epoch,
    probabilities_full_epoch,
    y_full_epoch,
    save_flops,
    save_loss,
) = train_model(
    device,
    d,
    out_file,
    output_dir,
    filename,
    epochs=num_epochs,
    optimizer=optimizer,
    model=model,
    initial_model=initial_model,
    loss_fn=loss_fn,
    minibatch_size=MINIBATCH_SIZE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    n_flops=n_flops,
)

final_epoch = 0
for folder in range(0, FOLDER + 1):
    final_epoch = final_epoch + best_epoch[f"folder {folder}"] + 1

filename = f"{round(test_accuracy, 2)}Acc_{filename}"

# storing losses and accuracies
save_text_output(
    output_dir,
    filename,
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies,
    train_losses_stdev,
    validation_losses_stdev,
    train_accuracies_stdev,
    validation_accuracies_stdev,
)

save_flop_text(save_flops, save_loss, output_dir, filename)

plot_flops(save_flops, save_loss, output_dir, filename)

# save training text file with all the info
if d == "cluster":
    shutil.copyfile(out_file, f"{output_dir}training_logs/{filename}.txt")

del train_accuracies
del validation_accuracies
del train_losses
del validation_losses
torch.cuda.empty_cache()
gc.collect()


print_or_output(f"Done: {filename}", d, out_file)
