import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.globals import FOLDER, UU_CREAM, UU_ORANGE


def save_text_output(
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
):
    """
    Save the training and validation losses and accuracies to text files, including the stdev for each epoch.

    Args:
    output_dir (str): The directory to save the files to.
    filename (str): The filename to save the files as.
    train_losses (dict): The training losses for each folder.
    validation_losses (dict): The validation losses for each folder.
    train_accuracies (dict): The training accuracies for each folder.
    validation_accuracies (dict): The validation accuracies for each folder.
    train_losses_stdev (dict): The training losses stdev for each folder.
    validation_losses_stdev (dict): The validation losses stdev for each folder.
    train_accuracies_stdev (dict): The training accuracies stdev for each folder.
    validation_accuracies_stdev (dict): The validation accuracies stdev for each folder.

    Returns:
    None
    """
    with open(
        f"{output_dir}losses/train_{filename}.txt",
        "w",
    ) as f:
        for folder in range(0, FOLDER + 1):
            f.write(f"Folder {folder}\n")
            for item in range(len(train_losses[f"folder {folder}"])):
                string = (
                    str(train_losses[f"folder {folder}"][item])
                    + ", "
                    + str(train_losses_stdev[f"folder {folder}"][item])
                    + "\n"
                )
                f.write(string)

    with open(
        f"{output_dir}losses/validation_{filename}.txt",
        "w",
    ) as f:
        for folder in range(0, FOLDER + 1):
            f.write(f"Folder {folder}\n")
            for item in range(len(validation_losses[f"folder {folder}"])):
                string = (
                    str(validation_losses[f"folder {folder}"][item])
                    + ", "
                    + str(validation_losses_stdev[f"folder {folder}"][item])
                    + "\n"
                )
                f.write(string)

    with open(
        f"{output_dir}accuracies/train_{filename}.txt",
        "w",
    ) as f:
        for folder in range(0, FOLDER + 1):
            f.write(f"Folder {folder}\n")
            for item in range(len(train_accuracies[f"folder {folder}"])):
                string = (
                    str(train_accuracies[f"folder {folder}"][item])
                    + ", "
                    + str(train_accuracies_stdev[f"folder {folder}"][item])
                    + "\n"
                )
                f.write(string)

    with open(
        f"{output_dir}accuracies/validation_{filename}.txt",
        "w",
    ) as f:
        for folder in range(0, FOLDER + 1):
            f.write(f"Folder {folder}\n")
            for item in range(len(validation_accuracies[f"folder {folder}"])):
                string = (
                    str(validation_accuracies[f"folder {folder}"][item])
                    + ", "
                    + str(validation_accuracies_stdev[f"folder {folder}"][item])
                    + "\n"
                )
                f.write(string)


def save_flop_text(flops, loss, output_dir, filename):
    """
    Save the cumulative FLOPs and loss to a text file.

    Args:
    flops (float): The cumulative FLOPs for each minibatch.
    loss (float): The average training loss for each minibatch.
    output_dir (str): The directory to save the files to.
    filename (str): The filename to save the files as.

    Returns:
    None
    """
    with open(f"{output_dir}flops_text/{filename}.txt", "w") as f:
        f.write(f"{flops}, {loss}\n")  # flops, loss


def plot_flops(flops, loss, output_dir, filename):
    """
    Plots the cumulative FLOPs against the average training loss per minibatch.

    Args:
    flops (list): The cumulative FLOPs for each minibatch.
    loss (list): The average training loss for each minibatch.
    output_dir (str): The directory to save the files to.
    filename (str): The filename to save the files as.

    Returns:
    None
    """

    plt.clf()
    plt.plot(flops, loss, color=UU_CREAM)
    plt.title("FLOPs vs Loss")
    plt.xlabel("FLOPs")
    plt.ylabel("Loss")
    plt.savefig(f"{output_dir}flops_plots/{filename}.png")
