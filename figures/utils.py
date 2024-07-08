import numpy as np
from math import sqrt

def calculate_best_epochs(epoch_steps, best_epochs):
    # Initialize lists to store the total epochs and the best epochs calculated
    total_epochs = []
    best_epochs_calculated = []
    
    # Initialize a variable to keep track of the cumulative sum of epoch steps
    cs = 0
    
    # Calculate the total epochs and the best epochs for each step
    for i in range(len(epoch_steps)):
        cs += epoch_steps[i]
        total_epochs.append(cs)
        best_epoch_at_step = (sum(epoch_steps[:i]) if i > 0 else 0) + best_epochs[i]
        best_epochs_calculated.append(best_epoch_at_step)
    return best_epochs_calculated

def select_values(values_list, indices):
    return [values_list[i] for i in indices]


def cumulative_sum(lst):
    result = []
    cumulative = 0
    for num in lst:
        cumulative += num
        result.append(cumulative)
    return result
    
def process_model_data(model, n_folders, n_samples_train, n_samples_val, 
                       filename_learning_loss_train, filename_learning_loss_val, 
                       filename_learning_acc_train, filename_learning_acc_val):

    def read_file(filename):
        with open(filename, "r") as f:
            return f.readlines()

    def process_acc_data(lines):
        acc, acc_stdev = [], []
        n_epochs, epoch_counter = [], 0

        for line in lines:
            if "Folder" in line:
                if acc:
                    n_epochs.append(epoch_counter)
                epoch_counter = 0  # Reset for new folder
            else:
                acc.append(float(line.split(", ")[0]))
                acc_stdev.append(float(line.split(", ")[1]))
                epoch_counter += 1

        n_epochs.append(epoch_counter)
        return acc, acc_stdev, n_epochs

    def process_loss_data(lines):
        loss, loss_stdev = [], []
        n_epochs, epoch_counter = [], 0

        for line in lines:
            if "Folder" in line:
                if loss:
                    n_epochs.append(epoch_counter)
                epoch_counter = 0  # Reset for new folder
            else:
                loss.append(float(line.split(", ")[0][7:-2]))
                loss_stdev.append(float(line.split(", ")[1][7:-2]))
                epoch_counter += 1

        n_epochs.append(epoch_counter)
        return loss, loss_stdev, n_epochs

    def calculate_shaded_loss(loss, loss_stdev, n_epochs, n_samples, n_folders):
        shaded_losses = []
        n_epochs = cumulative_sum(n_epochs)
        for i in range(n_folders):

            if i == 0:
                shaded_loss = [3 * x / np.sqrt(n_samples * 2 * (i + 1)) for x in loss[0:n_epochs[i]]]
            else:
                shaded_loss.extend([3 * x / np.sqrt(n_samples * 2 * (i + 1)) for x in loss_stdev[n_epochs[i-1]:n_epochs[i]]])

        return shaded_loss

    def calculate_best_epoch(lines, n_folders):
        best_epoch = [0] * n_folders
        epoch_counter = 0
        best_loss = np.inf
        current_folder = 0

        for line in lines:
            if "Folder" in line:
                current_folder = int(line.split(" ")[1])
                best_loss = np.inf
                epoch_counter = 0
            else:
                val_loss = float(line.split(", ")[0][7:-2])
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch[current_folder] = epoch_counter + 1
                epoch_counter += 1

        return best_epoch

    # Process training loss
    lines = read_file(filename_learning_loss_train)
    train_loss, train_loss_stdev, n_epochs = process_loss_data(lines)
    shaded_loss_train = calculate_shaded_loss(train_loss, train_loss_stdev, n_epochs, n_samples_train, n_folders)
    
    # Process validation loss
    lines = read_file(filename_learning_loss_val)
    val_loss, val_loss_stdev, _ = process_loss_data(lines)
    shaded_loss_val = calculate_shaded_loss(val_loss, val_loss_stdev, n_epochs, n_samples_val, n_folders)
    best_epoch = calculate_best_epoch(lines, n_folders)

    # Process training accuracy
    lines = read_file(filename_learning_acc_train)
    train_acc, train_acc_stdev, _ = process_acc_data(lines)
    shaded_acc_train = calculate_shaded_loss(train_acc, train_acc_stdev, n_epochs, n_samples_train, n_folders)

    # Process validation accuracy
    lines = read_file(filename_learning_acc_val)
    val_acc, val_acc_stdev, _ = process_acc_data(lines)
    shaded_acc_val = calculate_shaded_loss(val_acc, val_acc_stdev, n_epochs, n_samples_val, n_folders)

    return {
        "train_loss": train_loss,
        "train_loss_stdev": train_loss_stdev,
        "val_loss": val_loss,
        "val_loss_stdev": val_loss_stdev,
        "train_acc": train_acc,
        "train_acc_stdev": train_acc_stdev,
        "val_acc": val_acc,
        "val_acc_stdev": val_acc_stdev,
        "shaded_loss_train": shaded_loss_train,
        "shaded_loss_val": shaded_loss_val,
        "shaded_acc_train": shaded_acc_train,
        "shaded_acc_val": shaded_acc_val,
        "best_epoch": best_epoch,
        "n_epochs": n_epochs
    }

def process_lines(lines, decision_threshold):
    true_y, pred_y, pisnr = [], [], []
    tmp_true_y, tmp_pred_y, tmp_pisnr = [], [], []

    for line in lines:
        if "Folder" in line:
            if tmp_true_y:
                true_y.append(tmp_true_y)
                pred_y.append(tmp_pred_y)
                pisnr.append(tmp_pisnr)
            tmp_true_y, tmp_pred_y, tmp_pisnr = [], [], []
        else:
            parts = line.split(": ")
            tmp_true_y.append(int(float(parts[0])))
            prediction = float(parts[1])
            tmp_pred_y.append(1 if prediction > decision_threshold else 0)
            tmp_pisnr.append(float(parts[2]))
    
    if tmp_true_y:
        true_y.append(tmp_true_y)
        pred_y.append(tmp_pred_y)
        pisnr.append(tmp_pisnr)
    
    return true_y, pred_y, pisnr

def update_metrics(true_y, pred_y, pisnr, cases):
    n_folders = len(true_y)
    avg_pisnr = [[[] for _ in range(len(cases) + 1)] for _ in range(n_folders)]
    tp = [[0] * (len(cases) + 1) for _ in range(n_folders)]
    fn = [[0] * (len(cases) + 1) for _ in range(n_folders)]

    for i in range(n_folders):
        for j in range(len(true_y[i])):
            tmp_tp = int(true_y[i][j] == 1 and pred_y[i][j] == 1)
            tmp_fn = int(true_y[i][j] == 1 and pred_y[i][j] == 0)

            for k, case in enumerate(cases):
                if pisnr[i][j] < case:
                    avg_pisnr[i][k].append(pisnr[i][j])
                    tp[i][k] += tmp_tp
                    fn[i][k] += tmp_fn
                    break
            else:
                avg_pisnr[i][len(cases)].append(pisnr[i][j])
                tp[i][len(cases)] += tmp_tp
                fn[i][len(cases)] += tmp_fn
    
    return avg_pisnr, tp, fn

def calculate_metrics(avg_pisnr, tp, fn, cases):
    x = [[] for _ in range(len(cases) + 1)]
    y = [[] for _ in range(len(cases) + 1)]

    for i in range(len(tp)):
        for k in range(len(cases) + 1):
            x[k] = np.mean(avg_pisnr[i][k]) if avg_pisnr[i][k] else None
            y[k] = tp[i][k] / (tp[i][k] + fn[i][k]) if (tp[i][k] + fn[i][k]) > 0 else None
    
    return x, y

