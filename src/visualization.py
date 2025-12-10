import matplotlib.pyplot as plt

def plot_train_val_loss(epochs_for_points, train_losses, val_losses):
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_for_points, train_losses, label="Train loss")
    plt.plot(epochs_for_points, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss vs epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/loss_plot.png")
    plt.close()
