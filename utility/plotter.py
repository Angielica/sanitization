import matplotlib.pyplot as plt

def plot_loss(loss_values, val_loss_values, path):
    num_epochs = len(loss_values)

    plt.figure(figsize=(15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), loss_values[:num_epochs], label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_values[:num_epochs], label='Validation Loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()


def plot_loss_log(loss_values, val_loss_values, path):
    num_epochs = len(loss_values)

    plt.figure(figsize=(15, 10))
    plt.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), loss_values[:num_epochs], label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_values[:num_epochs], label='Validation Loss')
    plt.yscale('log')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center')
    plt.savefig(path)
    plt.show()