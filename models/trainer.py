import numpy as np
import torch
from time import time
import sys

from models.losses import DenoiseLoss
from utility.plotter import plot_loss, plot_loss_log
class Trainer:
    def __init__(self, model, params):
        self.device = params['device']
        self.params = params
        self.model = model
        self.model = self.model.to(self.device)

        self.is_plus = params['is_plus']

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_criterion = DenoiseLoss(is_plus=self.is_plus)

    def step(self, x, y):
        self.model.zero_grad()

        if self.is_plus:
            y_rec, y_rec0, y_rec1, y_rec2 = self.model(x)
            loss_batch = self.loss_criterion(y, y_rec, y_rec0, y_rec1, y_rec2)
        else:
            y_rec = self.model(x)
            loss_batch = self.loss_criterion(y_rec, y)

        loss_batch.backward()
        self.optimizer.step()

        return loss_batch

    def eval(self, data_loader):
        self.model.eval()

        loss = 0.
        for _, x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                if self.is_plus:
                    y_rec, y_rec0, y_rec1, y_rec2 = self.model(x_batch)
                    loss_batch = self.loss_criterion(y_batch, y_rec, y_rec0, y_rec1, y_rec2)
                else:
                    y_rec = self.model(x_batch)
                    loss_batch = self.loss_criterion(y_rec, y_batch)

                loss += loss_batch

        return loss / len(data_loader)

    def train(self, tr_loader, ts_loader, batch_size=32, num_epochs=10, step=20):
        losses = np.zeros(num_epochs)
        tr_losses = np.zeros(num_epochs)
        te_losses = np.zeros(num_epochs)

        best_loss = np.Inf

        total_steps = (len(tr_loader.dataset) // batch_size)  # *num_epochs
        print("[INFO] Starting training phase...")
        start = time()

        try:
            step_count = 0
            for epoch in range(num_epochs):
                i = 0
                self.model.train()
                for _, x_batch, y_batch in tr_loader:
                    i += 1
                    step_count += 1
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    loss = self.step(x_batch, y_batch)

                    losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss.item() * (1. / (i + 1.))

                    if (i + 1) % step == 0:
                        sys.stdout.write(
                            '\nEpoch [{:>3}/{}] | Step [{:>3}/{}]| loss: {:.4f} |'
                            .format(epoch + 1, num_epochs, i + 1, total_steps, loss.item()))
                        sys.stdout.flush()

                #                tr_losses[epoch] = self.eval(tr_loader)
                te_losses[epoch] = self.eval(ts_loader)
                if te_losses[epoch] < best_loss:
                    best_loss = te_losses[epoch]
                    # Save the model checkpoints
                    torch.save(self.model.state_dict(), self.params['best_path_model'])
                sys.stdout.write(
                    '\nEND Epoch [{:>3}/{}] | Train loss: {:.4f} | Test loss: {:.4f} '
                    .format(epoch + 1, num_epochs, losses[epoch], te_losses[epoch]))
                sys.stdout.flush()

        except KeyboardInterrupt:
            print('\n')
            print('-' * 89)
            print('[INFO] Exiting from training early')
        print(f'\n[INFO] Training phase... Elapsed time: {(time() - start):.0f} seconds\n')

        torch.save(self.model.state_dict(), self.params['last_path_model'])

        plot_loss(losses[:epoch], te_losses[:epoch], self.params['path_training_loss'])
        plot_loss_log(losses[:epoch], te_losses[:epoch], self.params['path_training_loss_log'])