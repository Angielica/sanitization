import torch
import torch.nn as nn

class DenoiseLoss(nn.Module):
    def __init__(self, is_plus=False):
        super(DenoiseLoss, self).__init__()
        self.criterion = nn.MSELoss()

        self.is_plus = is_plus

    def forward(self, x_true, y_rec, y_rec0=None, y_rec1=None, y_rec2=None):
        if self.is_plus:
            mse1 = self.criterion(y_rec0.flatten(), x_true.flatten())
            mse2 = self.criterion(y_rec1.flatten(), x_true.flatten())
            mse3 = self.criterion(y_rec2.flatten(), x_true.flatten())
            mse4 = self.criterion(y_rec.flatten(), x_true.flatten())

            return torch.mean(mse1 + mse2 + mse3 + mse4)
        else:
            return self.criterion(x_true.flatten(), y_rec.flatten())

