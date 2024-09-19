from time import time
import torch

from sklearn.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR

import os
import torchvision.transforms as transforms


def test(test_loader, model, params, printout=True):
    model_name = params['model_name']
    dataset_name = params["dataset_name"]

    model.eval()

    mal_distances = clean_distances = 0.
    total_mse = total_psnr = cnt = 0
    total_mse_se = total_psnr_se = 0
    total_mse_no = total_psnr_no = 0

    with torch.no_grad():
        for name, x, y in test_loader:
            bs = y.size(0)
            x = x.to(params['device'])

            if params['is_plus']:
                y_rec, _, _, _ = model(x).detach().cpu()
            else:
                y_rec = model(x).detach().cpu()
            x = x.cpu()

            clean_distances += torch.sum(torch.mean(torch.abs(y.view(bs, -1) - y_rec.view(bs, -1))), -1)
            mal_distances += torch.sum(torch.mean(torch.abs(y.view(bs, -1) - x.view(bs, -1))), -1)

            total_mse += MSE(y.view(bs, -1), y_rec.view(bs, -1))
            total_psnr += PSNR(y.view(bs, -1).numpy(), y_rec.view(bs, -1).numpy())

            total_mse_se += MSE(x.view(bs, -1), y_rec.view(bs, -1))
            total_psnr_se += PSNR(x.view(bs, -1).numpy(), y_rec.view(bs, -1).numpy())

            total_mse_no += MSE(x.view(bs, -1), y.view(bs, -1))
            total_psnr_no += PSNR(x.view(bs, -1).numpy(), y.view(bs, -1).numpy())

            cnt += 1

            if printout:
                for i in range(bs):
                    image = transforms.ToPILImage()(y_rec[i])
                    image.save(os.path.join(params['OUT_DIR'], params['TYPE_DIR'], f'{dataset_name}_sanitized_{model_name}' + name[i]))

    print(f'Difference w.r.t. malicious:\t{mal_distances}\nDifference .w.r.t. sanitized:\t{clean_distances}')
    print('Original --> x_mal w.r.t. y_clean')
    print(f'MSE: {(total_mse_no / cnt).item()}, PSRN: {(total_psnr_no / cnt).item()}')

    print('Image preservation --> y_clean w.r.t. x_rec')
    print(f'MSE: {(total_mse / cnt).item()}, PSRN: {(total_psnr / cnt).item()}')

    print('Secret elimination --> x_mal w.r.t. x_rec')
    print(f'MSE: {(total_mse_se / cnt).item()}, PSRN: {(total_psnr_se / cnt).item()}')