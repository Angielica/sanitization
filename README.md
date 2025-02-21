## A deep learning-based approach for stegomalware sanitization in digital images

This is the accompanying code for the paper "A deep learning-based approach for stegomalware sanitization in digital images" that has been submitted for possible publication in the Journal of Intelligent Information Systems (JIIS).

The repository contains all code to re-execute our AE-based sanitizer presented in the paper.

The repository contains the following folders:
- config, it contains experiment configurations for each dataset and for each model;
- dataloader, with the code to load the dataset;
- utils, with utility code;
- models, with the source code of each AE-based models presented in the paper.

The main.py file contains the code for running the experiments. You have to pass the experiment configuration:

```
python3 main.py config/{config_name}.json
```
 where config_name is the name of the experimental configuration file.
For example,

```
python3 main.py config/config-UNetPlus.json
 ```
In this case, you will train the UNet+ model using the FFHQ70 dataset.

For the dataset, set "dataset_name" equal to "FFHQ-subset" if you want to test the model on FFHQ70. 
Set the variable equal to "FFHQ-subset-big" if you want to use the dataset "FFHQ210".

For the BSD70 dataset, please use the jupyter notebook used for the paper [1]:
- "UNetPlus_BSD70.ipynb" for training the UNet+;
- "UNet_BSD70.ipynb" for UNet;
- "AE_BSD70.ipynb" for the AE model.

[1] Liguori, A., Zuppelli, M., Gallo, D., Guarascio, M., Caviglione, L. (2024). Erasing the Shadow: Sanitization of Images with Malicious Payloads Using Deep Autoencoders. In: Appice, A., Azzag, H., Hacid, MS., Hadjali, A., Ras, Z. (eds) Foundations of Intelligent Systems. ISMIS 2024. Lecture Notes in Computer Science, vol 14670. Springer, Cham. https://doi.org/10.1007/978-3-031-62700-2_11