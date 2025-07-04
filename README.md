This repository contains the implementation of our [extended paper](http://doi.org/10.21203/rs.3.rs-4883147/v1) _"Advancing Training Stability in Unsupervised SEM Image Segmentation for IC Layout Extraction"_ based on the original [short paper](https://doi.org/10.1145/3605769.3624000) _"Towards Unsupervised SEM Image Segmentation for IC Layout Extraction"_.
The full dataset is available [here](https://doi.org/10.17617/3.HY5SYN) and the source code for the original paper can be found [here](https://github.com/emsec/unsupervised-ic-sem-segmentation).

# Setup
Tested on Python 3.8.20 with `torch` 1.12.1.

Next to the dependencies from `requirements.txt` (install using `pip install -r requirements.txt`),
this project requires [`torch`](https://pytorch.org/) with a matching CUDA installation, as well as OpenCL at least in version 2.0.

For more information on the parameters required by the scripts below, please run them with `--help`.

# Preparing the Dataset
The scripts for preparing the dataset for supervised and unsupervised training (our approach) can be found in the folder `prepare_dataset`.

### `gen_bin_track_labels.py`
Creates binary track labels from the SVG labels in our dataset for supervised training and evaluating both approaches.

### `split_patches.py`
Splits the SEM images from our dataset linked above and the generated binary track labels into 512x512 px patches used for supervised and unsupervised training.
The patches are saved in two subfolders, `image` and `label`.

### `gen_pseudo_masks.py`
Creates 3 channel pseudo-masks for unsupervised or 1 channel ones for supervised training from the SEM image patches using the conventional segmentation algorithms described in our paper.

### `gen_gradients.py`
Creates the image gradients used for unsupervised decoder training from the SEM image patches.

### `train_val_test_splitter.py`
Creates three text files that split the patches from the dataset into train, validation, and test sets. Each text file contains lists of patch filenames.
Use the parent folder of the `image` directory containing the SEM image patches as dataset directory.

# Training
The folder `train` contains the scripts for both supervised and unsupervised training, with validation after each epoch. They require the dataset to be prepared using the scripts from the previous section, including text files describing for the train and validation split.

## Unsupervised Approach
The following scripts implement our improved unsupervised approach from the extended paper.

### `train_decoder.py`
Trains the decoder network on the generated pseudo-mask patches using image gradient patches as labels.

### `pretrain_encoder.py`
Pre-trains the encoder network in isolation on the SEM image patches using the generated pseudo-masks as labels.

### `train_encoder.py`
Trains the encoder network on the SEM image patches using the decoder network and gradient patches to compute a loss function.

## Supervised Training

### `train_supervised.py`
Trains one of three supervised network architectures either on binary track label or binary pseudo-mask patches, as described in our extended paper.
Use the parent folder of the `image` directory containing the SEM image patches as dataset directory.

# Evaluation
The scripts in the `test` folder evaluate the trained models on the test set from the dataset split, save the results, report the test loss, and compute ESD errors and per-pixels metrics from the predictions.

### `test_decoder.py`
Predicts the image gradients from pseudo-mask patches and reports the loss compared to the ground-truth gradient patches. Running this script is not required to evaluate our approach.

### `test_encoder.py`
Uses trained encoder and decoder networks to predict masks and image gradients from SEM image patches. Reports the encoder loss wrt. the predicted gradient patches.

### `test_supervised.py`
Predicts binary track labels from one or more trained supervised models on the test set. Currently, does not report test loss.

### `compute_errors.py`
Computes per-pixel metrics and ESD errors from 3 channel (unsupervised) or binary (supervised) mask patches. Saves the error metrics in a file and can draw ESD error visualizations.

# Utils

### `plot_histogram.py`
This script from the `utils` folder can compute the histogram of all SEM images in the dataset. We used it to determine track and via thresholds for the conventional segmentation algorithms.

# Academic Context
If you want to cite the work please don't hesitate to cite the [extended paper preprint](http://doi.org/10.21203/rs.3.rs-4883147/v1).
```latex
@article{2024rothaug,
  title = {Advancing Training Stability in Unsupervised SEM Image Segmentation for IC Layout Extraction},
  url = {http://doi.org/10.21203/rs.3.rs-4883147/v1},
  DOI = {10.21203/rs.3.rs-4883147/v1},
  journal = {Journal of Cryptographic Engineering},
  publisher = {Springer Science and Business Media LLC},
  author = {Rothaug, Nils and Cheng, Deruo and Klix, Simon and Auth, Nicole and B\"{o}cker, Sinan and Puschner, Endres and Becker, Steffen and Paar, Christof},
  year = {2024},
  month = {08},
  note = {under submission}
}
```
