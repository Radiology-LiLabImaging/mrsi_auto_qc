# Automated Voxel-wise MRSI Quality Labeling

This repository contains sample data, model definiitions, pre-trained models preprocessing parameters, and example code for voxel-wise quality control classification of MR spectra using various models. The models and workflow are based on the methods described in:

> Vaziri S, Liu H, Xie E, Ratiney H, Sdika M, Lupo JM, Xu D, Li Y. **Evaluation of deep learning models for quality control of MR spectra.** *Frontiers in Neuroscience*. 2023;17:1219343. doi:10.3389/fnins.2023.1219343.

The study evaluates machine learning approaches, including random forests, CNNs, and inception CNNs (ICNNs), for automated voxel-wise quality control of 3D MR spectroscopic imaging data.

## Repository Contents

### Data

#### `ndd_sample_100voxels.pkl`

Random sample of 100 voxels from the NDD dataset.

Data are stored as a Pandas DataFrame with the following columns:

| Column          | Description                                   |
| --------------- | --------------------------------------------- |
| `index`         | Voxel number                                  |
| `Subject ID`    | Patient identifier                            |
| `Subject Visit` | Patient visit identifier                      |
| `Subject DC`  | Patient diagnosis                                  |
| `spectrum`      | Real component of the raw voxel spectrum      |
| `imspectrum`    | Imaginary component of the raw voxel spectrum |
| `label`         | Voxel classification label                    |

`spectrum` and `imspectrum` represent the real and imaginary components of the raw voxel spectrum.

#### `example_healthy_pt.pkl`

Sample voxel-wise data from a single healthy patient. This patient is excluded from `ndd_sample_100voxels.pkl`.

Data are stored as a voxel-wise Pandas DataFrame with the following columns:

| Column         | Description                                   |
| -------------- | --------------------------------------------- |
| `index`        | Voxel number                                  |
| `patient_type` | Patient type                                  |
| `row`          | Voxel row position in the full 3D dataset     |
| `column`       | Voxel column position in the full 3D dataset  |
| `slice`        | Voxel slice position in the full 3D dataset   |
| `spectrum`     | Real component of the raw voxel spectrum      |
| `imspectrum`   | Imaginary component of the raw voxel spectrum |
| `label`        | Voxel classification label                    |

The `row`, `column`, and `slice` columns specify each voxel's position in the full 3D dataset.


### Models

#### `models.py`

Contains the definitions of the CNN and ICNN models implemented in PyTorch.

* **`CNN_QC`** — CNN model for voxel classification.
* **`InceptionBlock`** — Inception blocks used by the ICNN architecture.
* **`ICNN_QC`** — ICNN model for voxel classification.


### Training and Example Code

#### `training_example.ipynb`

Jupyter Notebook demonstrating how to:

* Preprocess the NDD dataset.
* Train the CNN and ICNN models.
* Run the main training loop.
* Forward learned CNN/ICNN features to a Random Forest classifier.
* Load pretrained CNN model parameters trained on the full NDD dataset.
* Run voxel-wise label estimation on a sample patient dataset (`example_healthy_pt.pkl`).
* Visualize correctly and incorrectly classified voxels.


### Pretrained Model

#### `model_cnn.ckpt`

Pretrained parameters for the CNN model trained on the full NDD dataset, excluding the patient included in `example_healthy_pt.pkl`.

The model parameters can be loaded into the `models.CNN_QC()` class.


#### Preprocessing Parameters

##### `preprocessing_cnn.pkl`

Contains the preprocessing parameters used during training of `model_cnn.ckpt`, including:

* StandardScaler parameters
* OneHotEncoder parameters

These parameters should be used when preprocessing new data to ensure they are consistent with training data.

## Workflow

The typical workflow is:

1. Load voxel-wise data from a `.pkl` file.
2. Apply the preprocessing parameters stored in `preprocessing_cnn.pkl`.
3. Load the pretrained CNN model from `model_cnn.ckpt`.
4. Run the model to estimate voxel labels.
5. Visualize the resulting voxel classifications.

See `training_example.ipynb` for a complete example.

## Reference

Vaziri S, Liu H, Xie E, Ratiney H, Sdika M, Lupo JM, Xu D, Li Y. **Evaluation of deep learning models for quality control of MR spectra.** *Frontiers in Neuroscience*. 2023;17:1219343. https://doi.org/10.3389/fnins.2023.1219343
v
