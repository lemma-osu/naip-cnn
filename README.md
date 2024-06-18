# NAIP - CNN

Predicting moderate resolution forest attributes (e.g. cover and height) from high-resolution NAIP aerial imagery using ConvNets.

## Usage

See [CONTRIBUTING.md](CONTRIBUTING.md) for Docker setup instructions.

### Generating Training Data

Training data is sampled from NAIP imagery and LiDAR acquisitions stored on Google Earth Engine. Run `notebooks/01_generate_training_data.ipynb` to export and process tabular training data.

### Model Training

Edit `src/naip_cnn/cli/config.py` to specify the training data and model parameters. Run the following command to train a model:

```bash
python -m naip_cnn.cli train 
```

The trained model, configuration, and summary metrics will be logged to Weights & Biases.

### Inference

To predict raster attributes from a trained model, first download the corresponding NAIP as a TFRecord dataset by running `notebooks/02_export_naip.ipynb`. Then run:

```bash
python -m naip_cnn.cli predict [MODEL_PATH] [DATASET_ID] [DATASET_YEAR]
```

