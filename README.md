# NAIP - CNN

Predicting moderate resolution forest attributes (e.g. cover and height) from high-resolution NAIP aerial imagery using ConvNets.

## Usage

See [CONTRIBUTING.md](CONTRIBUTING.md) for Docker setup instructions.

### Generating Training Data

Training data is sampled from NAIP imagery and LiDAR acquisitions stored on Google Earth Engine. Run `notebooks/01_generate_training_data.ipynb` to export and process tabular training data.

### Model Training

Edit `src/naip_cnn/cli/config.py` to specify the training data and model parameters. Run the following command to train a model:

```bash
naip_cnn train 
```

The trained model, configuration, and summary metrics will be logged to Weights & Biases.

### Inference

To predict raster attributes from a trained model stored on Weights & Biases, first download the corresponding NAIP as a TFRecord dataset (and an optional GeoTIFF mask) by running `notebooks/02_export_naip.ipynb`. Then run:

```bash
naip_cnn predict [MODEL_PATH] [DATASET_ID] [DATASET_YEAR]
```

For example:

```bash
naip_cnn predict aazuspan-team/naip-cnn/dmrpcb14 MALHEUR 2009
```

The output GeoTIFF will be written to `data/pred/`.

If you exported a mask alongside the TFRecord, use the `--apply-mask` option to set a NoData mask in the prediction.