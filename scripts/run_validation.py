from naip_cnn.cli.validate import validate

runs = [
    # cover - combined
    "aazuspan-team/naip-cnn/dmrpcb14",
    # cover - 2009
    "aazuspan-team/naip-cnn/fkx5bh62",
    # cover - 2016
    "aazuspan-team/naip-cnn/5bihh19b",
    # cover - 2020
    "aazuspan-team/naip-cnn/dx2qsyp5",
    # rh95 - combined
    "aazuspan-team/naip-cnn/dhuhb3d9",
    # rh95 - 2009
    "aazuspan-team/naip-cnn/ts0egdg6",
    # rh95 - 2016
    "aazuspan-team/naip-cnn/jsyizfiw",
    # rh95 - 2020
    "aazuspan-team/naip-cnn/s4a6mnwt",
]


validation_acquisitions = [
    "MAL2007",
    "MAL2008_CampCreek",
    "MAL2010",
    "MAL2014",
    "MAL2016_CanyonCreek",
    "MAL2018_Aldrich_UpperBear",
    "MAL2020_UpperJohnDay",
]


if __name__ == "__main__":
    for run in runs:
        for acquisition in validation_acquisitions:
            print(f"Validating {run} with {acquisition}")
            metrics = validate(run, acquisition, dry_run=False)
            print(metrics)
