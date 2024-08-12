# HydroNODE

NeuralODE models for hydrology

marvin.hoege@eawag.ch

Code repo for publication at https://hess.copernicus.org/articles/26/5085/2022/. Cite when using the code.


### Installation
Install [Julia](https://julialang.org/downloads/)

All required packages are installed automatically in a seperate environment (see https://pkgdocs.julialang.org/v1/toml-files/) when `HydroNODE.jl` is executed for the first time.

### Data
- download `CAMELS time series meteorology, observed flow, meta data (.zip)` from https://ral.ucar.edu/solutions/products/camels
- unzip and refer to folder `basin_dataset_public_v1p2` as `data_path` in `HydroNODE_main.jl`

### Train models
- Set user specific settings like `basin_id` in `HydroNODE_main.jl`
  and execute it.
