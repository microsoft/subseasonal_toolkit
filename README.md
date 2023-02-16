# Subseasonal Forecasting Toolkit

The `subseasonal_toolkit` package provides implementations of the subseasonal forecasting ABC model of

[Adaptive Bias Correction for Subseasonal Forecasting](https://arxiv.org/pdf/2209.10666.pdf)  
Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Judah Cohen, Miruna Oprescu, Ernest Fraenkel, and Lester Mackey.  Sep. 2022.

```
@article{
  mouatadid2022adaptive,
  title={Adaptive Bias Correction for Subseasonal Forecasting},
  author={Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Judah Cohen, Miruna Oprescu, Ernest Fraenkel, and Lester Mackey},
  journal={arXiv preprint arXiv:2209.10666},
  year={2022}
}
```

and the machine learning models and meteorological baselines of

[Learned Benchmarks for Subseasonal Forecasting](https://arxiv.org/pdf/2109.10399.pdf)  
Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, and Lester Mackey.  Sep. 2021.

```
@article{
  mouatadid2021toolkit,
  title={Learned Benchmarks for Subseasonal Forecasting},
  author={Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, and Lester Mackey},
  journal={arXiv preprint arXiv:2109.10399},
  year={2021}
}
```

## System Requirements

This package has been tested with the following operating system and Python pairings:
+ macOS Monterey 12.6.3 with Python 3.9.12
+ Linux CentOS 7 with Python 3.7.9

A complete list of Python dependencies can be found in `setup.cfg`; these dependencies are required upon installation.

## Getting Started

- Install the subseasonal toolkit package: `pip install subseasonal-toolkit`
  - Installation completed in under 1 minute with pip 22.2.2 on a 2021 MacBook Pro with 16 GB of RAM running macOS Monterey version 12.6.3.
- Define the environment variable `$SUBSEASONALDATA_PATH` to point to your desired data directory; any data files needed by a model will be read from, saved to, or synced with this directory
- Run the following demo which generates and evaluates Raw CFSv2 precipitation forecasts across the contiguous U.S. for the 2018-2021 `std_paper_forecast` evaluation period of "Adaptive Bias Correction for Subseasonal Forecasting":
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m raw_cfsv2 --task us_tmp2m_1.5x1.5_34w`
  - This demo ran to completion in 7 minutes with Python 3.9.12 on a 2021 MacBook Pro with 16 GB of RAM running macOS Monterey version 12.6.3.
  - Expected outputs 
    - A forecast folder `models/cfsv2pp/submodel_forecasts/cfsv2pp-debiasFalse_years12_margin0_days1-1_leads15-15_lossmse/us_tmp2m_1.5x1.5_34w/` containing daily forecast files from 20180101 through 20211231
    - A metrics folder `eval/metrics/raw_cfsv2/submodel_forecasts/cfsv2pp-debiasFalse_years12_margin0_days1-1_leads15-15_lossmse/us_tmp2m_1.5x1.5_34w/` containing 6 evaluation metrics:
      - `lat_lon_error-us_tmp2m_1.5x1.5_34w-std_paper_forecast.h5`
      - `lat_lon_rmse-us_tmp2m_1.5x1.5_34w-std_paper_forecast.h5`
      - `lat_lon_skill-us_tmp2m_1.5x1.5_34w-std_paper_forecast.h5`
      - `rmse-us_tmp2m_1.5x1.5_34w-std_paper_forecast.h5`
      - `score-us_tmp2m_1.5x1.5_34w-std_paper_forecast.h5`
      - `skill-us_tmp2m_1.5x1.5_34w-std_paper_forecast.h5`

## Generating Model Forecasts

The following examples demonstrate how to generate contiguous US forecasts for the target dates evaluated in "Adaptive Bias Correction for Subseasonal Forecasting" or "Learned Benchmarks for Subseasonal Forecasting" using each implemented model.

- ABC-CCSM4:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m ccsm4`
- ABC-CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m cfsv2`
- ABC-ECMWF:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m ecmwf`
- ABC-FIMr1p1:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m fimr1p1pp`
- ABC-GEFS:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m gefs`
- ABC-GEMS:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m gems`
- ABC-GEOS_v2p1:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m geos_v2p1`
- ABC-NESM:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m nesm`
- ABC-SubX:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -a -m subx_mean`
- AutoKNN:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m autoknn`
- CCSM4++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m ccsm4pp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m ccsm4pp`
- CFSv2++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m cfsv2pp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m cfsv2pp`
- Climatology:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m climatology`
- Climatology++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m climpp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m climpp`
- Debiased CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_cfsv2`
- Debiased ECMWF Control and Ensemble:
  - First, select your desired source (control or ensemble) for debiasing and forecasting in `subseasonal_toolkit/models/deb_ecmwf/selected_submodel.json` by setting the `forecast_with` and `debias_with` keys as described in `deb_ecmwf.ipynb`.
  - Then, run the selected model: `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_ecmwf`
- Debiased LOESS CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_loess_cfsv2`
- Debiased LOESS ECMWF:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_loess_ecmwf`
- Debiased Quantile Mapping CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_quantile_cfsv2`
- Debiased Quantile Mapping ECMWF:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_quantile_ecmwf`
- Debiased SubX:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m deb_subx_mean`
- ECMWF++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m ecmwfpp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m ecmwfpp`
- FIMr1p1++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m fimr1p1pp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m fimr1p1pp`
- GEFS++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m gefspp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m gefspp`
- GEM++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m gempp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m gempp`
- GEOS++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m geos_v2p1pp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m geos_v2p1pp`
- Informer:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m informer`
- LocalBoosting:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -u -b -m localboosting`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -tu -m localboosting`
- MultiLLR:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m multillr`
- N-BEATS:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m nbeats`
- NN-A:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m nn-a`
- Online Ensemble:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m online_learning`
- Persistence:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m persistence`
- Persistence++ CCSM4:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m perpp_ccsm4`
- Persistence++ CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m perpp_cfsv2`
- Persistence++ ECMWF:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m perpp_ecmwf`
- Persistence++ FIMr1p1:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m perpp_fimr1p1 -u -e`
- Persistence++ GEFS:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m perpp_gefs -u -e`
- Persistence++ GEM:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m perpp_gem -u -e`
- Persistence++ GEOS_v2p1:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m perpp_geos_v2p1 -u -e`
- Persistence++ NESM:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m perpp_nesm -u -e`
- Persistence++ SubX:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m perpp_subx_mean -u -e`
- Prophet:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m prophet`
- Raw CCSM4:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m raw_ccsm4`
- Raw CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m raw_cfsv2`
- Raw ECMWF:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m raw_ecmwf`
- Raw FIMr1p1:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m raw_fimr1p1 -u -e`
- Raw GEFS:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m raw_gefs -u -e`
- Raw GEM:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m raw_gem -u -e`
- Raw GEOS_v2p1:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m raw_geos_v2p1 -u -e`
- Raw NESM:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m raw_nesm -u -e`
- Raw SubX:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -m raw_subx_mean -u -e`
- Salient2:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -u -b -m salient2`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -tu -m salient2`
- SubX++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m subx_meanpp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -tu -m subx_meanpp`
- Uniform Ensemble:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -e -u -m linear_ensemble`

## For Developers

### Installation

After cloning this repository, install from source in editable mode using `pip install -e .` in this directory or `pip install -e path/to/directory` from another directory. 

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
