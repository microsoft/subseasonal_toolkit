# Subseasonal Forecasting Toolkit

The `subseasonal_toolkit` package provides implementations of the subseasonal forecasting toolkit models, machine learning models, and meteorological baselines presented in the preprint

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

## Getting Started

- Install the subseasonal toolkit package: `pip install subseasonal-toolkit`
- Define the environment variable `$SUBSEASONALDATA_PATH` to point to your desired data directory; any data files needed by a model will be read from, saved to, or synced with this directory

## Generating Model Forecasts

The following examples demonstrate how to generate contiguous US forecasts for the target dates evaluated in "Learned Benchmarks for Subseasonal Forecasting" using each implemented model.

- AutoKNN:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m autoknn`
- CFSv2++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -u -b -m cfsv2pp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -tu -m cfsv2pp`
- Climatology:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m climatology`
- Climatology++:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -u -b -m climpp`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -tu -m climpp`
- Debiased CFSv2:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m deb_cfsv2`
- Debiased ECMWF Control and Ensemble:
  - First, select your desired source (control or ensemble) for debiasing and forecasting in `subseasonal_toolkit/models/deb_ecmwf/selected_submodel.json` by setting the `forecast_with` and `debias_with` keys as described in `deb_ecmwf.ipynb`.  
  - Then, run the selected model: `python -m subseasonal_toolkit.generate_predictions -t std_ecmwf -e -u -m deb_ecmwf`
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
- Online Ensemble:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m online_learning`
- Persistence:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m persistence`
- Persistence++:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m perpp`
- Prophet:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m prophet`
- Salient2:
  - First generate predictions for each model configuration
  `python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -u -b -m salient2`
  - Then select a model configuration using the tuner
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -tu -m salient2`
- Uniform Ensemble:
  `python -m subseasonal_toolkit.generate_predictions -t std_paper -u -m linear_ensemble`

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
