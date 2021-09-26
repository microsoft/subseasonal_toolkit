# import pandas as pd
import os
import catboost
from argparse import ArgumentParser
# import matplotlib.pyplot as plt
# import shap
# import random
# import re

from subseasonal_toolkit.utils.general_util import string_to_dt, tic, toc, make_directories
from subseasonal_toolkit.models.localboosting.utils import load_train_data
# from subseasonal_toolkit.models.localboosting.attributes import get_submodel_name
# from subseasonal_toolkit.models.localboosting.utils import get_feature_importance_plot


# Load command line arguments
parser = ArgumentParser()
parser.add_argument(
    "pos_vars", nargs="*",  # choices=["contest_tmp2m", "contest_precip", "34w", "56w"]
)  # gt_id and horizon
# parser.add_argument("--n_features", "-nf", default="all")
# parser.add_argument("--first_train_date", "-f", default="19800101")
# parser.add_argument("--last_train_date", "-l", default="20161231")
# parser.add_argument("--iterations", "-i", default=100, type=int)
# parser.add_argument("--depth", "-d", default=2, type=int)
# parser.add_argument("--learning_rate", "-lr", default=0.17, type=float)
# parser.add_argument("--number_of_validation_years", "-vy", default=1, type=int)
args = parser.parse_args()


print(f"\nSet variables.")
tic()
gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
n_features = "all"
# region_extension = 3
first_train_date = string_to_dt("19800101")
last_train_date = string_to_dt("20001231")
iterations = 50
depth = 2
learning_rate = 0.17
number_of_validation_years = 3

model_name = "localboosting"
# submodel_name = get_submodel_name(
#     region_extension,
#     n_features,
#     margin_of_days,
#     iterations,
#     depth,
#     learning_rate,
# )
output_folder = os.path.join("models",model_name,"model_analysis",f"{gt_id}_{horizon}")
make_directories(output_folder)
toc()

print(f"\nLoad train data.")
tic()
X_train, y_train, X_val, y_val = load_train_data(
    gt_id, horizon,
    first_train_date,
    last_train_date,
    number_of_validation_years,
    n_features
)
toc()

print("\nCreate train and val Pools.")
tic()
train_pool = catboost.Pool(X_train, y_train)
val_pool = catboost.Pool(X_val, y_val)
toc()

print(f"\nRun {model_name} model.")
tic()
model = catboost.CatBoostRegressor(
    iterations=iterations,
    depth=depth,
    learning_rate=learning_rate,
    loss_function="RMSE",
    random_seed=123,
    eval_metric="RMSE",
    verbose=True,)
# trained_model_folder = f"models/{model_name}/trained_submodels/{gt_id}_{horizon}"
# trained_model_filename = f"{submodel_name}.model"
# model.load_model(f"{trained_model_folder}/{trained_model_filename}")
model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
toc()

print(f"\n\nFeature importance")

print(f"\nFeature importance by prediction values change")
tic()
pvc_pretty = model.get_feature_importance(
    type="PredictionValuesChange", prettified=True
)
with open(
    f"{output_folder}/list-feature_importance-predictions_values_change.txt", "w+"
) as f:
    for feature in pvc_pretty["Feature Id"]:
        f.write(str(feature) + "\n")
# pvc = model.get_feature_importance(type="PredictionValuesChange")
# pvc_n_features = 30
#
# print("\nSaving prediction_values_change plot")
# get_feature_importance_plot(
#     pvc,
#     method_name="PredictionValuesChange",
#     X_train=X_train,
#     output_folder=output_folder,
#     n_features=15,
# )
# toc()

# # print(f"\nFeature importance by loss function change")
# # tic()
# # lfc_pretty = model.get_feature_importance(
# #     data=val_pool, type="LossFunctionChange", prettified=True
# # )
# # with open(
# #     f"{output_folder}/list-feature_importance-loss_function_change.txt", "w"
# # ) as f:
# #     for feature in lfc_pretty["Feature Id"]:
# #         f.write(str(feature) + "\n")
# # lfc = model.get_feature_importance(data=val_pool, type="LossFunctionChange")
# # lfc_n_features = 30
# # get_feature_importance_plot(
# #     lfc,
# #     method_name="LossFunctionChange",
# #     X_train=X_train,
# #     output_folder=output_folder,
# #     n_features=15,
# # )
# # toc()
#
# print(f"\nFeature importance by Shap values")
#
# print(f"\nGet Shap values")
# tic()
# shap_values = model.get_feature_importance(data=train_pool, type="ShapValues")
#
# expected_value = shap_values[0, -1]  # mean shap value is in last entry
# shap_values = shap_values[:, :-1]  # same shape of X_train: n_obs x features
# toc()
#
# print(f"Plot Shap feature importance")
# tic()
# fig, ax = plt.subplots()  # 20 minutes to run this plot
# shap.summary_plot(shap_values, X_train, show=False)
# ax.set_title("Shap Feature Importance", fontsize=16)
# fig.savefig(f"{output_folder}/shap-summary_plot.png", bbox_inches="tight", dpi=150)
# plt.close()
# toc()
#
# tic()
# fig, ax = plt.subplots()
# shap.summary_plot(shap_values, X_train, show=False, plot_type="bar")
# ax.set_title("Shap Feature Importance", fontsize=16)
# fig.savefig(f"{output_folder}/shap-summary_barplot.png", bbox_inches="tight", dpi=150)
# plt.close()
# toc()
#
# print(f"Plot Shap dependence plot for seven highest features")
# tic()
# for feature in pvc_pretty["Feature Id"][:7]:
#     tic()
#     plt.figure()
#     shap.dependence_plot(feature, shap_values, X_train, show=False)
#     plt.title(f"Shap Feature Importance ({feature})", fontsize=16)
#     plt.savefig(
#         f"{output_folder}/shap-dependence_plot-{feature}.png",
#         bbox_inches="tight",
#         dpi=150,
#     )
#     plt.close()
#     toc()
# toc()
#
# print(f"Plot Shap force plot for single observation")
# tic()
# plt.figure()
# observation_to_plot = 1
# shap.force_plot(
#     expected_value,
#     shap_values[observation_to_plot, :],
#     X_train.iloc[observation_to_plot, :],
#     show=False,
#     matplotlib=True,
# )
# plt.title(
#     f"Shap Feature Values for Observation {observation_to_plot}", fontsize=16, pad=120
# )
# plt.savefig(
#     f"{output_folder}/shap-force_plot-observation_{observation_to_plot}.png",
#     bbox_inches="tight",
# )
# plt.close()
# toc()
#
# print(f"Save Shap force plot for all observations")
# tic()
# random.seed(0)
# number_of_observations = 20
# subsample_index = random.sample(range(len(X_train)), number_of_observations)
# force_plot = shap.force_plot(
#     expected_value, shap_values[subsample_index], X_train.iloc[subsample_index]
# )
# shap.save_html(f"{output_folder}/shap-force_plot-all.html", force_plot)
# toc()
#
# print(f"Plot Shap waterfall plot for single observation")
# tic()
# plt.figure()
# observation_to_plot = 1
# shap.waterfall_plot(
#     expected_value,
#     shap_values[observation_to_plot, :],
#     feature_names=X_train.columns.values,
#     max_display=5,
#     show=False,
# )
# plt.title(f"Shap Feature Values for Observation {observation_to_plot}", fontsize=16)
# plt.savefig(
#     f"{output_folder}/shap-waterfall-observation_{observation_to_plot}.png",
#     bbox_inches="tight",
# )
# plt.close()
# toc()
#
# print(f"Plot Shap decision plot")
# tic()
# random.seed(10)
# number_of_observations = 20
# number_of_features_to_plot = 7
# subsample_index = random.sample(range(len(X_train)), number_of_observations)
# plt.figure()
# shap.decision_plot(
#     expected_value,
#     shap_values[subsample_index],
#     X_train.columns,
#     feature_display_range=slice(-1, -number_of_features_to_plot, -1),
#     show=False,
# )
# plt.title(f"Shap Decision Plot (logit)", fontsize=16, pad=30)
# plt.savefig(f"{output_folder}/shap-decision_plot.png", bbox_inches="tight")
# plt.close()
# toc()
#
#
# print(f"\nFeature importance by interaction")
# tic()
# plt.figure()
# interaction_pretty = model.get_feature_importance(
#     data=train_pool, type="Interaction", prettified=True
# )
# interaction = model.get_feature_importance(data=train_pool, type="Interaction",)
# interaction_n_features = 30
# interaction_pretty["First Feature"] = [
#     X_train.columns[i] for i in interaction_pretty["First Feature Index"]
# ]
# interaction_pretty["Second Feature"] = [
#     X_train.columns[i] for i in interaction_pretty["Second Feature Index"]
# ]
# interaction_pretty[["First Feature", "Second Feature", "Interaction"]]
# interaction_list = []
# for k, item in enumerate(interaction):
#     first = X_train.dtypes.index[interaction[k][0]]
#     second = X_train.dtypes.index[interaction[k][1]]
#     if first != second:
#         interaction_list.append([first + "_" + second, interaction[k][2]])
# interaction_df = pd.DataFrame(interaction_list, columns=["Feature-Pair", "Score"])
# feature_score = interaction_df.sort_values(
#     by="Score", ascending=False, inplace=False, kind="quicksort", na_position="last"
# )
# feature_score = feature_score[:interaction_n_features]
# plt.rcParams["figure.figsize"] = (16, 7)
# ax = feature_score.plot("Feature-Pair", "Score", kind="bar", color="c")
# ax.set_title("Pairwise Feature Importance", fontsize=14)
# ax.set_xlabel(None)
# plt.savefig(
#     f"{output_folder}/interaction-pairwise_feature_importance.png", bbox_inches="tight"
# )
# plt.close()
# toc()
#
# # print(f"\nPlot feature errors on bins")
# # tic()
# # features = [f"{gt_id.split('_')[1]}_clim"]
# # for feature in features:
# #     model.calc_feature_statistics(
# #         X_val,
# #         y_val,
# #         feature,
# #         prediction_type="Probability",
# #         plot_file=f"{output_folder}/feature_stats-{feature}.html",
# #         plot=False,
# #     )
# # toc()
#
# # print(f"\nPlot first three trees")
# # tic()
# # for i in range(3):
# #     fig = model.plot_tree(tree_idx=i, pool=train_pool)
# #     fig.render(f"{output_folder}/tree-{i}", format="png")
# #     os.remove(f"{output_folder}/tree-{i}")
# # toc()
#
#
# # print(f"\n\nObject importance")
# #
# # print(f"\nPlot force_plot of three most negative objects")
# # tic()
# # # Get the three datapoints in training set that most negatively impact the KS metric
# # observations_min, _ = model.get_object_importance(
# #     train_pool,
# #     train_pool,
# #     top_size=3,
# #     thread_count=-1,
# #     importance_values_sign="Negative",
# # )
# # for i, observation in enumerate(observations_min):
# #     plt.figure()
# #     shap.force_plot(
# #         expected_value,
# #         shap_values[observation, :],
# #         feature_names=X_train.columns.values,
# #         matplotlib=True,
# #         show=False,
# #         link="logit",
# #     )
# #     plt.title(
# #         f"Shap Feature Values for Observation {observation} (Probability)",
# #         fontsize=16,
# #         pad=120,
# #     )
# #     plt.savefig(
# #         f"{output_folder}/shap-force_plot-min_{i}_observation_{observation}.png",
# #         bbox_inches="tight",
# #     )
# #     plt.close()
# # toc()
# #
# # print(f"\nPlot force_plot of three most positive objects")
# # tic()
# # # Get the three datapoints in training set that most positively impact the KS metric
# # observations_max, _ = model.get_object_importance(
# #     train_pool,
# #     train_pool,
# #     top_size=3,
# #     thread_count=-1,
# #     importance_values_sign="Positive",
# # )
# #
# # for i, observation in enumerate(observations_max):
# #     plt.figure()
# #     shap.force_plot(
# #         expected_value,
# #         shap_values[observation, :],
# #         feature_names=X_train.columns.values,
# #         matplotlib=True,
# #         show=False,
# #         link="logit",
# #     )
# #     plt.title(
# #         f"Shap Feature Values for Observation {observation} (Probability)",
# #         fontsize=16,
# #         pad=120,
# #     )
# #     plt.savefig(
# #         f"{output_folder}/shap-force_plot-max_{i}_observation_{observation}.png",
# #         bbox_inches="tight",
# #     )
# #     plt.close()
# # toc()
