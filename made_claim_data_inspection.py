import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


dat = pd.read_csv("part2_training_data.csv")


features = dat.drop(columns=["claim_amount", "made_claim"]).to_numpy()
target = dat["made_claim"].to_numpy()

print(np.unique(target, return_counts=True)[1])
#
# scaler = StandardScaler().fit(features)
# rescaled_features = scaler.transform(features)
# np.set_printoptions(precision=3)
# #
# # print(rescaled_features)
# # Find NAN values
# # for feature in features.transpose():
# #     print(sum(np.isnan(feature)))
#
# df_rescaled = pd.DataFrame(rescaled_features, columns=list(dat.drop(columns=["claim_amount", "made_claim"]).columns.values))
# # print(df_rescaled.describe())
# df_rescaled["made_claim"] = target
# df_rescaled["made_claim"] = df_rescaled["made_claim"].astype("category")
# df_rescaled["made_claim"] = df_rescaled["made_claim"].cat.codes
# #
# print("\nrescaled data description\n", df_rescaled.describe())
# print("data types of columns", df_rescaled.dtypes)
# #
# print("")
# features = df_rescaled.values[:, 0:9].astype(float)
# target = df_rescaled.values[:, 9].astype(int)
#
# print(df_rescaled.shape)
# print("features dtype:", features.dtype, "and shape", features.shape)
# print("target dtype:", target.dtype, "and shape", target.shape)
#
# print("\nLabel counts:\n", df["made_claim"].value_counts())



