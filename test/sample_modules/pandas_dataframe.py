import pandas as pd

my_data = pd.read_csv('../resources/15pass-normalization.csv', delimiter=",")
my_data_test = my_data["WICPPHPB"].values

print(my_data_test)
