import pandas as pd

df = pd.read_csv("grid_manifest.csv")

print(df.query('base_init_stddev == 0.25'))