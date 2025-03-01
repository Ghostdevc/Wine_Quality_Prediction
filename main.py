from quality_prediction.utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Load the Data
df_train = load_train()
df_test = load_test()
df_complete = concat_df_on_y_axis(df_train, df_test)

# EDA
check_df(df_train)


num_cols_without_target = [col for col in df_train.columns if df_train[col].dtype in [int, float] and col not in 'quality']

for col in num_cols_without_target:
    target_summary_with_num(df_train, 'quality', col)

for col in df_train.columns:
    num_summary(df_train, col, plot=True)

outlier_containing_cols = [col for col in num_cols_without_target if check_outlier(df_train, col)]

for col in outlier_containing_cols:
    print(f"{col.upper()}: {grab_outliers(df_train, col)}")

correlation_matrix(df_train, df_train.columns)
