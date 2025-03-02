from quality_prediction.utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def quality_prediction_data_prep():
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

    check_df(df_complete)

    y = df_train['quality']
    X = df_train.drop('quality', axis=1)

    return X, y, df_test


def main():

    X, y, df_test = quality_prediction_data_prep()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    base_models_qwk(x_train, y_train)

    best_models = hyperparameter_optimization_qwk(x_train, y_train)

    for model_name, model in best_models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        print(f"{model_name} QWK: {quadratic_weighted_kappa(y_test, pred)}")

    cart_fit = best_models['CART'].fit(X, y)
    plot_importance(cart_fit, x_train, 10)

    test_pred = pd.DataFrame()
    test_pred['id'] = df_test.index
    predictions = cart_fit.predict(df_test)
    test_pred['quality'] = np.round(predictions).astype(int)
    test_pred.to_csv("quality_prediction/data/quality_prediction_results.csv", index=False)



if __name__ == "__main__":
    print("Process Started...")
    main()