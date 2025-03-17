#from quality_prediction.utils import load_train, load_test, check_df, concat_df_on_y_axis, target_summary_with_num, num_summary, check_outlier
#from quality_prediction.utils import grab_outliers, correlation_matrix, base_models_qwk, hyperparameter_optimization_qwk, quadratic_weighted_kappa, plot_importance, train_test_split, warnings, ConvergenceWarning, np, pd
from quality_prediction.utils import *
from scipy.special import boxcox1p
from scipy.stats import normaltest
from scipy.optimize import minimize

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
    check_df(df_complete)
    df_train.columns

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
    
    # Outliers
    print('*------------------------*')

    for col in num_cols_without_target:
        print(col, check_outlier(df_complete, col))

    for col in num_cols_without_target:
        replace_with_thresholds(df_complete, col)

    print('------------------------')

    for col in num_cols_without_target:
        print(col, check_outlier(df_complete, col))

    print('*------------------------*')


    # Feature Engineering
    df_complete["NEW_volatile_acidity_to_fixed_acidity"] = df_complete["volatile acidity"] / df_complete["fixed acidity"]
    df_complete["NEW_acidity_to_ph"] = ((df_complete["fixed acidity"] + df_complete["volatile acidity"]) / df_complete["pH"])
    df_complete["NEW_free_sulfur_dioxide_to_total_sulfur_dioxide"] = df_complete["free sulfur dioxide"] / df_complete["total sulfur dioxide"]
    df_complete["NEW_total_sulfur_dioxide_to_density"] = df_complete["total sulfur dioxide"] / df_complete["density"]
    df_complete["NEW_sugar_to_density"] = df_complete["residual sugar"] / df_complete["density"]
    df_complete["NEW_alcohol_times_sugar"] = df_complete["alcohol"] * df_complete["residual sugar"]
    df_complete["NEW_chlorides_times_fixed_acidity"] = df_complete["chlorides"] * df_complete["fixed acidity"]
    df_complete["NEW_sulphates_times_alcohol"] = df_complete["sulphates"] * df_complete["alcohol"]


    cat_cols, num_cols, cat_but_car = grab_col_names(df_complete)
    print(f"cat_cols: {cat_cols}, num_cols: {num_cols}, cat_but_car: {cat_but_car}")
    
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    df_complete[num_cols] = pd.DataFrame(scaler.fit_transform(df_complete[num_cols]))


    # PCA
    from sklearn.decomposition import PCA

    # PCA için kalite değişkenini çıkartıyoruz
    X_pca = df_complete.drop(columns=["quality"])

    # PCA modeli (bütün bileşenleri alarak başlıyoruz)
    pca = PCA(n_components=X_pca.shape[1])
    X_pca_transformed = pca.fit_transform(X_pca)

    # Açıklanan varyans oranlarını inceleyelim
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # İlk kaç bileşen kaç % varyansı açıklıyor?

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Bileşen Sayısı')
    plt.ylabel('Açıklanan Varyans Oranı')
    plt.title('PCA Açıklanan Varyans Oranı')
    plt.grid()
    plt.show()

    # İlk 3 bileşeni seçelim
    num_pca_components = 3
    df_pca = pca.transform(X_pca)[:, :num_pca_components]

    # Yeni bileşenleri veri setine ekleyelim
    df_complete["pca_1"] = df_pca[:, 0]
    df_complete["pca_2"] = df_pca[:, 1]
    df_complete["pca_3"] = df_pca[:, 2]

    # PCA bileşenlerinin hangi değişkenlerle ilişkili olduğunu görelim
    pca_components = pd.DataFrame(pca.components_[:num_pca_components], 
                                columns=X_pca.columns, 
                                index=[f"PCA_{i+1}" for i in range(num_pca_components)])
    print(pca_components.T)

    df_complete[num_cols] = pd.DataFrame(scaler.inverse_transform(df_complete[num_cols]))

    check_df(df_complete)

    cat_cols, num_cols, cat_but_car = grab_col_names(df_complete)

    df_scaled = df_complete.copy()
    
    #normality test
    for col in num_cols:
        if col != "quality":
            print(col, normaltest(df_scaled[col], nan_policy='raise'))
    
    for col in num_cols:
        check_skew(df_scaled, col)

    # Şiddetli çarpık değişkenler için log veya Box-Cox
    for col in ["residual sugar", "total sulfur dioxide", "sulphates", 
                "NEW_total_sulfur_dioxide_to_density", "NEW_sugar_to_density", 
                "NEW_alcohol_times_sugar"]:
        df_scaled = transform_feature(df_scaled, col, method="boxcox")

    # Orta derecede çarpık değişkenler için Yeo-Johnson
    for col in ["fixed acidity", "free sulfur dioxide", "alcohol",
                "NEW_acidity_to_ph", "NEW_chlorides_times_fixed_acidity",
                "NEW_sulphates_times_alcohol"]:
        df_scaled = transform_feature(df_scaled, col, method="yeojohnson")
    
    skewed_cols = [
        "chlorides", "NEW_volatile_acidity_to_fixed_acidity",
        "NEW_free_sulfur_dioxide_to_total_sulfur_dioxide",
        "pca_1", "pca_2", "pca_3"
    ]

    for col in skewed_cols:
        skewness = df_scaled[col].skew()
        
        if abs(skewness) > 1:
            method = "log" if (df_scaled[col] > 0).all() else "yeojohnson"
        elif abs(skewness) > 0.5:
            method = "boxcox" if (df_scaled[col] > 0).all() else "yeojohnson"
        else:
            continue  # Dönüşüm gerekmiyor
    
        df_scaled = transform_feature(df_scaled, col, method)

    for col in num_cols:
        check_skew(df_scaled, col)

    check_df(df_scaled)

    df_train_processed = df_scaled[df_scaled["quality"].notnull()]

    for col in num_cols:
        num_summary(df_train_processed, col, True)

    #X = df_train_processed[["pca_1", "pca_2", "pca_3"]]
    #y = df_train_processed["quality"]
    #df_test_processed = df_scaled[df_scaled["quality"].isnull()][["pca_1", "pca_2", "pca_3"]]
    #linear_model = LinearRegression()
    #linear_model.fit(X, y)

    # qwk cross validation

    #qwk_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)

    #def qwk_cross_val(model, X, y, cv=5):
        #qwk_scores = cross_val_score(model, X, y, cv=cv, scoring=qwk_scorer)
        #return qwk_scores.mean()
    
    #print(qwk_cross_val(linear_model, X, y))
    
    #base_models_qwk(X, y)


    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit_predict(df_train_processed[num_cols])
    df_scores = lof.negative_outlier_factor_
    
    print(np.sort(df_scores)[0:5])

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 10], style='.-')
    plt.show()
    th = np.sort(df_scores)[3]

    print(df_train_processed[df_scores < th].shape)

    df_train_processed[df_scores < th].drop(axis=0, labels=df_train_processed[df_scores < th].index)


    check_df(df_scaled)

    y = df_train_processed['quality']
    X = df_train_processed.drop('quality', axis=1)

    df_test_scaled = df_scaled[df_scaled["quality"].isnull()].drop('quality', axis=1)

    rf_model = RandomForestRegressor(random_state=46, n_jobs=-1).fit(X, y)

    importances = plot_importance(rf_model, X, num=30)


    return X, y, df_test_scaled


def main():

    X, y, df_test = quality_prediction_data_prep()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    base_models_qwk(x_train, y_train)
    #base_models_qwk(X, y)
    #best_models = hyperparameter_optimization_qwk(X, y)

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