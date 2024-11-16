# Optunaの目的関数
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

def objective(trial,df_x_train,df_y_train,df_x_test,df_y_test):
    #set_config
    param = {
    'objective': 'regression',  # 回帰問題の場合、目的関数として'regression'を指定
    'metric': 'rmse',  # モデル評価の指標としてRMSE（Root Mean Squared Error）を使用
    'boosting_type': 'gbdt',  # ブースティングのアルゴリズムとして勾配ブースティング（GBDT）を使用
    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),  # 学習率、ログスケールで探索。0.01から0.1の範囲
    'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # 木の葉の数。モデルの複雑さを決定（一般的に、数が多いほど過学習しやすい）
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),  # 葉に含まれる最小サンプル数。これを増やすことで過学習を防ぐ
    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),  # 特徴量のサンプリング率。0.4～1.0の範囲でランダムに特徴量をサンプリング
    'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),  # バギングのサンプリング率。0.4～1.0の範囲でランダムにサンプルをサンプリング
    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),  # バギングを行う頻度。1回ごとにバギングを行う場合は1を指定
     'verbosity': 0,  # LightGBMの詳細なログ出力を抑制
    }
    verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される

    train_data = lgb.Dataset(df_x_train, label=df_y_train)
    valid_data = lgb.Dataset(df_x_test, label=df_y_test, reference=train_data)
    model = lgb.train(param, 
                      train_data, 
                      valid_sets=[valid_data], 
                      callbacks=[
                                 #評価データのスコアが 100 ラウンド連続して改善されない場合に、トレーニングを停止することを意味します。
                                 lgb.early_stopping(stopping_rounds=100,verbose=False), 
                                 #  lgb.log_evaluation(verbose_eval)
                                 ]
                    )
    
    preds = model.predict(df_x_test, num_iteration=model.best_iteration)
    rmse = mean_squared_error(df_y_test, preds, squared=False)
    
    return rmse