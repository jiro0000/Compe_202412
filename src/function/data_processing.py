import pandas as pd

def addtional_process(df):
    #以下特殊処理
    # 2025年からの経過年数に変換(先頭4文字を取得)
    df['year_built'] = df['year_built'].fillna('199000').astype(str)
    df['year_built'] = df['year_built'].str[:4].astype(int)
    df['year_built'] = 2025 - df['year_built']
    # object型のでnanを欠損値処理
    df = df.fillna('NA')
    # 0埋め
    df['addr1_1'] = df['addr1_1'].str.zfill(2)
    return df

def test():
    print("hello")

def test2():
    data = [1,2,3,4]
    df = pd.DataFrame(data)
    return df