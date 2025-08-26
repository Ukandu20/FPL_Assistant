import pandas as pd, numpy as np

df = pd.read_csv("data/models/minutes/versions/v1/expected_minutes.csv", parse_dates=["date_played"])

# GK starters sanity
gk_s = df[(df.pos=="GK") & (df.is_starter==1)]
print("GK starters: mean true minutes:", gk_s.minutes_true.mean().round(1),
      "mean predicted:", gk_s.pred_minutes.mean().round(1),
      "MAE:", np.mean(np.abs(gk_s.minutes_true - gk_s.pred_minutes)).round(2))

# Bench cameo sanity
bch = df[df.is_starter==0]
print("Bench cameo rate (true):", (bch.minutes_true>0).mean().round(3),
      "pred p_cameo (avg):", bch.p_cameo.mean().round(3))
print("Bench minutes: true mean:", bch.minutes_true.mean().round(2),
      "pred bench head mean:", bch.pred_bench_head.mean().round(2))

# 60-min classification
th=60
y_true=(df.minutes_true>=th).astype(int); y_pred=(df.pred_minutes>=th).astype(int)
acc=(y_true==y_pred).mean()
print("≥60 accuracy:", round(acc,3))

# Are there any DNPs in TEST?
print("DNP rows:", ((df.is_starter==0) & (df.minutes_true==0)).sum())
print(df.groupby(['pos','is_starter'])['minutes_true'].agg(n='size',
          cameo_rate=lambda s: (s>0).mean()).reset_index())

# Confirm we filtered the same file the model just wrote
print(df[['season','gw_orig']].drop_duplicates().sort_values(['season','gw_orig']).tail())
print(df[['p_cameo','pred_bench_head']].describe())

# Sanity: per-pos p_cameo means on true bench rows
print(df[df.is_starter==0].groupby('pos')['p_cameo'].mean())
