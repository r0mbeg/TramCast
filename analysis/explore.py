"""Run from any directory: python analysis/explore.py (pandas, numpy, matplotlib)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]


def score(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    assert y.shape == prediction.shape and np.isfinite(prediction).all()
    return max(0, 1 - np.abs(y - prediction).sum() / y.sum()) if y.sum() else np.nan


assert score([10, 0], [8, 2]) == 0.6
assert np.isnan(score([0], [0]))
raw = pd.concat([
    pd.read_csv(ROOT / "dataset/labels" / name, sep=";", parse_dates=["date"])
    for name in ["labels_day_train.csv", "labels_day_test.csv"]
])
assert not raw.duplicated(["route", "date", "hour"]).any()
assert raw.notna().all().all() and (raw.boardings >= 0).all()
index = pd.MultiIndex.from_product(
    [ROUTES, pd.date_range("2025-01-01", "2025-10-31"), range(24)],
    names=["route", "date", "hour"],
)
data = raw.set_index(["route", "date", "hour"]).reindex(index)
data["observed"] = data.boardings.notna()
# ponytail: absent label keys assumed zero; reconcile raw events before interpreting outages.
data["boardings"] = data.boardings.fillna(0)
data = data.reset_index()
assert len(data) == 72960 and data.boardings.sum() == raw.boardings.sum()
data["dow"] = data.date.dt.dayofweek
data["month"] = data.date.dt.month
daily = data.groupby(["date", "route"]).boardings.sum().unstack("route")
daily.to_csv(OUT / "daily_boardings.csv", sep=";")

rows = []
for cutoff, end in [("2025-06-30", "2025-08-31"), ("2025-08-31", "2025-10-31")]:
    train = data[data.date <= cutoff]
    valid = data[(data.date > cutoff) & (data.date <= end)]
    assert train.date.max() < valid.date.min()
    for weeks in [4, 8, 12, 99]:
        history = train[train.date > pd.Timestamp(cutoff) - pd.Timedelta(weeks=weeks)]
        for statistic in ["mean", "median"]:
            profile = history.groupby(["route", "dow", "hour"]).boardings.agg(statistic)
            pred = valid.merge(profile.rename("prediction"), on=["route", "dow", "hour"],
                               how="left", validate="many_to_one")
            assert len(pred) == len(valid) and pred.prediction.notna().all()
            rows.append(dict(cutoff=cutoff, end=end, weeks=weeks, statistic=statistic,
                             wape_score=score(pred.boardings, pred.prediction)))
pd.DataFrame(rows).to_csv(OUT / "seasonal_probe.csv", sep=";", index=False)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
total = daily.sum(axis=1)
total.groupby(total.index.month).mean().div(1000).plot.bar(ax=axes[0, 0], color="#277e9c")
axes[0, 0].set(title="Среднесуточные посадки: летний спад и осенний рост",
               xlabel="Месяц 2025 года", ylabel="Тысяч посадок в сутки")
for weekend, label in [(False, "Будни"), (True, "Суббота и воскресенье")]:
    hourly = data[data.dow.ge(5).eq(weekend)].groupby(["date", "hour"]).boardings.sum()
    hourly.groupby("hour").mean().div(1000).plot(ax=axes[0, 1], label=label)
axes[0, 1].set(title="Почасовой профиль всех маршрутов", xlabel="Час", ylabel="Тысяч посадок")
axes[0, 1].legend()
daily[7].rolling(7).mean().div(1000).plot(ax=axes[1, 0], color="#277e9c")
axes[1, 0].set(title="Маршрут 7: провал летом и восстановление",
               xlabel="Дата", ylabel="Тысяч посадок в сутки, среднее за 7 дней")
for weekend, label in [(False, "Будни"), (True, "Суббота и воскресенье")]:
    z = daily.loc[(daily.index.dayofweek >= 5) == weekend, 50]
    z.groupby(z.index.month).mean().div(1000).plot(ax=axes[1, 1], marker="o", label=label)
axes[1, 1].set(title="Маршрут 50: осенью поток в выходные почти исчез",
               xlabel="Месяц 2025 года", ylabel="Тысяч посадок в сутки")
axes[1, 1].legend()
fig.suptitle("История января–октября 2025 • отсутствующие ключи labels приняты за 0", fontsize=13)
fig.savefig(OUT / "insights.png", dpi=150)
plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
