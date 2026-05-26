import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

# =========================
# ДАННЫЕ
# =========================

years = np.array([
    1990, 1992, 1994, 1996, 1998,
    2000, 2002, 2004, 2006, 2008,
    2010, 2012, 2014, 2016, 2018,
    2020, 2022
])

smoking = np.array([
    32.2, 31.5, 30.7, 29.9, 28.9,
    27.9, 27.1, 26.2, 25.5, 24.8,
    24.1, 23.4, 22.7, 22.2, 21.7,
    21.2, 20.6
])

# Порядковый номер наблюдения
t = np.arange(1, len(smoking) + 1)

# =========================
# ТАБЛИЦА
# =========================

df = pd.DataFrame({
    "Год": years,
    "Курение_%": smoking
})

print(df)

# =========================
# ГРАФИК ВРЕМЕННОГО РЯДА
# =========================

plt.figure(figsize=(10, 5))
plt.plot(years, smoking, marker='o')
plt.title("Глобальная распространённость курения")
plt.xlabel("Год")
plt.ylabel("Доля курящих, %")
plt.grid(True)
plt.show()

# =========================
# СКОЛЬЗЯЩИЕ СРЕДНИЕ
# =========================

ma3 = pd.Series(smoking).rolling(window=3, center=True).mean()
ma5 = pd.Series(smoking).rolling(window=5, center=True).mean()

# =========================
# АВТОКОРРЕЛЯЦИЯ
# =========================

print("\nАвтокорреляции:")

for lag in range(1, 8):
    r = np.corrcoef(smoking[:-lag], smoking[lag:])[0, 1]
    print(f"Лаг {lag}: r = {r:.4f}")

# Коррелограмма
plot_acf(smoking, lags=7)
plt.title("Коррелограмма")
plt.show()

# =========================
# АНОМАЛИИ (метод 3σ)
# =========================

mean_value = np.mean(smoking)
std_value = np.std(smoking, ddof=1)

lower_bound = mean_value - 3 * std_value
upper_bound = mean_value + 3 * std_value

print("\nПроверка на аномалии (3σ):")

for value in smoking:
    if value < lower_bound or value > upper_bound:
        print(f"Аномалия: {value}")

# =========================
# ЛИНЕЙНЫЙ ТРЕНД
# y = a0 + a1*t
# =========================

linear_coef = np.polyfit(t, smoking, 1)
linear_model = np.poly1d(linear_coef)

linear_pred = linear_model(t)

# Метрики
linear_r2 = r2_score(smoking, linear_pred)
linear_mae = mean_absolute_error(smoking, linear_pred)

print("\nЛинейный тренд")
print(f"y = {linear_coef[0]:.4f} * t + {linear_coef[1]:.4f}")
print(f"R² = {linear_r2:.4f}")
print(f"MAE = {linear_mae:.4f}")

# =========================
# КВАДРАТИЧНЫЙ ТРЕНД
# =========================

quad_coef = np.polyfit(t, smoking, 2)
quad_model = np.poly1d(quad_coef)

quad_pred = quad_model(t)

quad_r2 = r2_score(smoking, quad_pred)
quad_mae = mean_absolute_error(smoking, quad_pred)

print("\nКвадратичный тренд")
print(f"R² = {quad_r2:.4f}")
print(f"MAE = {quad_mae:.4f}")

# =========================
# ЭКСПОНЕНЦИАЛЬНЫЙ ТРЕНД
# y = a * exp(b*t)
# =========================

log_y = np.log(smoking)

exp_coef = np.polyfit(t, log_y, 1)

b = exp_coef[0]
a = np.exp(exp_coef[1])

exp_pred = a * np.exp(b * t)

exp_r2 = r2_score(smoking, exp_pred)
exp_mae = mean_absolute_error(smoking, exp_pred)

print("\nЭкспоненциальный тренд")
print(f"y = {a:.4f} * exp({b:.4f} * t)")
print(f"R² = {exp_r2:.4f}")
print(f"MAE = {exp_mae:.4f}")

# =========================
# ГРАФИК ВСЕХ МОДЕЛЕЙ
# =========================

plt.figure(figsize=(12, 6))

plt.plot(years, smoking, 'o-', label='Исходный ряд')
plt.plot(years, ma3, label='Скользящая средняя (m=3)')
plt.plot(years, ma5, label='Скользящая средняя (m=5)')

plt.plot(years, linear_pred, label='Линейный тренд')
plt.plot(years, quad_pred, label='Квадратичный тренд')
plt.plot(years, exp_pred, label='Экспоненциальный тренд')

plt.xlabel("Год")
plt.ylabel("Курение, %")
plt.title("Сглаживание и тренды")
plt.legend()
plt.grid(True)

plt.show()

# =========================
# ОСТАТКИ ЛИНЕЙНОЙ МОДЕЛИ
# =========================

residuals = smoking - linear_pred

plt.figure(figsize=(10, 5))
plt.plot(years, residuals, marker='o')
plt.axhline(0, linestyle='--')
plt.title("Остатки линейной модели")
plt.xlabel("Год")
plt.ylabel("Остаток")
plt.grid(True)
plt.show()

# =========================
# КРИТЕРИЙ ДАРБИНА-УОТСОНА
# =========================

dw = durbin_watson(residuals)

print(f"\nDurbin-Watson = {dw:.4f}")

# =========================
# ПРОГНОЗ НА 2026
# =========================

future_t = 19

forecast_2026 = quad_model(future_t)

print(f"\nПрогноз на 2026 год: {forecast_2026:.2f}%")
