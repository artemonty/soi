"""
=============================================================================
Статистическое исследование данных
Тема: Распространённость курения среди взрослого населения по странам мира
Источник: WHO Global Health Observatory, 2022
Автор: Федотов Артемий Андреевич, группа 4417, ГУАП
=============================================================================

Данные: Age-standardized prevalence of current tobacco smoking,
        both sexes (%), 2022
Ссылка: https://www.who.int/data/gho/data/indicators/indicator-details/GHO/
        gho-tobacco-age-standardized-prevalence-of-current-tobacco-use
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

# ─── Настройка графиков ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})

BLUE  = '#1D4ED8'
LBLUE = '#93C5FD'
RED   = '#DC2626'
GRN   = '#16A34A'

os.makedirs('figs', exist_ok=True)

# =============================================================================
# 1. ДАННЫЕ
# =============================================================================
# WHO GHO: Age-standardised prevalence of current tobacco smoking (%), 2022
countries_data = [
    ("Afghanistan", 29.2), ("Albania", 29.7), ("Algeria", 20.1), ("Angola", 11.3),
    ("Argentina", 22.1), ("Armenia", 27.8), ("Australia", 13.0), ("Austria", 26.3),
    ("Azerbaijan", 23.7), ("Bangladesh", 35.3), ("Belarus", 30.4), ("Belgium", 23.4),
    ("Bolivia", 20.8), ("Bosnia and Herz.", 37.6), ("Brazil", 12.1), ("Bulgaria", 38.9),
    ("Cambodia", 27.7), ("Cameroon", 10.8), ("Canada", 13.0), ("Chile", 29.3),
    ("China", 25.6), ("Colombia", 10.2), ("Congo", 12.1), ("Costa Rica", 8.7),
    ("Croatia", 36.8), ("Cuba", 29.8), ("Czech Republic", 30.5), ("Denmark", 17.8),
    ("Ecuador", 9.6), ("Egypt", 22.4), ("El Salvador", 11.2), ("Estonia", 29.3),
    ("Ethiopia", 4.4), ("Finland", 18.8), ("France", 30.1), ("Georgia", 31.5),
    ("Germany", 27.3), ("Ghana", 5.6), ("Greece", 37.0), ("Guatemala", 12.0),
    ("Hungary", 31.1), ("India", 28.5), ("Indonesia", 39.9), ("Iran", 14.4),
    ("Iraq", 22.7), ("Ireland", 21.1), ("Israel", 20.4), ("Italy", 23.7),
    ("Japan", 20.1), ("Jordan", 30.3), ("Kazakhstan", 27.3), ("Kenya", 11.2),
    ("South Korea", 22.9), ("Kuwait", 17.1), ("Kyrgyzstan", 25.6), ("Latvia", 32.3),
    ("Lebanon", 42.6), ("Libya", 22.9), ("Lithuania", 29.2), ("Luxembourg", 22.0),
    ("Malaysia", 23.3), ("Mali", 14.6), ("Mexico", 9.1), ("Moldova", 30.9),
    ("Mongolia", 30.2), ("Morocco", 18.0), ("Mozambique", 12.4), ("Myanmar", 35.2),
    ("Nepal", 28.7), ("Netherlands", 21.7), ("New Zealand", 14.6), ("Nicaragua", 9.8),
    ("Nigeria", 3.9), ("North Macedonia", 40.2), ("Norway", 16.4), ("Pakistan", 22.8),
    ("Panama", 5.3), ("Paraguay", 19.2), ("Peru", 7.0), ("Philippines", 24.3),
    ("Poland", 29.9), ("Portugal", 25.4), ("Qatar", 12.2), ("Romania", 28.4),
    ("Russia", 30.8), ("Saudi Arabia", 15.2), ("Senegal", 10.0), ("Serbia", 40.4),
    ("Slovakia", 27.5), ("Slovenia", 22.4), ("South Africa", 20.3), ("Spain", 27.6),
    ("Sri Lanka", 22.4), ("Sudan", 11.5), ("Sweden", 13.8), ("Switzerland", 25.4),
    ("Syria", 35.3), ("Tajikistan", 17.8), ("Tanzania", 12.1), ("Thailand", 21.3),
    ("Tunisia", 30.6), ("Turkey", 30.7), ("Turkmenistan", 22.6), ("Uganda", 8.5),
    ("Ukraine", 30.0), ("UAE", 15.1), ("United Kingdom", 17.4),
    ("United States", 14.7), ("Uruguay", 20.0), ("Uzbekistan", 22.2),
    ("Venezuela", 10.6), ("Vietnam", 25.9), ("Yemen", 21.4), ("Zambia", 14.0),
    ("Zimbabwe", 15.4),
]

names = [d[0] for d in countries_data]
data  = np.array([d[1] for d in countries_data])
n     = len(data)

print(f"Объём выборки: n = {n} стран\n")

# =============================================================================
# 2. ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ
# =============================================================================
mean_   = np.mean(data)
median_ = np.median(data)
std_    = np.std(data, ddof=1)        # несмещённое СКО
var_    = np.var(data, ddof=1)        # несмещённая дисперсия
sem_    = std_ / np.sqrt(n)           # стандартная ошибка среднего
skew_   = stats.skew(data)            # коэффициент асимметрии
kurt_   = stats.kurtosis(data)        # избыточный эксцесс
min_    = np.min(data)
max_    = np.max(data)
q1      = np.percentile(data, 25)
q3      = np.percentile(data, 75)
iqr_    = q3 - q1
cv_     = std_ / mean_ * 100          # коэффициент вариации, %

print("=" * 55)
print("ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ")
print("=" * 55)
print(f"  Среднее (x̄):                  {mean_:.4f} %")
print(f"  Медиана (Me):                  {median_:.4f} %")
print(f"  Минимум:                       {min_:.1f} %")
print(f"  Максимум:                      {max_:.1f} %")
print(f"  Размах (R):                    {max_ - min_:.1f} %")
print(f"  Дисперсия (s²):                {var_:.4f} %²")
print(f"  СКО (s):                       {std_:.4f} %")
print(f"  Коэф. вариации (V):            {cv_:.2f} %")
print(f"  1-й квартиль (Q₁):             {q1:.4f} %")
print(f"  3-й квартиль (Q₃):             {q3:.4f} %")
print(f"  МКР (IQR):                     {iqr_:.4f} %")
print(f"  Асимметрия (Aₛ):               {skew_:.4f}")
print(f"  Эксцесс (E, избыточный):       {kurt_:.4f}")

# =============================================================================
# 3. ПРОВЕРКА ГИПОТЕЗЫ О НОРМАЛЬНОСТИ
# =============================================================================
print("\n" + "=" * 55)
print("ПРОВЕРКА НОРМАЛЬНОСТИ (α = 0.05)")
print("=" * 55)

# 3.1 Шапиро–Уилк
sw_stat, sw_p = stats.shapiro(data)
print(f"\n1. Шапиро–Уилк:")
print(f"   W = {sw_stat:.4f},  p = {sw_p:.4f}")
print(f"   Вывод: H₀ {'не отвергается' if sw_p > 0.05 else 'ОТВЕРГАЕТСЯ'}")

# 3.2 Андерсон–Дарлинг
ad_res = stats.anderson(data, dist='norm')
ad_stat = ad_res.statistic
ad_crit_5 = ad_res.critical_values[2]   # 5% уровень
print(f"\n2. Андерсон–Дарлинг:")
print(f"   A² = {ad_stat:.4f},  крит.(5%) = {ad_crit_5:.4f}")
print(f"   Вывод: H₀ {'не отвергается' if ad_stat < ad_crit_5 else 'ОТВЕРГАЕТСЯ'}")

# 3.3 Лиллиефорс (через KS с оценёнными параметрами)
ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, mean_, std_))
print(f"\n3. Лиллиефорс (KS с оценёнными параметрами):")
print(f"   D = {ks_stat:.4f},  p = {ks_p:.4f}")
print(f"   Вывод: H₀ {'не отвергается' if ks_p > 0.05 else 'ОТВЕРГАЕТСЯ'}")

# 3.4 Хи-квадрат Пирсона
k_bins = 8
observed, bin_edges = np.histogram(data, bins=k_bins)
expected = np.array([
    n * (stats.norm.cdf(bin_edges[i+1], mean_, std_) -
         stats.norm.cdf(bin_edges[i],   mean_, std_))
    for i in range(k_bins)
])

# Объединяем интервалы с ожидаемой частотой < 5
def merge_bins(obs, exp):
    obs, exp = list(obs), list(exp)
    changed = True
    while changed:
        changed = False
        for i in range(len(exp)):
            if exp[i] < 5:
                j = i + 1 if i == 0 else i - 1
                obs[j] += obs[i]; exp[j] += exp[i]
                obs.pop(i); exp.pop(i)
                changed = True
                break
    return np.array(obs), np.array(exp)

obs_m, exp_m = merge_bins(observed, expected)
k_eff   = len(obs_m)
chi2_st = np.sum((obs_m - exp_m)**2 / exp_m)
df_chi  = k_eff - 1 - 2        # -2: оцениваем μ и σ по выборке
chi2_cr = stats.chi2.ppf(0.95, df=df_chi)
p_chi   = 1 - stats.chi2.cdf(chi2_st, df=df_chi)

print(f"\n4. Хи-квадрат (Пирсон):")
print(f"   k_эфф = {k_eff},  df = {df_chi}")
print(f"   χ²набл = {chi2_st:.4f},  χ²крит = {chi2_cr:.4f},  p = {p_chi:.4f}")
print(f"   Вывод: H₀ {'не отвергается' if chi2_st < chi2_cr else 'ОТВЕРГАЕТСЯ'}")

# =============================================================================
# 4. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (95%)
# =============================================================================
alpha = 0.05
t_cr      = stats.t.ppf(1 - alpha/2, df=n - 1)
ci_mean_lo = mean_ - t_cr * sem_
ci_mean_hi = mean_ + t_cr * sem_

chi2_lo_ci = stats.chi2.ppf(alpha/2,   df=n - 1)
chi2_hi_ci = stats.chi2.ppf(1-alpha/2, df=n - 1)
ci_std_lo  = np.sqrt((n-1) * var_ / chi2_hi_ci)
ci_std_hi  = np.sqrt((n-1) * var_ / chi2_lo_ci)
ci_var_lo  = (n-1) * var_ / chi2_hi_ci
ci_var_hi  = (n-1) * var_ / chi2_lo_ci

print("\n" + "=" * 55)
print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (95%)")
print("=" * 55)
print(f"  t_крит (df={n-1}):  {t_cr:.4f}")
print(f"  ДИ для μ:  [{ci_mean_lo:.4f}%; {ci_mean_hi:.4f}%]")
print(f"  ДИ для σ:  [{ci_std_lo:.4f}%; {ci_std_hi:.4f}%]")
print(f"  ДИ для σ²: [{ci_var_lo:.4f}; {ci_var_hi:.4f}] %²")

# =============================================================================
# 5. АНАЛИЗ ВЫБРОСОВ
# =============================================================================
lf = q1 - 1.5 * iqr_
uf = q3 + 1.5 * iqr_
outlier_idx = np.where((data < lf) | (data > uf))[0]

print("\n" + "=" * 55)
print("АНАЛИЗ ВЫБРОСОВ (правило 1.5×IQR)")
print("=" * 55)
print(f"  Нижняя граница: {lf:.2f}%")
print(f"  Верхняя граница: {uf:.2f}%")
if len(outlier_idx) > 0:
    for i in outlier_idx:
        print(f"  ВЫБРОС: {names[i]} ({data[i]:.1f}%)")
else:
    print("  Статистических выбросов не обнаружено")

# =============================================================================
# 6. ГРУППИРОВКА СТРАН
# =============================================================================
low_mask  = data < 15
mid_mask  = (data >= 15) & (data <= 30)
high_mask = data > 30

print("\n" + "=" * 55)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ГРУПП")
print("=" * 55)
for label, mask in [('Низкий (< 15%)', low_mask),
                    ('Средний (15–30%)', mid_mask),
                    ('Высокий (> 30%)', high_mask)]:
    g = data[mask]
    print(f"  {label}: n={len(g)}, "
          f"mean={np.mean(g):.2f}%, std={np.std(g, ddof=1):.2f}%, "
          f"min={np.min(g):.1f}%, max={np.max(g):.1f}%")

# =============================================================================
# 7. ГРАФИКИ
# =============================================================================

# ── Рисунок 1: Гистограмма с нормальной кривой ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.hist(data, bins=10, density=True, color=LBLUE,
        edgecolor='white', lw=1.0, alpha=0.85, label='Гистограмма (плотность)')
xr = np.linspace(min_ - 3, max_ + 3, 400)
ax.plot(xr, stats.norm.pdf(xr, mean_, std_), color=RED, lw=2.5,
        label=f'N({mean_:.2f}; {std_:.2f}²)')
ax.axvline(mean_,   color=BLUE, lw=2.0, ls='--', label=f'Среднее = {mean_:.2f}%')
ax.axvline(median_, color=GRN,  lw=2.0, ls=':',  label=f'Медиана = {median_:.2f}%')
ax.set_xlabel('Распространённость курения, %', fontsize=13)
ax.set_ylabel('Плотность вероятности', fontsize=13)
ax.set_title('Рисунок 1 — Распределение распространённости курения по странам мира (2022 г.)',
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.35)
plt.tight_layout()
plt.savefig('figs/fig1_hist.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("\nФигура 1 сохранена: figs/fig1_hist.png")

# ── Рисунок 2: Столбчатая — топ/антитоп 15 ───────────────────────────────
sidx = np.argsort(data)[::-1]
t15n = [names[i] for i in sidx[:15]];  t15v = [data[i] for i in sidx[:15]]
b15n = [names[i] for i in sidx[-15:]]; b15v = [data[i] for i in sidx[-15:]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
bars1 = ax1.barh(range(15), t15v[::-1], color='#EF4444', alpha=0.85,
                 edgecolor='white', lw=0.8)
ax1.set_yticks(range(15)); ax1.set_yticklabels(t15n[::-1], fontsize=9)
ax1.set_xlabel('Уровень курения, %', fontsize=11)
ax1.set_title('Страны с наибольшим уровнем', fontsize=11)
ax1.axvline(mean_, color=BLUE, lw=1.8, ls='--', alpha=0.8, label=f'Среднее={mean_:.1f}%')
ax1.legend(fontsize=9); ax1.grid(axis='x', alpha=0.3)
for b, v in zip(bars1, t15v[::-1]):
    ax1.text(v + 0.3, b.get_y() + b.get_height()/2, f'{v}%', va='center', fontsize=8)

bars2 = ax2.barh(range(15), b15v, color=LBLUE, alpha=0.9,
                 edgecolor='white', lw=0.8)
ax2.set_yticks(range(15)); ax2.set_yticklabels(b15n, fontsize=9)
ax2.set_xlabel('Уровень курения, %', fontsize=11)
ax2.set_title('Страны с наименьшим уровнем', fontsize=11)
ax2.axvline(mean_, color=RED, lw=1.8, ls='--', alpha=0.8, label=f'Среднее={mean_:.1f}%')
ax2.legend(fontsize=9); ax2.grid(axis='x', alpha=0.3)
for b, v in zip(bars2, b15v):
    ax2.text(v + 0.1, b.get_y() + b.get_height()/2, f'{v}%', va='center', fontsize=8)

fig.suptitle('Рисунок 2 — Страны с наибольшим и наименьшим уровнем курения (2022 г.)',
             fontsize=12)
plt.tight_layout()
plt.savefig('figs/fig2_bar.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Фигура 2 сохранена: figs/fig2_bar.png")

# ── Рисунок 3: Круговая диаграмма ─────────────────────────────────────────
bins_e = [0, 10, 20, 30, 40, 100]
gc = [np.sum((data >= bins_e[i]) & (data < bins_e[i+1])) for i in range(4)]
gc.append(np.sum(data >= 40))
pie_c = ['#BFDBFE', '#60A5FA', '#2563EB', '#1E3A8A', '#DC2626']
pie_l = [f'< 10%\n({gc[0]} стран)', f'10–20%\n({gc[1]} стран)',
         f'20–30%\n({gc[2]} стран)', f'30–40%\n({gc[3]} стран)',
         f'> 40%\n({gc[4]} стран)']
fig, ax = plt.subplots(figsize=(9, 7))
ax.pie(gc, labels=pie_l, colors=pie_c, autopct='%1.1f%%',
       startangle=140, pctdistance=0.72,
       wedgeprops=dict(edgecolor='white', lw=2))
ax.set_title('Рисунок 3 — Доли стран по уровню курения (2022 г.)', fontsize=12)
plt.tight_layout()
plt.savefig('figs/fig3_pie.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Фигура 3 сохранена: figs/fig3_pie.png")

# ── Рисунок 4: Кумулятивная кривая ────────────────────────────────────────
sorted_d = np.sort(data)[::-1]
cum = np.arange(1, n + 1) / n * 100

fig, ax1 = plt.subplots(figsize=(16, 6))
ax1.bar(range(n), sorted_d, color=LBLUE, alpha=0.85, edgecolor='none', width=1.0)
ax1.set_ylabel('Уровень курения, %', fontsize=12, color=BLUE)
ax1.tick_params(axis='y', labelcolor=BLUE)
ax1.set_xticks([])
ax1.set_xlabel('Страны (в порядке убывания)', fontsize=11)
ax1.axhline(mean_, color=BLUE, lw=1.8, ls='--', alpha=0.7,
            label=f'Среднее = {mean_:.1f}%')
ax2_cum = ax1.twinx()
ax2_cum.plot(range(n), cum, color=RED, lw=2.5, label='Кумулятивная доля')
ax2_cum.set_ylabel('Накопленная доля стран, %', fontsize=12, color=RED)
ax2_cum.tick_params(axis='y', labelcolor=RED)
ax2_cum.set_ylim(0, 108)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2_cum.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=10, loc='upper right')
ax1.set_title('Рисунок 4 — Уровень курения по странам с кумулятивной кривой (2022 г.)',
              fontsize=12)
plt.tight_layout()
plt.savefig('figs/fig4_cum.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Фигура 4 сохранена: figs/fig4_cum.png")

# ── Рисунок 5: Boxplot + Violin ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8))

ax1.boxplot(data, vert=True, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor='#BFDBFE', color=BLUE),
            medianprops=dict(color=RED, lw=3),
            whiskerprops=dict(color=BLUE, lw=1.5),
            capprops=dict(color=BLUE, lw=2),
            flierprops=dict(marker='o', color=RED, ms=5, alpha=0.7))
ax1.set_ylabel('Распространённость курения, %', fontsize=11)
ax1.set_title('Ящик с усами (Boxplot)', fontsize=12)
ax1.set_xticks([])
for val, lbl, col in [(q1, f'Q₁ = {q1:.2f}%', BLUE),
                       (median_, f'Me = {median_:.2f}%', RED),
                       (q3, f'Q₃ = {q3:.2f}%', BLUE)]:
    ax1.annotate(lbl, xy=(1, val), xytext=(1.28, val), fontsize=9.5, color=col,
                 va='center', arrowprops=dict(arrowstyle='-', color=col, lw=0.8))
ax1.grid(axis='y', alpha=0.35)

vp = ax2.violinplot(data, positions=[1], showmeans=True, showmedians=True)
vp['bodies'][0].set_facecolor(LBLUE); vp['bodies'][0].set_alpha(0.75)
vp['cmeans'].set_color(BLUE);   vp['cmeans'].set_linewidth(2.5)
vp['cmedians'].set_color(RED);  vp['cmedians'].set_linewidth(2.5)
ax2.set_ylabel('Распространённость курения, %', fontsize=11)
ax2.set_title('Скрипичная диаграмма (Violin plot)', fontsize=12)
ax2.set_xticks([])
ax2.grid(axis='y', alpha=0.35)
ax2.legend(handles=[
    mlines.Line2D([], [], color=BLUE, lw=2, label=f'Среднее = {mean_:.2f}%'),
    mlines.Line2D([], [], color=RED,  lw=2, label=f'Медиана = {median_:.2f}%'),
], fontsize=10)

fig.suptitle('Рисунок 5 — Boxplot и Violin plot: распространённость курения (2022 г.)',
             fontsize=12)
plt.tight_layout()
plt.savefig('figs/fig5_box.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Фигура 5 сохранена: figs/fig5_box.png")

# ── Рисунок 6: Q-Q plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
(osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', plot=None)
ax.scatter(osm, osr, color=BLUE, alpha=0.65, s=28, zorder=3, label='Наблюдения')
ax.plot(osm, slope * osm + intercept, color=RED, lw=2.5,
        label=f'Теоретическая линия (r = {r:.4f})')
ax.set_xlabel('Теоретические квантили', fontsize=12)
ax.set_ylabel('Наблюдаемые квантили', fontsize=12)
ax.set_title('Рисунок 6 — Q-Q plot: проверка нормальности распределения', fontsize=12)
ax.legend(fontsize=11); ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig('figs/fig6_qq.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Фигура 6 сохранена: figs/fig6_qq.png  (r = {r:.4f})")

print("\n✓ Все расчёты и графики выполнены успешно.")
print(f"  Графики сохранены в папке: figs/")
