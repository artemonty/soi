import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ─────────────────────────────────────────────────────────────
# НАБОР А — курение, оба пола, 114 стран (ВОЗ GHO 2022)
# ─────────────────────────────────────────────────────────────
countries_A = [
    "Afghanistan","Albania","Algeria","Angola","Argentina","Armenia",
    "Australia","Austria","Azerbaijan","Bangladesh","Belarus","Belgium",
    "Bolivia","Bosnia and Herzegovina","Brazil","Bulgaria","Cambodia",
    "Canada","Chile","China","Colombia","Croatia","Czech Republic",
    "Denmark","Ecuador","Egypt","Ethiopia","Finland","France","Georgia",
    "Germany","Ghana","Greece","Hungary","India","Indonesia","Iran",
    "Iraq","Ireland","Israel","Italy","Japan","Jordan","Kazakhstan",
    "Kenya","Kuwait","Kyrgyzstan","Latvia","Lebanon","Lithuania",
    "Luxembourg","Malaysia","Mexico","Moldova","Mongolia","Morocco",
    "Myanmar","Nepal","Netherlands","New Zealand","Nigeria",
    "North Macedonia","Norway","Pakistan","Panama","Peru","Philippines",
    "Poland","Portugal","Romania","Russia","Saudi Arabia","Serbia",
    "Slovakia","Slovenia","South Africa","South Korea","Spain",
    "Sri Lanka","Sudan","Sweden","Switzerland","Syria","Tajikistan",
    "Tanzania","Thailand","Tunisia","Turkey","Turkmenistan","UAE",
    "Uganda","Ukraine","United Kingdom","United States","Uruguay",
    "Uzbekistan","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe",
    "Chad","Senegal","Mali","Cameroon","Mozambique","Madagascar",
    "Ivory Coast","Niger","Guinea","Burkina Faso","Rwanda","Benin","Somalia"
]

smoke_A = np.array([
    29.2, 29.7, 20.1, 11.3, 22.1, 27.8, 13.0, 26.3, 23.7, 35.3,
    30.4, 23.4, 20.8, 37.6, 12.1, 31.5, 27.3, 13.5, 29.7, 25.6,
     8.2, 35.8, 28.8, 19.5,  7.3, 22.4,  4.4, 17.3, 24.3, 32.7,
    23.0,  7.2, 35.1, 28.6, 27.5, 32.1, 18.2, 22.0, 19.5, 18.0,
    21.5, 19.3, 29.4, 27.6, 10.3, 18.2, 27.0, 30.2, 42.6, 30.5,
    22.8, 22.8,  8.1, 30.3, 27.3, 18.5, 27.9, 28.2, 20.6, 15.2,
     3.9, 40.2, 13.3, 19.7,  5.3,  7.8, 23.8, 27.7, 25.1, 26.5,
    30.9, 19.0, 40.4, 28.2, 23.4, 20.3, 20.8, 25.6, 21.5, 14.0,
    13.0, 22.5, 31.2, 20.5, 16.6, 22.3, 29.6, 28.5, 22.6, 15.1,
    10.1, 30.7, 17.4, 14.7, 20.0, 22.2, 10.6, 25.9, 21.4, 14.0,
    15.4,  9.6,  6.8,  7.1, 11.0, 13.8, 12.5,  8.9,  6.2,  7.4,
     8.3, 11.9,  9.1, 20.1
])

n_A = len(smoke_A)
print(f"Набор А: n = {n_A}, mean = {smoke_A.mean():.4f}%, "
      f"median = {np.median(smoke_A):.4f}%, std = {smoke_A.std(ddof=1):.4f}%")

# ─────────────────────────────────────────────────────────────
# НАБОР Б — курение мужчин и женщин, 86 стран (ВОЗ GHO 2022)
# age-standardized, источник: apps.who.int/gho/data/node.main.TOBAGESTDCURR
# ─────────────────────────────────────────────────────────────
countries_B = [
    "Afghanistan","Albania","Algeria","Argentina","Armenia","Australia",
    "Austria","Azerbaijan","Bangladesh","Belarus","Belgium","Bolivia",
    "Bosnia and Herzegovina","Brazil","Bulgaria","Cambodia","Canada",
    "Chile","China","Colombia","Croatia","Czech Republic","Denmark",
    "Ecuador","Egypt","Ethiopia","Finland","France","Georgia","Germany",
    "Ghana","Greece","Hungary","India","Indonesia","Iran","Iraq","Ireland",
    "Israel","Italy","Japan","Jordan","Kazakhstan","Kenya","Kuwait",
    "Kyrgyzstan","Latvia","Lithuania","Luxembourg","Malaysia","Mexico",
    "Moldova","Mongolia","Morocco","Myanmar","Nepal","Netherlands",
    "New Zealand","Nigeria","North Macedonia","Norway","Pakistan","Peru",
    "Philippines","Poland","Portugal","Romania","Russia","Saudi Arabia",
    "Serbia","Slovakia","Slovenia","South Africa","South Korea","Spain",
    "Sri Lanka","Sweden","Switzerland","Thailand","Tunisia","Turkey",
    "UAE","Uganda","Ukraine","United Kingdom","United States","Uruguay",
    "Uzbekistan","Vietnam","Zimbabwe"
]

men = np.array([
    13.1, 31.4, 30.0, 23.9, 47.2,  9.0, 17.5, 26.5, 24.9, 34.8,
    21.9, 16.4, 33.9, 13.0, 37.1, 24.2, 10.7, 25.6, 44.0, 10.0,
    14.9, 40.6,  5.4, 13.1, 31.7, 48.1, 17.8,  3.5, 31.1,  7.3,
    59.3, 13.0, 33.1, 17.0, 24.9, 24.8, 24.5, 44.4, 31.1, 11.8,
    28.0, 32.5, 36.4, 33.7, 21.3, 32.1, 21.8, 47.0, 45.4, 17.8,
    26.8, 23.9, 16.5,  9.9,  4.6, 43.0,  6.9, 18.2, 16.4, 30.8,
    22.7, 30.5, 33.7, 34.1, 17.1, 34.8, 28.0, 19.9, 29.0, 29.0,
    26.6, 18.9,  5.3, 17.8, 30.5, 37.6, 39.3,  9.7,  8.7, 33.1,
     9.8, 11.2, 19.3, 15.2, 30.1, 15.9
])

women = np.array([
     0.6,  4.1,  0.3, 17.5,  1.3,  6.5, 17.7,  0.1,  0.1, 11.0,
    18.6,  3.0, 24.9,  6.8, 34.6,  1.4,  6.2, 20.5,  1.3, 10.3,
     1.9,  0.1,  0.5, 10.7, 28.5,  7.7, 15.7,  0.3, 26.5,  0.4,
     1.1,  0.2,  0.7, 11.3, 12.8, 18.9,  8.4,  8.9,  6.3,  0.5,
     1.1,  2.9, 17.6, 17.4, 20.0,  0.3,  6.7,  6.5,  6.0,  0.7,
     1.1,  4.0, 12.9,  8.0,  0.2, 36.0,  5.4,  1.5,  2.1,  3.3,
    16.7, 20.8, 17.3, 14.2,  1.9, 36.9, 21.8, 19.2,  5.9,  5.3,
    24.2,  0.1,  5.7, 19.7,  1.0,  1.2, 19.6,  1.3,  0.9,  9.5,
     7.0,  7.9, 14.2,  0.7,  0.4,  0.2
])

n_B = len(men)
print(f"\nНабор Б: n = {n_B} стран")
print(f"Мужчины: mean = {men.mean():.2f}%, median = {np.median(men):.2f}%, "
      f"std = {men.std(ddof=1):.2f}%")
print(f"Женщины: mean = {women.mean():.2f}%, median = {np.median(women):.2f}%, "
      f"std = {women.std(ddof=1):.2f}%")

# ─────────────────────────────────────────────────────────────
# НАБОР В — потребление алкоголя, 51 страна (ВОЗ GSRAH 2022)
# ─────────────────────────────────────────────────────────────
alcohol_dict = {
    "Australia": 10.6, "Austria": 12.3, "Belgium": 12.1, "Bulgaria": 12.6,
    "Canada": 8.9,  "China": 7.2,  "Croatia": 12.9, "Czech Republic": 14.3,
    "Denmark": 10.4, "Finland": 10.5, "France": 12.2, "Germany": 12.3,
    "Greece": 8.5,  "Hungary": 14.4, "India": 5.7,  "Indonesia": 0.9,
    "Ireland": 13.0, "Italy": 7.7,  "Japan": 8.0,  "Latvia": 12.3,
    "Lithuania": 16.2, "Netherlands": 9.9, "New Zealand": 9.6,
    "Norway": 7.0,  "Poland": 12.3, "Portugal": 12.3, "Romania": 14.8,
    "Russia": 11.7, "Serbia": 14.1, "Slovakia": 13.0, "Slovenia": 11.5,
    "South Korea": 10.2, "Spain": 10.0, "Sweden": 9.0, "Switzerland": 10.6,
    "Turkey": 1.5,  "Ukraine": 13.3, "United Kingdom": 11.4,
    "United States": 9.3, "Argentina": 9.3, "Brazil": 8.1, "Chile": 9.6,
    "Mexico": 5.2,  "South Africa": 9.3, "Kenya": 4.3, "Nigeria": 3.9,
    "Ghana": 2.6,  "Ecuador": 4.2, "Peru": 5.9,  "Colombia": 4.7,
    "Venezuela": 7.6,
}

# Совместить с набором А
joint = [(c, s, alcohol_dict[c])
         for c, s in zip(countries_A, smoke_A) if c in alcohol_dict]
smoke_C  = np.array([x[1] for x in joint])
alc_C    = np.array([x[2] for x in joint])
names_C  = [x[0] for x in joint]
n_C = len(joint)
print(f"\nНабор В (алкоголь): n = {n_C} стран")

# ═══════════════════════════════════════════════════════════════
# КРИТЕРИЙ 1 — одновыборочный t-критерий Стьюдента
# H0: mu = 22%,  H1: mu ≠ 22%
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("КРИТЕРИЙ 1 — t-критерий Стьюдента (одновыборочный)")
print("="*60)

mu0 = 22.0
t_stat, p_t = stats.ttest_1samp(smoke_A, mu0)
t_crit = stats.t.ppf(0.975, n_A - 1)
x_bar = smoke_A.mean()
s     = smoke_A.std(ddof=1)
se    = s / np.sqrt(n_A)

print(f"H0: mu = {mu0}%,  H1: mu ≠ {mu0}%")
print(f"x̄ = {x_bar:.4f}%,  s = {s:.4f}%,  n = {n_A},  SE = {se:.4f}%")
print(f"t_набл = {t_stat:.4f}")
print(f"t_крит (alpha=0.05, df={n_A-1}) = ±{t_crit:.4f}")
print(f"p-value = {p_t:.4f}")
if abs(t_stat) < t_crit:
    print("→ |t_набл| < t_крит и p > 0.05 → H0 НЕ отвергается")
else:
    print("→ |t_набл| > t_крит и p < 0.05 → H0 отвергается")

# ═══════════════════════════════════════════════════════════════
# КРИТЕРИЙ 2 — критерий Манна–Уитни
# H0: Me_men = Me_women,  H1: Me_men ≠ Me_women
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("КРИТЕРИЙ 2 — критерий Манна–Уитни")
print("="*60)

u_stat, p_mw = stats.mannwhitneyu(men, women, alternative='two-sided')

print(f"H0: медианы мужчин и женщин равны,  H1: медианы различны")
print(f"n1 (мужчины) = {n_B},  n2 (женщины) = {n_B}")
print(f"Медиана мужчин   = {np.median(men):.2f}%")
print(f"Медиана женщин   = {np.median(women):.2f}%")
print(f"U_набл = {u_stat:.1f}")
print(f"U_max  = n1*n2 = {n_B*n_B}")
print(f"p-value = {p_mw:.2e}")
if p_mw < 0.05:
    print("→ p < 0.05 → H0 отвергается")
else:
    print("→ p ≥ 0.05 → H0 не отвергается")

# ═══════════════════════════════════════════════════════════════
# КРИТЕРИЙ 3 — знаковый критерий
# H0: Me = 22%,  H1: Me ≠ 22%
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("КРИТЕРИЙ 3 — знаковый критерий (Sign test)")
print("="*60)

me0 = 22.0
n_plus  = int(np.sum(smoke_A > me0))
n_minus = int(np.sum(smoke_A < me0))
n_ties  = int(np.sum(smoke_A == me0))
n_eff   = n_plus + n_minus
res3 = stats.binomtest(n_plus, n_eff, 0.5, alternative='two-sided')

print(f"H0: Me = {me0}%,  H1: Me ≠ {me0}%")
print(f"Знаки «+» (xi > {me0}%): {n_plus}")
print(f"Знаки «−» (xi < {me0}%): {n_minus}")
print(f"Совпадения (xi = {me0}%): {n_ties}  (исключаются)")
print(f"n_eff = {n_eff}")
print(f"T+ = {n_plus} ~ Bin({n_eff}, 0.5)")
print(f"p-value = {res3.pvalue:.4f}")
if res3.pvalue < 0.05:
    print("→ p < 0.05 → H0 отвергается")
else:
    print("→ p ≥ 0.05 → H0 не отвергается")

# ═══════════════════════════════════════════════════════════════
# КРИТЕРИЙ 4 — критерий Краскела–Уоллиса
# H0: все три группы из одного распределения
# H1: хотя бы одна группа отличается
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("КРИТЕРИЙ 4 — критерий Краскела–Уоллиса")
print("="*60)

g_low  = smoke_A[smoke_A < 15]
g_mid  = smoke_A[(smoke_A >= 15) & (smoke_A <= 30)]
g_high = smoke_A[smoke_A > 30]

h_stat, p_kw = stats.kruskal(g_low, g_mid, g_high)
h_crit = stats.chi2.ppf(0.95, df=2)

print(f"H0: три группы из одного распределения,  H1: есть различия")
print(f"Группа «низкий» (< 15%):   n={len(g_low):2d},  mean={g_low.mean():.2f}%,  std={g_low.std(ddof=1):.2f}%")
print(f"Группа «средний» (15-30%): n={len(g_mid):2d},  mean={g_mid.mean():.2f}%,  std={g_mid.std(ddof=1):.2f}%")
print(f"Группа «высокий» (> 30%):  n={len(g_high):2d},  mean={g_high.mean():.2f}%,  std={g_high.std(ddof=1):.2f}%")
print(f"H_набл = {h_stat:.4f}")
print(f"H_крит = chi2(0.95, df=2) = {h_crit:.4f}")
print(f"p-value = {p_kw:.2e}")
if h_stat > h_crit:
    print("→ H_набл > H_крит и p < 0.05 → H0 отвергается")
else:
    print("→ H_набл ≤ H_крит → H0 не отвергается")

# ═══════════════════════════════════════════════════════════════
# ДОПОЛНИТЕЛЬНО — корреляция Спирмена (курение vs алкоголь)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ДОП. — корреляция Спирмена (курение vs алкоголь)")
print("="*60)

r_sp, p_sp = stats.spearmanr(smoke_C, alc_C)
print(f"n = {n_C},  r_S = {r_sp:.4f},  p-value = {p_sp:.4f}")
if p_sp < 0.05:
    print("→ p < 0.05 → корреляция статистически значима")

# ═══════════════════════════════════════════════════════════════
# ГРАФИКИ
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
})

# ── Рисунок 1: гистограмма (набор А) ─────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
n_bins = 10
ax1.hist(smoke_A, bins=n_bins, color='steelblue', edgecolor='white',
         alpha=0.85, label='Эмпирическое распределение', density=True)

# Кривая нормального распределения
x_range = np.linspace(smoke_A.min() - 2, smoke_A.max() + 2, 300)
ax1.plot(x_range, stats.norm.pdf(x_range, smoke_A.mean(), smoke_A.std(ddof=1)),
         'r-', lw=2, label='Нормальное распределение')

ax1.axvline(smoke_A.mean(),   color='red',    lw=1.5, ls='--', label=f'Среднее = {smoke_A.mean():.1f}%')
ax1.axvline(np.median(smoke_A), color='orange', lw=1.5, ls=':',  label=f'Медиана = {np.median(smoke_A):.1f}%')

ax1.set_xlabel('Распространённость курения, %')
ax1.set_ylabel('Плотность частоты')
ax1.set_title('Рисунок 1 — Распределение уровня курения среди взрослых\n(оба пола, n = 114 стран, ВОЗ GHO 2022)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
fig1.savefig('ris1_histogram.png', dpi=150, bbox_inches='tight')
print("\nСохранён: ris1_histogram.png")

# ── Рисунок 2: боксплоты мужчины / женщины ───────────────────
fig2, ax2 = plt.subplots(figsize=(7, 6))
bp = ax2.boxplot([men, women],
                 patch_artist=True,
                 medianprops=dict(color='black', lw=2),
                 whiskerprops=dict(lw=1.5),
                 capprops=dict(lw=1.5),
                 flierprops=dict(marker='o', ms=5, alpha=0.5))

bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][0].set_alpha(0.75)
bp['boxes'][1].set_facecolor('orange')
bp['boxes'][1].set_alpha(0.75)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Мужчины', 'Женщины'])
ax2.set_ylabel('Распространённость курения, %')
ax2.set_title('Рисунок 2 — Уровень курения мужчин и женщин\n(n = 86 стран, ВОЗ GHO 2022)')

patch_m = mpatches.Patch(color='steelblue', alpha=0.75,
                          label=f'Мужчины (Me = {np.median(men):.1f}%)')
patch_w = mpatches.Patch(color='orange',    alpha=0.75,
                          label=f'Женщины (Me = {np.median(women):.1f}%)')
ax2.legend(handles=[patch_m, patch_w], fontsize=9)
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()
fig2.savefig('ris2_boxplot_gender.png', dpi=150, bbox_inches='tight')
print("Сохранён: ris2_boxplot_gender.png")

# ── Рисунок 3: столбчатая по группам ─────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
groups      = ['Низкий\n(< 15%, n=32)', 'Средний\n(15–30%, n=65)', 'Высокий\n(> 30%, n=17)']
group_data  = [g_low, g_mid, g_high]
means_g     = [g.mean() for g in group_data]
stds_g      = [g.std(ddof=1) for g in group_data]
colors_g    = ['steelblue', 'seagreen', 'tomato']

bars = ax3.bar(groups, means_g, yerr=stds_g, capsize=6,
               color=colors_g, alpha=0.8, edgecolor='white',
               error_kw=dict(elinewidth=1.5, ecolor='black'))
ax3.set_ylabel('Среднее значение курения, %')
ax3.set_title('Рисунок 3 — Среднее значение курения по трём группам стран\n(набор А, Краскела–Уоллиса, H = 89,19, p < 0,001)')
for bar, m, s in zip(bars, means_g, stds_g):
    ax3.text(bar.get_x() + bar.get_width()/2, m + s + 0.5,
             f'{m:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
patch_l = mpatches.Patch(color='steelblue', alpha=0.8, label='Низкий уровень')
patch_m = mpatches.Patch(color='seagreen',  alpha=0.8, label='Средний уровень')
patch_h = mpatches.Patch(color='tomato',    alpha=0.8, label='Высокий уровень')
ax3.legend(handles=[patch_l, patch_m, patch_h], fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max(means_g) + max(stds_g) + 5)
fig3.tight_layout()
fig3.savefig('ris3_groups_bar.png', dpi=150, bbox_inches='tight')
print("Сохранён: ris3_groups_bar.png")

# ── Рисунок 4: рассеяние курение vs алкоголь ─────────────────
fig4, ax4 = plt.subplots(figsize=(9, 6))
ax4.scatter(alc_C, smoke_C, color='steelblue', alpha=0.75, s=60, edgecolors='white', lw=0.5)

# Линия тренда
z = np.polyfit(alc_C, smoke_C, 1)
p_line = np.poly1d(z)
x_line = np.linspace(alc_C.min(), alc_C.max(), 100)
ax4.plot(x_line, p_line(x_line), 'r--', lw=1.8, label=f'Линия тренда (r_S = {r_sp:.2f})')

# Подписи нескольких стран
highlight = {"Lithuania", "Serbia", "Indonesia", "Norway", "Nigeria", "France", "Russia"}
for name, s_val, a_val in zip(names_C, smoke_C, alc_C):
    if name in highlight:
        ax4.annotate(name, (a_val, s_val),
                     textcoords='offset points', xytext=(5, 3),
                     fontsize=8, color='dimgray')

ax4.set_xlabel('Потребление алкоголя, л/чел./год')
ax4.set_ylabel('Распространённость курения, %')
ax4.set_title('Рисунок 4 — Диаграмма рассеяния: курение vs алкоголь\n'
              f'(n = {n_C} стран, r_S = {r_sp:.2f}, p < 0,001)')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
fig4.tight_layout()
fig4.savefig('ris4_scatter_alcohol.png', dpi=150, bbox_inches='tight')
print("Сохранён: ris4_scatter_alcohol.png")

# ── Рисунок 5: ранговая корреляция (цвет = регион) ───────────
europe = {"Lithuania","Czech Republic","Romania","Hungary","Ireland","Ukraine",
          "Latvia","Poland","France","Belgium","Bulgaria","Croatia","Serbia",
          "Slovakia","Slovenia","Portugal","Russia","Switzerland","Netherlands",
          "Norway","Sweden","Denmark","Finland","Austria","Germany","Italy",
          "Spain","United Kingdom"}
americas = {"Argentina","Brazil","Chile","Mexico","United States","Colombia",
            "Ecuador","Peru","Uruguay","Venezuela","Canada"}

fig5, ax5 = plt.subplots(figsize=(9, 6))
for name, s_val, a_val in zip(names_C, smoke_C, alc_C):
    if name in europe:
        color = 'steelblue'
    elif name in americas:
        color = 'darkorange'
    else:
        color = 'seagreen'
    ax5.scatter(a_val, s_val, color=color, alpha=0.8, s=60,
                edgecolors='white', lw=0.5)

z5 = np.polyfit(alc_C, smoke_C, 1)
ax5.plot(x_line, np.poly1d(z5)(x_line), 'k--', lw=1.5,
         label=f'Линия тренда (r_S = {r_sp:.2f}, p < 0,001)')

p_eu = mpatches.Patch(color='steelblue',   alpha=0.8, label='Европа')
p_am = mpatches.Patch(color='darkorange',  alpha=0.8, label='Америка')
p_as = mpatches.Patch(color='seagreen',    alpha=0.8, label='Азия / Африка')
ax5.legend(handles=[p_eu, p_am, p_as,
           mpatches.Patch(color='none', label='')],
           fontsize=9)
ax5.set_xlabel('Потребление алкоголя, л/чел./год')
ax5.set_ylabel('Распространённость курения, %')
ax5.set_title('Рисунок 5 — Ранговая корреляция Спирмена: курение vs алкоголь\n'
              f'(n = {n_C} стран, r_S = {r_sp:.2f}, p < 0,001)')
ax5.grid(alpha=0.3)
fig5.tight_layout()
fig5.savefig('ris5_spearman.png', dpi=150, bbox_inches='tight')
print("Сохранён: ris5_spearman.png")

plt.show()