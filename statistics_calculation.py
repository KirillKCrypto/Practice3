from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent
XLSX = BASE.parent / 'Ответы ИИ.xlsx'
services = ['ChatGPT', 'Google Gemini', 'DeepSeek', 'YandexGPT / Алиса', 'Microsoft Copilot']
criteria = ['Удобство использования', 'Качество ответов', 'Полезность для учёбы', 'Полезность для программирования', 'Скорость получения ответа', 'Доверие к ответам сервиса']

def split_counter(series, exclude=()):
    cnt = Counter()
    for x in series.fillna(''):
        for item in str(x).split(','):
            item = item.strip()
            if item and item not in exclude:
                cnt[item] += 1
    return cnt

def bartlett_sphericity(X):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    det = max(float(np.linalg.det(R)), 1e-12)
    chi_sq = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    df_val = p * (p - 1) // 2
    return chi_sq, df_val, chi2.sf(chi_sq, df_val), R

def kmo_measure(X):
    X = np.asarray(X, dtype=float)
    R = np.corrcoef(X, rowvar=False)
    invR = np.linalg.pinv(R)
    D = np.diag(1 / np.sqrt(np.diag(invR)))
    partial = -D @ invR @ D
    np.fill_diagonal(partial, 0)
    R2 = R ** 2
    P2 = partial ** 2
    np.fill_diagonal(R2, 0)
    np.fill_diagonal(P2, 0)
    return float(R2.sum() / (R2.sum() + P2.sum()))

def varimax(Phi, gamma=1.0, q=30, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
        R = u @ vh
        d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R

def pca_factor(X, n_factors=None):
    Z = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    R = np.corrcoef(Z, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    if n_factors is None:
        n_factors = max(1, int((vals > 1).sum()))
    loadings = vecs[:, :n_factors] * np.sqrt(vals[:n_factors])
    if n_factors > 1:
        loadings = varimax(loadings)
    for j in range(loadings.shape[1]):
        if loadings[:, j].sum() < 0:
            loadings[:, j] *= -1
    return vals, vals / len(vals), loadings

def main():
    df = pd.read_excel(XLSX, sheet_name=0)
    service_score = pd.DataFrame(index=df.index)
    for s in services:
        cols = [f'{c} [{s}]' for c in criteria]
        service_score[s] = df[cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    freq_map = {'Никогда': 0, 'Реже одного раза в месяц': 1, 'Несколько раз в месяц': 2, 'Несколько раз в неделю': 3, 'Каждый день': 4}
    gpa_map = {'Ниже 3.5': 3.25, '3.5–4.0': 3.75, '4.1–4.5': 4.30, '4.6–5.0': 4.80}
    chars = pd.DataFrame({
        'цифровая компетентность': pd.to_numeric(df['Насколько хорошо вы разбираетесь в цифровых технологиях?'], errors='coerce'),
        'частота использования ИИ': df['Как часто вы используете ИИ-сервисы?'].map(freq_map),
        'число использованных сервисов': df['Какие ИИ-сервисы вы использовали?'].fillna('').apply(lambda x: len([i.strip() for i in str(x).split(',') if i.strip() and i.strip() != 'Не использовал(а)'])),
        'число учебных задач': df['Для каких учебных задач вы используете ИИ?'].fillna('').apply(lambda x: len([i.strip() for i in str(x).split(',') if i.strip() and i.strip() != 'Не использую для учёбы'])),
        'ускорение заданий': pd.to_numeric(df['ИИ помогает мне быстрее выполнять учебные задания'], errors='coerce'),
        'понимание сложных тем': pd.to_numeric(df['ИИ помогает мне лучше понимать сложные темы'], errors='coerce'),
        'доверие к ИИ': pd.to_numeric(df['Я доверяю ответам ИИ'], errors='coerce'),
        'желание чаще использовать': pd.to_numeric(df['Я хотел(а) бы чаще использовать ИИ в обучении'], errors='coerce'),
    })
    chars = chars.fillna(chars.mean(numeric_only=True))

    chi_obj, df_obj, p_obj, _ = bartlett_sphericity(service_score.values)
    kmo_obj = kmo_measure(service_score.values)
    eig_obj, exp_obj, load_obj = pca_factor(service_score.values)
    chi_ch, df_ch, p_ch, _ = bartlett_sphericity(chars.values)
    kmo_ch = kmo_measure(chars.values)
    eig_ch, exp_ch, load_ch = pca_factor(chars.values, n_factors=2)

    ranks = service_score.apply(lambda row: rankdata(row, method='average'), axis=1, result_type='expand')
    ranks.columns = services
    fried_stat, fried_p = friedmanchisquare(*[service_score[s].values for s in services])
    kendall_w = fried_stat / (len(df) * (len(services) - 1))
    mad = (ranks - ranks.median()).abs().mean(axis=1)
    good_mask = mad <= mad.quantile(0.75)
    fried_stat2, fried_p2 = friedmanchisquare(*[service_score.loc[good_mask, s].values for s in services])
    kendall_w2 = fried_stat2 / (good_mask.sum() * (len(services) - 1))

    X_reg = pd.DataFrame({
        'успеваемость': df['Ваша средняя успеваемость'].map(gpa_map),
        'цифровая компетентность': chars['цифровая компетентность'],
        'частота использования ИИ': chars['частота использования ИИ'],
        'число сервисов': chars['число использованных сервисов'],
        'число учебных задач': chars['число учебных задач'],
    })
    X_reg = X_reg.fillna(X_reg.mean(numeric_only=True))
    y = service_score.mean(axis=1)
    reg = LinearRegression().fit(X_reg, y)
    result = {
        'service_means': service_score.mean().round(3).to_dict(),
        'object_factor_analysis': {'bartlett_chi2': round(chi_obj, 3), 'df': df_obj, 'p': p_obj, 'kmo': round(kmo_obj, 3), 'eigenvalues': np.round(eig_obj, 3).tolist()},
        'expert_factor_analysis': {'bartlett_chi2': round(chi_ch, 3), 'df': df_ch, 'p': p_ch, 'kmo': round(kmo_ch, 3), 'eigenvalues': np.round(eig_ch, 3).tolist()},
        'agreement': {'friedman_chi2': round(fried_stat, 3), 'p': fried_p, 'kendall_w': round(kendall_w, 3)},
        'agreement_after_filter': {'n': int(good_mask.sum()), 'friedman_chi2': round(fried_stat2, 3), 'p': fried_p2, 'kendall_w': round(kendall_w2, 3)},
        'regression': {'intercept': round(float(reg.intercept_), 4), 'coefficients': pd.Series(reg.coef_, index=X_reg.columns).round(4).to_dict(), 'r2': round(r2_score(y, reg.predict(X_reg)), 4)}
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
