import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pymc as pm
import arviz as az
import time
import plotly.express as pex

# --- APP CONFIG ---
st.set_page_config(page_title="Football AI Pro", layout="wide")

# Initialize Session State to prevent losing data on UI change
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None

st.markdown("""
<style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4451; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Herramienta de Predicción Multimodelo (Pro)")

# --- SIDEBAR: CSV UPLOAD ---
st.sidebar.header("📂 Datos")
uploaded_file = st.sidebar.file_uploader("Cargar Datos Históricos (CSV)", type=["csv"])

if uploaded_file:
    if st.session_state['df'] is None:
        st.session_state['df'] = pd.read_csv(uploaded_file)
    df = st.session_state['df']
    st.sidebar.success("✅ Datos cargados!")
else:
    st.info("👋 Sube un archivo CSV con las columnas HomeTeam, AwayTeam, FTHG, FTAG para comenzar.")
    st.stop()

# --- UTILS & MODELS ---
def get_poisson_market(home_exp, away_exp, max_goals=10):
    h_probs = [poisson.pmf(i, home_exp) for i in range(max_goals + 1)]
    a_probs = [poisson.pmf(i, away_exp) for i in range(max_goals + 1)]
    m = np.outer(h_probs, a_probs)
    h_win = np.sum(np.tril(m, -1))
    p_draw = np.sum(np.diag(m))
    a_win = np.sum(np.triu(m, 1))
    ou_results = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under_prob = sum(m[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j < threshold)
        ou_results[threshold] = {"Over": 1 - under_prob, "Under": under_prob}
    return h_win, p_draw, a_win, ou_results, m

@st.cache_data
def train_detailed_models(df_input):
    df_train = df_input.copy()
    df_train['Date'] = pd.to_datetime(df_train['Date'], dayfirst=True, errors='coerce')
    df_train = df_train.dropna(subset=['Date']).sort_values('Date').tail(1500)
    
    def get_stats(team, date, col=None, lookback=5):
        mask = (df_train['Date'] < date) & ((df_train['HomeTeam'] == team) | (df_train['AwayTeam'] == team))
        if col: mask &= (df_train[col] == team)
        p = df_train[mask].tail(lookback)
        if len(p) == 0: return 0, 0, 0
        
        pts = sum([(3 if (r['HomeTeam']==team and r['FTHG']>r['FTAG']) or (r['AwayTeam']==team and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0) for i,r in p.iterrows()])
        sc = sum([(r['FTHG'] if r['HomeTeam']==team else r['FTAG']) for i,r in p.iterrows()])
        co = sum([(r['FTAG'] if r['HomeTeam']==team else r['FTHG']) for i,r in p.iterrows()])
        return pts/(len(p)*3), sc/len(p), co/len(p)

    # Engineering features
    res = []
    for _, x in df_train.iterrows():
        h_f, h_s, h_c = get_stats(x['HomeTeam'], x['Date'])
        a_f, a_s, a_c = get_stats(x['AwayTeam'], x['Date'])
        hv_f, hv_s, hv_c = get_stats(x['HomeTeam'], x['Date'], col='HomeTeam')
        av_f, av_s, av_c = get_stats(x['AwayTeam'], x['Date'], col='AwayTeam')
        res.append([h_f, h_s, h_c, a_f, a_s, a_c, hv_f, hv_s, hv_c, av_f, av_s, av_c])
    
    f_cols = ['H_F','H_S','H_C','A_F','A_S','A_C','HV_F','HV_S','HV_C','AV_F','AV_S','AV_C']
    X_feat = pd.DataFrame(res, columns=f_cols)
    
    all_teams = sorted(list(set(df_input['HomeTeam'].unique()) | set(df_input['AwayTeam'].unique())))
    le = LabelEncoder().fit(all_teams)
    X_feat['H_Idx'] = le.transform(df_train['HomeTeam'])
    X_feat['A_Idx'] = le.transform(df_train['AwayTeam'])
    
    # Targets
    y_1x2 = np.select([(df_train['FTHG'] > df_train['FTAG']), (df_train['FTHG'] == df_train['FTAG'])], [2, 1], 0)
    m_1x2 = XGBClassifier(n_estimators=100, max_depth=4).fit(X_feat, y_1x2)
    
    # Goal Models FT (0.5 to 3.5)
    ft_models = {}
    for th in [0.5, 1.5, 2.5, 3.5]:
        y_o = (df_train['FTHG'] + df_train['FTAG'] > th).astype(int)
        ft_models[th] = XGBClassifier(n_estimators=80, max_depth=3).fit(X_feat, y_o)
        
    # Goal Models HT (0.5 to 1.5)
    ht_models = {}
    for th in [0.5, 1.5]:
        y_o = (df_train['HTHG'] + df_train['HTAG'] > th).astype(int)
        ht_models[th] = XGBClassifier(n_estimators=80, max_depth=3).fit(X_feat, y_o)
    
    # Keep processed data for pattern matching
    df_train[f_cols] = X_feat[f_cols].values
    return (m_1x2, ft_models, ht_models), le, df_train

def run_bayesian_simple(df, h_t, a_t):
    subset = df[(df['HomeTeam'].isin([h_t, a_t])) | (df['AwayTeam'].isin([h_t, a_t]))].tail(20)
    teams = sorted(list(set(subset['HomeTeam']) | set(subset['AwayTeam'])))
    t_idx = {t: i for i, t in enumerate(teams)}
    with pm.Model() as model:
        att = pm.Normal('att', 0, sigma=1, shape=len(teams))
        def_ = pm.Normal('def', 0, sigma=1, shape=len(teams))
        ha = pm.Normal('ha', 0.2, sigma=0.1)
        th_h = pm.math.exp(att[subset['HomeTeam'].map(t_idx).values] + def_[subset['AwayTeam'].map(t_idx).values] + ha)
        th_a = pm.math.exp(att[subset['AwayTeam'].map(t_idx).values] + def_[subset['HomeTeam'].map(t_idx).values])
        pm.Poisson('h_g', th_h, observed=subset['FTHG'].values)
        pm.Poisson('a_g', th_a, observed=subset['FTAG'].values)
        trace = pm.sample(300, tune=300, chains=1, progressbar=False)
    
    att_p = trace.posterior['att'].mean(dim=['chain', 'draw']).values
    def_p = trace.posterior['def'].mean(dim=['chain', 'draw']).values
    ha_p = trace.posterior['ha'].mean(dim=['chain', 'draw']).values
    
    h_e = np.exp(att_p[t_idx[h_t]] + def_p[t_idx[a_t]] + ha_p)
    a_e = np.exp(att_p[t_idx[a_t]] + def_p[t_idx[h_t]])
    return h_e, a_e

# --- SELECTION & RUN ---
teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
st.sidebar.divider()
h_t = st.sidebar.selectbox("🏠 Equipo Local", teams)
a_t = st.sidebar.selectbox("✈️ Equipo Visitante", teams, index=1)

if st.sidebar.button("🚀 Calcular Predicción"):
    with st.spinner('Analizando datos...'):
        # 1. Poisson FT
        # Global Averages with fallback (Full Time)
        avg_h = df['FTHG'].mean()
        avg_a = df['FTAG'].mean()
        if np.isnan(avg_h) or avg_h <= 0: avg_h = 1.35
        if np.isnan(avg_a) or avg_a <= 0: avg_a = 1.15
        
        # Global Averages with fallback (Half Time)
        ht_avg_h = df['HTHG'].mean()
        ht_avg_a = df['HTAG'].mean()
        if np.isnan(ht_avg_h) or ht_avg_h <= 0: ht_avg_h = 0.65
        if np.isnan(ht_avg_a) or ht_avg_a <= 0: ht_avg_a = 0.50
        
        # Robust metric calculation
        def get_team_stats(team, col, goal_col):
            rows = df[df[col]==team]
            if len(rows) == 0: return df[goal_col].mean()
            val = rows[goal_col].mean()
            return val if not np.isnan(val) else df[goal_col].mean()

        # FT Expectancies
        ha = get_team_stats(h_t, 'HomeTeam', 'FTHG') / avg_h
        hd = get_team_stats(h_t, 'HomeTeam', 'FTAG') / avg_a
        aa = get_team_stats(a_t, 'AwayTeam', 'FTAG') / avg_a
        ad = get_team_stats(a_t, 'AwayTeam', 'FTHG') / avg_h
        
        h_exp = np.clip(ha * ad * avg_h, 0.01, 10.0)
        a_exp = np.clip(aa * hd * avg_a, 0.01, 10.0)
        p_home, p_draw, p_away, pou, pm_mat = get_poisson_market(h_exp, a_exp)
        
        # HT Expectancies (Using REAL HTHG/HTAG data)
        ht_ha = get_team_stats(h_t, 'HomeTeam', 'HTHG') / ht_avg_h
        ht_hd = get_team_stats(h_t, 'HomeTeam', 'HTAG') / ht_avg_a
        ht_aa = get_team_stats(a_t, 'AwayTeam', 'HTAG') / ht_avg_a
        ht_ad = get_team_stats(a_t, 'AwayTeam', 'HTHG') / ht_avg_h
        
        ht_h_exp = np.clip(ht_ha * ht_ad * ht_avg_h, 0.01, 5.0)
        ht_a_exp = np.clip(ht_aa * ht_hd * ht_avg_a, 0.01, 5.0)
        
        ht_p_home, ht_p_draw, ht_p_away, ht_pou, ht_pm_mat = get_poisson_market(ht_h_exp, ht_a_exp)
        
        # Ensure matrices are valid
        pm_mat = np.nan_to_num(pm_mat)
        ht_pm_mat = np.nan_to_num(ht_pm_mat)
        
        # 2. IA Engine Update
        (m_1x2, ft_mods, ht_mods), le, df_feat = train_detailed_models(df)
        
        def get_live_stats(team, col=None):
            mask = ((df['HomeTeam']==team) | (df['AwayTeam']==team))
            if col: mask = (df[col]==team)
            p = df[mask].tail(5)
            if len(p) == 0: return 0.33, 1.0, 1.0
            pts = sum([(3 if (r['HomeTeam']==team and r['FTHG']>r['FTAG']) or (r['AwayTeam']==team and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0) for i,r in p.iterrows()])
            sc = sum([(r['FTHG'] if r['HomeTeam']==team else r['FTAG']) for i,r in p.iterrows()])
            co = sum([(r['FTAG'] if r['HomeTeam']==team else r['FTHG']) for i,r in p.iterrows()])
            return pts/(len(p)*3), sc/len(p), co/len(p)

        h_f, h_s, h_c = get_live_stats(h_t)
        a_f, a_s, a_c = get_live_stats(a_t)
        hv_f, hv_s, hv_c = get_live_stats(h_t, 'HomeTeam')
        av_f, av_s, av_c = get_live_stats(a_t, 'AwayTeam')
        
        cur_feat = pd.DataFrame([[h_f, h_s, h_c, a_f, a_s, a_c, hv_f, hv_s, hv_c, av_f, av_s, av_c, le.transform([h_t])[0], le.transform([a_t])[0]]], 
                               columns=['H_F','H_S','H_C','A_F','A_S','A_C','HV_F','HV_S','HV_C','AV_F','AV_S','AV_C','H_Idx','A_Idx'])
        
        ia_1x2 = m_1x2.predict_proba(cur_feat)[0]
        ia_ft_ou = {th: m.predict_proba(cur_feat)[0][1] for th, m in ft_mods.items()}
        ia_ht_ou = {th: m.predict_proba(cur_feat)[0][1] for th, m in ht_mods.items()}
        
        # --- PATTERN MATCHER (Goals integrated) ---
        tol = 0.15
        similar = df_feat[(df_feat['H_F'].between(h_f-tol, h_f+tol)) & (df_feat['A_F'].between(a_f-tol, a_f+tol))].tail(10)
        
        st.session_state['results'] = {
            'teams': (h_t, a_t),
            'ft_p': (p_home, p_draw, p_away, pou),
            'ht_p': (ht_p_home, ht_p_draw, ht_p_away, ht_pou),
            'ia_1x2': ia_1x2,
            'ia_ft_ou': ia_ft_ou,
            'ia_ht_ou': ia_ht_ou,
            'similar': similar[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].to_dict('records'),
            'stats': (h_f, h_s, h_c, a_f, a_s, a_c)
        }

# --- DISPLAY TABS ---
    with main_tabs[0]:
        st.subheader(f"📊 Consenso Estratégico: {h_t} vs {a_t}")
        p_h, p_d, p_a, p_ou = res['ft_p']
        ia_1x2 = res['ia_1x2']
        
        # Consenso 1x2 (Weighted)
        c1 = (p_h * 0.4) + (ia_1x2[2] * 0.6)
        cx = (p_d * 0.4) + (ia_1x2[1] * 0.6)
        c2 = (p_a * 0.4) + (ia_1x2[0] * 0.6)
        
        cols = st.columns(3)
        cols[0].metric("Local (1)", f"{c1*100:.1f}%")
        cols[1].metric("Empate (X)", f"{cx*100:.1f}%")
        cols[2].metric("Visitante (2)", f"{c2*100:.1f}%")
        
        st.divider()
        st.info(f"🏆 Pronóstico Recomendado: **{'Gana Local' if c1>cx and c1>c2 else 'Empate' if cx>c1 and cx>c2 else 'Gana Visitante'}**")

    with main_tabs[1]:
        st.subheader("📈 Mercado de Goles y Medio Tiempo")
        ht_h, ht_d, ht_a, ht_ou = res['ht_p']
        ia_ft_ou = res['ia_ft_ou']
        ia_ht_ou = res['ia_ht_ou']
        
        st.markdown("#### ⚽ Mercado Final (FT)")
        # Consensus Over Markets
        ft_ou_cols = st.columns(4)
        for i, th in enumerate([0.5, 1.5, 2.5, 3.5]):
            # Blend Poisson + IA
            p_val = p_ou[th]['Over']
            ia_val = ia_ft_ou[th]
            c_val = (p_val * 0.4) + (ia_val * 0.6)
            ft_ou_cols[i].metric(f"Consenso O {th}", f"{c_val*100:.1f}%")
        
        st.markdown("#### ⏱️ Medio Tiempo (HT)")
        ht_ou_cols = st.columns(2)
        for i, th in enumerate([0.5, 1.5]):
            p_val = ht_ou[th]['Over']
            ia_val = ia_ht_ou[th]
            c_val = (p_val * 0.4) + (ia_val * 0.6)
            ht_ou_cols[i].metric(f"Consenso HT O {th}", f"{c_val*100:.1f}%")
        
        st.divider()
        st.write("**Probabilidades al Descanso (1X2):**")
        ht_res_cols = st.columns(3)
        ht_res_cols[0].metric("HT Gana Local", f"{ht_h*100:.1f}%")
        ht_res_cols[1].metric("HT Empate", f"{ht_d*100:.1f}%")
        ht_res_cols[2].metric("HT Gana Visita", f"{ht_a*100:.1f}%")
        
        # Display Heatmap
        z_data = np.nan_to_num(pm_mat[:6, :6])
        
        if z_data.sum() > 0:
            # fig = pex.imshow(...)
            fig = pex.imshow(
                z_data,
                labels=dict(x="Goles Visitante", y="Goles Local", color="Probabilidad"),
                x=[str(i) for i in range(6)],
                y=[str(i) for i in range(6)],
                color_continuous_scale='Greens', 
                text_auto='.1%', 
                aspect="equal"
            )
            
            fig.update_xaxes(side="top")
            fig.update_layout(
                title="Probabilidad de Marcador Exacto",
                width=600, height=600,
                coloraxis_showscale=False # Remove scale to save space
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ Error crítico: La matriz de probabilidades se generó vacía. Esto puede deberse a que los datos del CSV tienen valores inválidos para estos equipos.")
        
        with st.expander("📝 Ver tabla técnica de probabilidades"):
            st.table(pd.DataFrame(z_data, columns=[f"V_{i}" for i in range(6)], index=[f"L_{i}" for i in range(6)]).style.format("{:.2%}"))

    with main_tabs[2]:
        st.subheader("💰 Análisis de Valor y Staking")
        c_bank, c_setup = st.columns([1, 2])
        bank = c_bank.number_input("Bankroll ($)", value=1000)
        kelly = c_setup.slider("Fracción Kelly", 0.05, 1.0, 0.25)
        
        st.write("#### Introduce Momios de tu Casa de Apuestas:")
        mc1, mc2, mc3, mco = st.columns(4)
        o1 = mc1.number_input("Momio Local", 1.01, 50.0, 2.0)
        ox = mc2.number_input("Momio Empate", 1.01, 50.0, 3.2)
        o2 = mc3.number_input("Momio Visitante", 1.01, 50.0, 3.8)
        oo25 = mco.number_input("Momio Over 2.5", 1.01, 50.0, 1.9)
        
        def display_val(prob, odd, label):
            ev = (prob*odd)-1
            if ev > 0:
                fk = ( (odd-1)*prob - (1-prob) ) / (odd-1)
                stake = max(0, fk * kelly * bank)
                st.success(f"✅ **{label}**: EV {ev*100:.1f}% | Sugerido: **${stake:.1f}**")
            else: st.write(f"⚪ {label}: Sin valor (EV {ev*100:.1f}%)")
            
        display_val(c1, o1, "Local")
        display_val(cx, ox, "Empate")
        display_val(c2, o2, "Visitante")
        display_val(pou[2.5]['Over'], oo25, "Over 2.5")

    with main_tabs[3]:
        st.subheader("🔬 Desglose de Modelos y Patrones")
        
        # Robust unpacking
        raw_inputs = res.get('ia_inputs', (0,0,0,0))
        h_f_g, a_f_g, h_f_v, a_f_v = raw_inputs if len(raw_inputs)==4 else (raw_inputs[0], raw_inputs[1], 0, 0)
        
        # --- NEW SECTION: HISTORICAL SIMILARITY ---
        st.markdown("#### 🕵️ Buscador de Patrones (Casos Similares)")
        st.write("He buscado partidos en tu CSV donde los equipos tenían una racha casi idéntica a la de hoy:")
        sim_data = res.get('similar', [])
        if sim_data:
            st.table(pd.DataFrame(sim_data))
            # Calculate outcome distribution
            results_list = [r['FTR'] for r in sim_data]
            total = len(results_list)
            h_pct = results_list.count('H')/total
            d_pct = results_list.count('D')/total
            a_pct = results_list.count('A')/total
            st.info(f"💡 En estos casos similares: **Local {h_pct*100:.0f}%** | **Empate {d_pct*100:.0f}%** | **Visita {a_pct*100:.0f}%**")
        else:
            st.write("No se encontraron casos históricos con un patrón de forma tan específico.")
        
        st.divider()
        
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.markdown(f"**{h_t} (Personalizado)**")
            st.write(f"Forma Global (L5): **{h_f_g*100:.1f}%**")
            st.write(f"Forma Local (L5 Casa): **{h_f_v*100:.1f}%**")
        with c_f2:
            st.markdown(f"**{a_t} (Personalizado)**")
            st.write(f"Forma Global (L5): **{a_f_g*100:.1f}%**")
            st.write(f"Forma Visita (L5 Fuera): **{a_f_v*100:.1f}%**")
        
        st.divider()
        d_c1, d_c2, d_c3 = st.columns(3)
        d_c1.write("**Poisson (Histórico):**")
        d_c1.write(f"1: {p_home*100:.1f}% | X: {p_draw*100:.1f}% | 2: {p_away*100:.1f}%")
        
        d_c2.write("**IA (Tendencia):**")
        d_c2.write(f"1: {ia_p[2]*100:.1f}% | X: {ia_p[1]*100:.1f}% | 2: {ia_p[0]*100:.1f}%")
        
        if b_res: 
            d_c3.write("**Bayesiano (Dinamico):**")
            d_c3.write(f"1: {b_res[0]*100:.1f}% | X: {b_res[1]*100:.1f}% | 2: {b_res[2]*100:.1f}%")
