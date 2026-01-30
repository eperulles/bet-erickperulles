import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pymc as pm
import arviz as az
import plotly.express as pex

# --- APP CONFIG ---
st.set_page_config(page_title="Football AI Pro", layout="wide")

# Initialize Session State
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

    res_feat = []
    for _, x in df_train.iterrows():
        h_f, h_s, h_c = get_stats(x['HomeTeam'], x['Date'])
        a_f, a_s, a_c = get_stats(x['AwayTeam'], x['Date'])
        hv_f, hv_s, hv_c = get_stats(x['HomeTeam'], x['Date'], col='HomeTeam')
        av_f, av_s, av_c = get_stats(x['AwayTeam'], x['Date'], col='AwayTeam')
        res_feat.append([h_f, h_s, h_c, a_f, a_s, a_c, hv_f, hv_s, hv_c, av_f, av_s, av_c])
    
    f_cols = ['H_F','H_S','H_C','A_F','A_S','A_C','HV_F','HV_S','HV_C','AV_F','AV_S','AV_C']
    X_feat = pd.DataFrame(res_feat, columns=f_cols)
    
    all_teams = sorted(list(set(df_input['HomeTeam'].unique()) | set(df_input['AwayTeam'].unique())))
    le = LabelEncoder().fit(all_teams)
    X_feat['H_Idx'] = le.transform(df_train['HomeTeam'])
    X_feat['A_Idx'] = le.transform(df_train['AwayTeam'])
    
    y_1x2 = np.select([(df_train['FTHG'] > df_train['FTAG']), (df_train['FTHG'] == df_train['FTAG'])], [2, 1], 0)
    m_1x2 = XGBClassifier(n_estimators=100, max_depth=4).fit(X_feat, y_1x2)
    
    ft_models = {th: XGBClassifier(n_estimators=80, max_depth=3).fit(X_feat, (df_train['FTHG']+df_train['FTAG'] > th).astype(int)) for th in [0.5, 1.5, 2.5, 3.5]}
    ht_models = {th: XGBClassifier(n_estimators=80, max_depth=3).fit(X_feat, (df_train['HTHG']+df_train['HTAG'] > th).astype(int)) for th in [0.5, 1.5]}
    
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

# --- SELECTION ---
teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
st.sidebar.divider()
h_t = st.sidebar.selectbox("🏠 Equipo Local", teams)
a_t = st.sidebar.selectbox("✈️ Equipo Visitante", teams, index=1)

if st.sidebar.button("🚀 Calcular Predicción"):
    with st.spinner('Analizando motores...'):
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        ht_avg_h, ht_avg_a = df['HTHG'].mean(), df['HTAG'].mean()
        
        def get_team_stats(team, col, g_col):
            rows = df[df[col]==team]
            return rows[g_col].mean() if len(rows)>0 else df[g_col].mean()

        ha, hd = get_team_stats(h_t, 'HomeTeam', 'FTHG')/avg_h, get_team_stats(h_t, 'HomeTeam', 'FTAG')/avg_a
        aa, ad = get_team_stats(a_t, 'AwayTeam', 'FTAG')/avg_a, get_team_stats(a_t, 'AwayTeam', 'FTHG')/avg_h
        h_exp, a_exp = np.clip(ha*ad*avg_h, 0.01, 10.0), np.clip(aa*hd*avg_a, 0.01, 10.0)
        p_h, p_d, p_a, p_ou, pm_mat = get_poisson_market(h_exp, a_exp)
        
        ht_ha, ht_hd = get_team_stats(h_t, 'HomeTeam', 'HTHG')/ht_avg_h, get_team_stats(h_t, 'HomeTeam', 'HTAG')/ht_avg_a
        ht_aa, ht_ad = get_team_stats(a_t, 'AwayTeam', 'HTAG')/ht_avg_a, get_team_stats(a_t, 'AwayTeam', 'HTHG')/ht_avg_h
        ht_h_exp, ht_a_exp = np.clip(ht_ha*ht_ad*ht_avg_h, 0.01, 5.0), np.clip(ht_aa*ht_hd*ht_avg_a, 0.01, 5.0)
        htp_h, htp_d, htp_a, htp_ou, htp_mat = get_poisson_market(ht_h_exp, ht_a_exp)
        
        (m_1x2, ft_mods, ht_mods), le, df_f = train_detailed_models(df)
        def get_l_stats(team, col=None):
            m = ((df['HomeTeam']==team)|(df['AwayTeam']==team))
            if col: m = (df[col]==team)
            p = df[m].tail(5)
            if len(p)==0: return 0.33, 1.0, 1.0
            pts = sum([(3 if (r['HomeTeam']==team and r['FTHG']>r['FTAG']) or (r['AwayTeam']==team and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0) for _,r in p.iterrows()])
            sc = sum([(r['FTHG'] if r['HomeTeam']==team else r['FTAG']) for _,r in p.iterrows()])
            co = sum([(r['FTAG'] if r['HomeTeam']==team else r['FTHG']) for _,r in p.iterrows()])
            return pts/(len(p)*3), sc/len(p), co/len(p)

        h_f, h_s, h_c = get_l_stats(h_t)
        a_f, a_s, a_c = get_l_stats(a_t)
        hv_f, hv_s, hv_c = get_l_stats(h_t, 'HomeTeam')
        av_f, av_s, av_c = get_l_stats(a_t, 'AwayTeam')
        
        feat = pd.DataFrame([[h_f,h_s,h_c, a_f,a_s,a_c, hv_f,hv_s,hv_c, av_f,av_s,av_c, le.transform([h_t])[0], le.transform([a_t])[0]]], 
                           columns=['H_F','H_S','H_C','A_F','A_S','A_C','HV_F','HV_S','HV_C','AV_F','AV_S','AV_C','H_Idx','A_Idx'])
        
        ia_1x2 = m_1x2.predict_proba(feat)[0]
        ia_ft_ou = {th: m.predict_proba(feat)[0][1] for th, m in ft_mods.items()}
        ia_ht_ou = {th: m.predict_proba(feat)[0][1] for th, m in ht_mods.items()}
        sim = df_f[(df_f['H_F'].between(h_f-0.15, h_f+0.15)) & (df_f['A_F'].between(a_f-0.15, a_f+0.15))].tail(5)
        
        try: bh, ba = run_bayesian_simple(df, h_t, a_t); b_res = get_poisson_market(bh, ba)
        except: b_res = None
        
        st.session_state['results'] = {
            'teams': (h_t, a_t),
            'ft_p': (p_h, p_d, p_a, p_ou, pm_mat),
            'ht_p': (htp_h, htp_d, htp_a, htp_ou),
            'ia_1x2': ia_1x2, 'ia_ft_ou': ia_ft_ou, 'ia_ht_ou': ia_ht_ou,
            'similar': sim[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].to_dict('records'),
            'stats': (h_f,h_s,h_c, a_f,a_s,a_c, hv_f,hv_s,hv_c, av_f,av_s,av_c),
            'b': b_res
        }

# --- DISPLAY ---
if st.session_state['results']:
    res = st.session_state['results']
    h_t, a_t = res['teams']
    p_h, p_d, p_a, p_ou, pm_mat = res['ft_p']
    ht_h, ht_d, ht_a, ht_ou = res['ht_p']
    ia_1x2, ia_ft_ou, ia_ht_ou = res['ia_1x2'], res['ia_ft_ou'], res['ia_ht_ou']
    b_res = res['b']
    
    tabs = st.tabs(["🎯 Pronóstico", "📈 Goles/HT", "💰 Valor", "🔬 Detalle"])
    
    with tabs[0]:
        st.subheader(f"📊 Consenso: {h_t} vs {a_t}")
        w_p, w_b, w_i = 0.2, 0.4, 0.4
        c1 = p_h*w_p + (b_res[0] if b_res else p_h)*w_b + ia_1x2[2]*w_i
        cx = p_d*w_p + (b_res[1] if b_res else p_d)*w_b + ia_1x2[1]*w_i
        c2 = p_a*w_p + (b_res[2] if b_res else p_a)*w_b + ia_1x2[0]*w_i
        cols = st.columns(3)
        cols[0].metric("Local", f"{c1*100:.1f}%")
        cols[1].metric("Empate", f"{cx*100:.1f}%")
        cols[2].metric("Visita", f"{c2*100:.1f}%")
        st.info(f"🏆 Recomendación: **{'Gana Local' if c1>cx and c1>c2 else 'Empate' if cx>c1 and cx>c2 else 'Gana Visita'}**")

    with tabs[1]:
        st.subheader("📈 Mercados de Goles")
        c_ft, c_ht = st.columns(2)
        with c_ft:
            st.write("**Full Time (Consenso IA+Poisson):**")
            for th in [0.5, 1.5, 2.5, 3.5]:
                cv = p_ou[th]['Over']*0.4 + ia_ft_ou[th]*0.6
                st.write(f"Over {th}: **{cv*100:.1f}%**")
        with c_ht:
            st.write("**Half Time (Consenso):**")
            for th in [0.5, 1.5]:
                cv = ht_ou[th]['Over']*0.4 + ia_ht_ou[th]*0.6
                st.write(f"HT Over {th}: **{cv*100:.1f}%**")
        
        st.divider()
        fig = pex.imshow(np.nan_to_num(pm_mat[:6,:6]), labels=dict(x="Goles V", y="Goles L"), color_continuous_scale='Greens', text_auto='.1%')
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("💰 Calculadora de Valor")
        bank = st.number_input("Bank ($)", 100, 100000, 1000)
        k = st.slider("Kelly", 0.1, 1.0, 0.2)
        m1, mx, m2 = st.columns(3)
        o1, ox, o2 = m1.number_input("ML", 1.1, 20.0, 2.0), mx.number_input("MX", 1.1, 20.0, 3.2), m2.number_input("M2", 1.1, 20.0, 3.5)
        
        def calc_v(p, o, lbl):
            ev = (p*o)-1
            if ev>0: st.success(f"✅ {lbl}: EV {ev*100:.1f}% | Stake: ${( (o*p-1)/(o-1) )*k*bank:.1f}")
            else: st.write(f"⚪ {lbl}: EV {ev*100:.1f}%")
        calc_v(c1, o1, "Local"), calc_v(cx, ox, "Empate"), calc_v(c2, o2, "Visita")

    with tabs[3]:
        st.subheader("🔬 Detalle de Modelos")
        st.markdown("#### 🕵️ Casos Similares en Historia")
        if res['similar']: st.table(pd.DataFrame(res['similar']))
        else: st.write("Sin patrones idénticos.")
        
        st.divider()
        d1, d2, d3 = st.columns(3)
        d1.write("**Poisson:**"); d1.write(f"1: {p_h*100:.1f}% | X: {p_d*100:.1f}%")
        d2.write("**IA:**"); d2.write(f"1: {ia_1x2[2]*100:.1f}% | X: {ia_1x2[1]*100:.1f}%")
        if b_res: d3.write("**Bayes:**"); d3.write(f"1: {b_res[0]*100:.1f}% | X: {b_res[1]*100:.1f}%")
