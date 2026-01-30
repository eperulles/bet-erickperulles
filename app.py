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
    draw = np.sum(np.diag(m))
    a_win = np.sum(np.triu(m, 1))
    ou_results = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        under_prob = sum(m[i, j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j < threshold)
        ou_results[threshold] = {"Over": 1 - under_prob, "Under": under_prob}
    return h_win, draw, a_win, ou_results, m

@st.cache_data
def train_xgboost(df_input):
    df_train = df_input.copy()
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train = df_train.sort_values('Date').tail(500)
    
    def calc_form(team, date, lookback=5):
        past = df_train[(df_train['Date'] < date) & ((df_train['HomeTeam'] == team) | (df_train['AwayTeam'] == team))].tail(lookback)
        if len(past) == 0: return 0
        pts = 0
        for _, r in past.iterrows():
            if r['HomeTeam'] == team:
                if r['FTHG'] > r['FTAG']: pts += 3
                elif r['FTHG'] == r['FTAG']: pts += 1
            else:
                if r['FTAG'] > r['FTHG']: pts += 3
                elif r['FTAG'] == r['FTAG']: pts += 1
        return pts / (len(past) * 3)

    df_train['H_Form'] = df_train.apply(lambda x: calc_form(x['HomeTeam'], x['Date']), axis=1)
    df_train['A_Form'] = df_train.apply(lambda x: calc_form(x['AwayTeam'], x['Date']), axis=1)

    all_teams = sorted(list(set(df_input['HomeTeam'].unique()) | set(df_input['AwayTeam'].unique())))
    le = LabelEncoder().fit(all_teams)
    df_train['HomeIdx'] = le.transform(df_train['HomeTeam'])
    df_train['AwayIdx'] = le.transform(df_train['AwayTeam'])
    
    conds = [(df_train['FTHG'] > df_train['FTAG']), (df_train['FTHG'] == df_train['FTAG']), (df_train['FTHG'] < df_train['FTAG'])]
    df_train['Result'] = np.select(conds, [2, 1, 0])
    
    X = df_train[['HomeIdx', 'AwayIdx', 'H_Form', 'A_Form']]
    y = df_train['Result']
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=50, max_depth=3)
    model.fit(X, y)
    return model, le

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
        # 1. Poisson
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        ha = df[df['HomeTeam']==h_t]['FTHG'].mean() / avg_h
        hd = df[df['HomeTeam']==h_t]['FTAG'].mean() / avg_a
        aa = df[df['AwayTeam']==a_t]['FTAG'].mean() / avg_a
        ad = df[df['AwayTeam']==a_t]['FTHG'].mean() / avg_h
        p1, px, p2, pou, pm_mat = get_poisson_market(ha*ad*avg_h, aa*hd*avg_a)
        
        # 2. IA
        xgb, le = train_xgboost(df)
        def get_f(t):
            p = df[((df['HomeTeam']==t) | (df['AwayTeam']==t))].tail(5)
            pts = sum([(3 if (r['HomeTeam']==t and r['FTHG']>r['FTAG']) or (r['AwayTeam']==t and r['FTAG']>r['FTHG']) else 1 if r['FTHG']==r['FTAG'] else 0) for i,r in p.iterrows()])
            return pts/15
        ia_p = xgb.predict_proba(pd.DataFrame([[le.transform([h_t])[0], le.transform([a_t])[0], get_f(h_t), get_f(a_t)]], columns=['HomeIdx','AwayIdx','H_Form','A_Form']))[0]
        
        # 3. Bayes (Simple)
        try: bh, ba = run_bayesian_simple(df, h_t, a_t); b_res = get_poisson_market(bh, ba); bayes_ok = True
        except: bayes_ok = False; b_res = None
        
        st.session_state['results'] = {
            'p': (p1, px, p2, pou, pm_mat),
            'ia': ia_p,
            'b': b_res,
            'teams': (h_t, a_t)
        }

# --- DISPLAY TABS ---
if st.session_state['results']:
    res = st.session_state['results']
    p1, px, p2, pou, pm_mat = res['p']
    ia_p = res['ia']
    b_res = res['b']
    h_t, a_t = res['teams']

    main_tabs = st.tabs(["🎯 Pronóstico Principal", "📈 Mercado de Goles", "💰 Calculadoras y Momios", "🔬 Detalle de Modelos"])

    with main_tabs[0]:
        st.subheader(f"📊 Consenso Estratégico: {h_t} vs {a_t}")
        w_p, w_b, w_i = 0.2, 0.4, 0.4
        c1 = p1*w_p + (b_res[0] if b_res else p1)*w_b + ia_p[2]*w_i
        cx = px*w_p + (b_res[1] if b_res else px)*w_b + ia_p[1]*w_i
        c2 = p2*w_p + (b_res[2] if b_res else p2)*w_b + ia_p[0]*w_i
        
        cols = st.columns(3)
        cols[0].metric("Local (1)", f"{c1*100:.1f}%")
        cols[1].metric("Empate (X)", f"{cx*100:.1f}%")
        cols[2].metric("Visitante (2)", f"{c2*100:.1f}%")
        st.divider()
        st.info(f"🏆 Pronóstico Recomendado: **{'Gana Local' if c1>cx and c1>c2 else 'Empate' if cx>c1 and cx>c2 else 'Gana Visitante'}**")

    with main_tabs[1]:
        st.subheader("Over / Under Markets")
        g_cols = st.columns(5)
        for i, th in enumerate([0.5, 1.5, 2.5, 3.5, 4.5]):
            g_cols[i].metric(f"O/U {th}", f"Over: {pou[th]['Over']*100:.1f}%", f"Under: {pou[th]['Under']*100:.1f}%")
        
        fig = go.Figure(data=go.Heatmap(z=pm_mat[:5,:5], x=list(range(5)), y=list(range(5)), colorscale='Blues'))
        fig.update_layout(title="Matriz de Goles Exactos (Poisson)", width=400, height=400)
        st.plotly_chart(fig)

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
        st.subheader("🔬 Desglose de Modelos")
        d_c1, d_c2, d_c3 = st.columns(3)
        d_c1.write("**Poisson:**"); d_c1.write(f"1: {p1*100:.1f}% | X: {px*100:.1f}% | 2: {p2*100:.1f}%")
        d_c2.write("**IA XGBoost:**"); d_c2.write(f"1: {ia_p[2]*100:.1f}% | X: {ia_p[1]*100:.1f}% | 2: {ia_p[0]*100:.1f}%")
        if b_res: d_c3.write("**Bayesiano:**"); d_c3.write(f"1: {b_res[0]*100:.1f}% | X: {b_res[1]*100:.1f}% | 2: {b_res[2]*100:.1f}%")
