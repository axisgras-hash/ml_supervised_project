import time
import random 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, random, warnings, io, requests
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML Studio — by Ankit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:2.6rem; font-weight:800; letter-spacing:-1px;
}
.metric-card {
    background:linear-gradient(135deg,#eef2ff 0%,#f5f3ff 100%);
    border:1px solid #c7d2fe; border-radius:12px;
    padding:18px 20px; text-align:center; margin-bottom:10px;
    box-shadow:0 2px 8px rgba(79,70,229,0.08);
}
.metric-card h2{color:#4f46e5;margin:0;font-size:2rem;font-weight:800;}
.metric-card p {color:#6b7280;margin:0;font-size:.82rem;letter-spacing:1px;text-transform:uppercase;}
.section-header{
    border-left:4px solid #4f46e5;padding-left:12px;
    margin:1.2rem 0 .8rem;font-size:1.15rem;font-weight:700;color:#1e1b4b;
}
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#f8f7ff 0%,#eef2ff 100%);
    border-right:1px solid #e0e7ff;
}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════
def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame to Arrow-safe types before st.dataframe().
    Fixes: ArrowInvalid when a column has mixed int/str (e.g. after encoding).
    Strategy: for each column, if it looks numeric try numeric; else stringify.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            # Try numeric coerce; if >50% convert keep it, else stringify
            num = pd.to_numeric(out[col], errors='coerce')
            if num.notna().mean() > 0.5:
                out[col] = num
            else:
                out[col] = out[col].astype(str)
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out


def _style_ax(ax, title=""):
    ax.set_facecolor('#f8f9ff')
    if title: ax.set_title(title, color='#1e1b4b', fontsize=10, pad=6)
    ax.tick_params(colors='#555', labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)

def _style_fig(fig):
    fig.patch.set_facecolor('white')

def _show_df(df, **kwargs):
    """Safe wrapper — always sanitise before display."""
    st.dataframe(_safe_df(df), **kwargs)


# ══════════════════════════════════════════════
# DATASET LOADING
# ══════════════════════════════════════════════
def load_dataset(path):
    encodings = ["utf-8","utf-8-sig","cp1252","latin1","ISO-8859-1",
                 "ISO-8859-15","cp1250","cp1251","utf-16","utf-16-le","utf-16-be"]
    file_name = path.name if hasattr(path, "name") else str(path)
    ext = file_name.split('.')[-1].lower()
    for enc in encodings:
        try:
            if ext == 'csv':            return pd.read_csv(path, encoding=enc)
            elif ext in ['xlsx','xls']: return pd.read_excel(path)
            elif ext == 'json':         return pd.read_json(path)
            elif ext == 'xml':          return pd.read_xml(path)
        except: continue
    return pd.DataFrame()


def load_from_url(url: str) -> pd.DataFrame:
    url = url.strip()
    headers = {"User-Agent": "Mozilla/5.0"}

    if 'docs.google.com/spreadsheets' in url:
        try:
            base = url.split('/edit')[0] if '/edit' in url else (
                   url.split('/pub')[0]  if '/pub'  in url else url.rsplit('/',1)[0])
            gid = ('&gid=' + url.split('gid=')[-1].split('&')[0]) if 'gid=' in url else ''
            r = requests.get(base + '/export?format=csv' + gid, headers=headers, timeout=15)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            raise ValueError(f"Google Sheets load failed: {e}")

    if 'github.com' in url and '/blob/' in url:
        url = url.replace('github.com','raw.githubusercontent.com').replace('/blob/','/')

    if 'kaggle.com' in url:
        raise ValueError(
            "Kaggle requires authentication — cannot load via URL.\n"
            "👉 Download the CSV from Kaggle and use **Upload File** instead.")

    ext = url.split('?')[0].split('.')[-1].lower()
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        if ext == 'json' or 'application/json' in r.headers.get('Content-Type',''):
            return pd.read_json(io.StringIO(r.text))
        elif ext in ['xlsx','xls']:
            return pd.read_excel(io.BytesIO(r.content))
        elif ext == 'xml':
            return pd.read_xml(io.StringIO(r.text))
        else:
            try:   return pd.read_csv(io.StringIO(r.text))
            except:
                t = pd.read_html(io.StringIO(r.text))
                if t: return t[0]
                raise ValueError("Could not parse as CSV or HTML table.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error: {e}")


# ══════════════════════════════════════════════
# EDA HELPERS
# ══════════════════════════════════════════════
def generate_suggestions(df, n=6):
    cols = df.columns.tolist()
    if not cols: return []
    pool = []
    for col in random.sample(cols, min(n, len(cols))):
        pool += [f"What is the average of {col}?",
                 f"Show distribution of {col}",
                 f"Any missing values in {col}?",
                 f"Give a summary of {col}",
                 f"What is the max of {col}?",
                 f"How many unique values in {col}?"]
    random.shuffle(pool); return pool[:n]


def show_textual_analysis(df):
    palettes = ["deep","muted","pastel","bright","rocket","mako","coolwarm","Spectral"]
    text_df = df.select_dtypes('object')
    col_len = len(text_df.columns)
    if col_len == 0: st.info("No categorical columns found."); return
    fixed_col = 3
    row_fixed = math.ceil(col_len / fixed_col)
    fig, axes = plt.subplots(row_fixed, fixed_col, figsize=(18, row_fixed*5))
    _style_fig(fig)
    axes_flat = axes.flatten() if hasattr(axes,'flatten') else [axes]
    for idx, col in enumerate(text_df.columns):
        ax = axes_flat[idx]; _style_ax(ax, col)
        series = text_df[col].value_counts().head(10)
        x = [str(n)[:20] for n in series.index]; y = series.values
        bars = ax.bar(x, y, color=sns.color_palette(random.choice(palettes), len(x)),
                      edgecolor='none', width=0.6)
        ax.set_xticklabels(x, rotation=20, ha='right', fontsize=7)
        for bar, v in zip(bars, y):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(y)*0.01,
                    str(v), ha='center', va='bottom', fontsize=7, color='#333')
        ax.yaxis.set_visible(False)
    for j in range(col_len, len(axes_flat)): axes_flat[j].set_visible(False)
    plt.tight_layout(pad=2); st.pyplot(fig); plt.close()


def show_numerical_analysis(df):
    colors = ['#667eea','#f093fb','#4facfe','#43e97b','#fa709a','#f6d365','#a78bfa','#34d399']
    num_df = df.select_dtypes('number')
    imp_cols = [c for c in num_df.columns if num_df[c].nunique() <= 2000]
    num_df = num_df[imp_cols]; col_len = len(num_df.columns)
    if col_len == 0: st.info("No numeric columns found."); return
    fixed_col = 3; row_fixed = math.ceil(col_len / fixed_col)
    fig, axes = plt.subplots(row_fixed, fixed_col, figsize=(18, row_fixed*4))
    _style_fig(fig)
    axes_flat = axes.flatten() if hasattr(axes,'flatten') else [axes]
    for idx, col in enumerate(num_df.columns):
        ax = axes_flat[idx]; _style_ax(ax, col)
        data = num_df[col].dropna()
        ax.hist(data, bins=40, color=colors[idx%len(colors)], alpha=0.8, edgecolor='none')
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data); xr = np.linspace(data.min(), data.max(), 200)
            ax2 = ax.twinx(); ax2.plot(xr, kde(xr), color='#4f46e5', lw=1.5, alpha=0.8)
            ax2.set_yticks([]); ax2.spines['right'].set_visible(False)
        except: pass
        ax.yaxis.set_visible(False)
    for j in range(col_len, len(axes_flat)): axes_flat[j].set_visible(False)
    plt.tight_layout(pad=2); st.pyplot(fig); plt.close()


def show_corr(df):
    num_df = df.select_dtypes('number')
    if num_df.shape[1] < 2: st.info("Need at least 2 numeric columns."); return None
    corr = num_df.corr()
    size = min(16, num_df.shape[1]*1.2+2)
    fig, ax = plt.subplots(figsize=(size, size*0.85))
    _style_fig(fig); ax.set_facecolor('#f8f9ff')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax,
                cmap='coolwarm', linewidths=0.5, linecolor='#e0e7ff',
                annot_kws={'size':8}, cbar_kws={'shrink':0.8})
    ax.tick_params(colors='#555', labelsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close(); return corr


def show_pair_plot(df):
    num_df = df.select_dtypes('number').iloc[:,:5]
    if num_df.shape[1] < 2: st.info("Not enough numeric columns."); return
    g = sns.pairplot(num_df.sample(min(300,len(num_df))), corner=True, diag_kind='hist',
                     plot_kws={'alpha':0.5,'color':'#667eea'},
                     diag_kws={'color':'#f093fb'})
    g.fig.patch.set_facecolor('white'); st.pyplot(g.fig); plt.close()


# ══════════════════════════════════════════════
# PREPROCESSING HELPERS
# ══════════════════════════════════════════════
def fill_na_col(series, method='median'):
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.mean() if method=='mean' else series.median())
    if pd.api.types.is_object_dtype(series):
        return series.fillna(series.mode()[0] if not series.mode().empty else 'Unknown')
    return series

def fill_all_na(df, method='median'):
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any(): df[col] = fill_na_col(df[col], method)
    return df

def encode_df(df, method='le', skip_cols=None):
    from sklearn.preprocessing import LabelEncoder
    skip_cols = skip_cols or []; tdf = df.copy()
    text_cols = [c for c in tdf.select_dtypes('object').columns if c not in skip_cols]
    if method == 'le':
        le = LabelEncoder()
        for col in text_cols: tdf[col] = le.fit_transform(tdf[col].astype(str))
    elif method == 'ohe':
        dummies = pd.get_dummies(tdf[text_cols], dtype=int, drop_first=True)
        tdf = pd.concat([tdf.drop(text_cols, axis=1), dummies], axis=1)
    return tdf

def scale_df(df, method='min_max', skip_cols=None):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    skip_cols = skip_cols or []; tdf = df.copy()
    num_cols = [c for c in tdf.select_dtypes('number').columns if c not in skip_cols]
    if not num_cols: return tdf
    scaler = MinMaxScaler() if method == 'min_max' else StandardScaler()
    tdf[num_cols] = scaler.fit_transform(tdf[num_cols])
    return tdf


# ══════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════
def run_models(X, y, problem_type='classification', cv=3):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                                  GradientBoostingClassifier, GradientBoostingRegressor)
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB

    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
            'KNN':                 KNeighborsClassifier(n_neighbors=5),
            'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
            'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM':                 SVC(kernel='rbf', C=1.0, probability=True),
            'Naive Bayes':         GaussianNB(),
        }
        metric = 'accuracy'
    else:
        models = {
            'Linear Regression':  LinearRegression(),
            'Ridge Regression':   Ridge(alpha=1.0),
            'Lasso Regression':   Lasso(alpha=0.1, max_iter=2000),
            'KNN':                KNeighborsRegressor(n_neighbors=5),
            'Decision Tree':      DecisionTreeRegressor(max_depth=6, random_state=42),
            'Random Forest':      RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR':                SVR(kernel='rbf', C=1.0),
        }
        metric = 'r2'

    try:
        from xgboost import XGBClassifier, XGBRegressor
        if problem_type == 'classification':
            models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=4,
                                              eval_metric='logloss', random_state=42)
        else:
            models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    except ImportError: pass

    results, trained_models = {}, {}
    progress = st.progress(0, text="Training models…")
    for i, (name, model) in enumerate(models.items()):
        try:
            score = cross_val_score(model, X, y, cv=cv, scoring=metric).mean()
            results[name] = round(float(score), 6)   # keep raw float, NOT *100 yet
            model.fit(X, y)
            trained_models[name] = model
        except Exception:
            results[name] = None
        progress.progress((i+1)/len(models), text=f"Trained: {name}")
    progress.empty()
    return results, metric, trained_models


# ══════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════
for _k, _v in {
    'df': None, 'processed_df': None,
    'model_results': None, 'trained_models': None,
    'train_feature_cols': None, 'train_target_col': None,
    'train_problem_type': None, 'best_model_name': None,
    'chat_history': [], 'prep_target_col': None,
    'prep_dropped_cols': [],
    'prediction_result': None,          # store prediction so page doesn't re-run
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 ML Studio  Your AI Powered ML Agent")
    st.markdown("**by Ankit** · Powered by scikit-learn")
    st.divider()

    st.markdown("### 📁 Load Dataset")
    source = st.radio("Source", ["Upload File","Paste URL"], horizontal=True)

    new_df = None
    if source == "Upload File":
        file = st.file_uploader("CSV / Excel / JSON / XML",
                                type=["csv","xlsx","xls","json","xml"])
        if file:
            with st.spinner("Loading…"):
                new_df = load_dataset(file)
            if new_df is not None and new_df.empty:
                st.error("File loaded but appears empty.")
    else:
        url_input = st.text_input("Dataset URL",
            placeholder="https://raw.githubusercontent.com/.../data.csv")
        with st.expander("ℹ️ Supported URLs"):
            st.markdown("✅ Direct CSV/JSON/Excel · ✅ GitHub blob · ✅ Google Sheets\n"
                        "❌ Kaggle (needs login — download & upload instead)")
        if url_input:
            with st.spinner("Fetching…"):
                try:
                    new_df = load_from_url(url_input)
                    if new_df is not None and new_df.empty:
                        st.error("URL loaded but dataset is empty.")
                except ValueError as e: st.error(str(e))
                except Exception as e:  st.error(f"Unexpected error: {e}")

    if new_df is not None and not new_df.empty:
        st.session_state['df'] = new_df
        for _k in ['processed_df','model_results','trained_models',
                   'train_feature_cols','train_target_col','train_problem_type',
                   'best_model_name','prep_target_col','prep_dropped_cols',
                   'prediction_result']:
            st.session_state[_k] = None
        st.session_state['prep_dropped_cols'] = []
        st.session_state['chat_history'] = []
        st.success(f"✅ {new_df.shape[0]:,} rows × {new_df.shape[1]} cols")

    st.divider()
    if st.session_state['df'] is not None:
        d = st.session_state['df']
        st.markdown("### 📊 Quick Info")
        st.markdown(f"- **Rows:** {d.shape[0]:,}")
        st.markdown(f"- **Cols:** {d.shape[1]}")
        st.markdown(f"- **Numeric:** {len(d.select_dtypes('number').columns)}")
        st.markdown(f"- **Categorical:** {len(d.select_dtypes('object').columns)}")
        st.markdown(f"- **Missing:** {d.isna().sum().sum():,}")
    st.divider()
    st.markdown("### ⚙️ Plot Settings")
    palette_theme = st.selectbox("Palette", ["deep","muted","bright","pastel","rocket","mako"])
    sns.set_palette(palette_theme)


# ══════════════════════════════════════════════
# GUARD
# ══════════════════════════════════════════════
st.markdown('<h1 class="main-title">🤖 ML Studio</h1>', unsafe_allow_html=True)
st.markdown("**Explore · Analyze · Preprocess · Train · Predict** — all in one place")

if st.session_state['df'] is None:
    st.info("👈 Upload a dataset or paste a URL in the sidebar to get started.")
    st.stop()

df = st.session_state['df']

c1,c2,c3,c4 = st.columns(4)
for _col, (val, label) in zip([c1,c2,c3,c4],[
    (df.shape[0],"Rows"),(df.shape[1],"Columns"),
    (df.isna().sum().sum(),"Missing Cells"),
    (len(df.select_dtypes('number').columns),"Numeric Cols"),
]):
    _col.markdown(f'<div class="metric-card"><h2>{val:,}</h2><p>{label}</p></div>',
                  unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════
tabs = st.tabs(["🗂️ Data Explorer","📊 EDA","🔗 Correlations",
                "🛠️ Preprocessing","🤖 Train Models","💬 Ask Your Data"])


# ─────────────────────────────────────────────
# TAB 1 — DATA EXPLORER
# ─────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    view_mode = st.radio("View",["Head","Tail","Random Sample","Full"], horizontal=True)
    n_rows = st.slider("Rows to display", 5, 100, 10)
    if view_mode == "Head":            preview = df.head(n_rows)
    elif view_mode == "Tail":          preview = df.tail(n_rows)
    elif view_mode == "Random Sample": preview = df.sample(min(n_rows,len(df)))
    else:                              preview = df
    _show_df(preview, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
        _show_df(pd.DataFrame({'Column':df.dtypes.index,'Type':df.dtypes.astype(str).values}),
                 use_container_width=True, height=220)
    with col_b:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        miss = df.isna().sum().reset_index()
        miss.columns = ['Column','Missing']
        miss['%'] = (miss['Missing']/len(df)*100).round(2)
        miss = miss[miss['Missing']>0].sort_values('Missing',ascending=False)
        if miss.empty: st.success("🎉 No missing values!")
        else: _show_df(miss, use_container_width=True, height=220)

    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    t_num, t_cat = st.tabs(["Numerical","Categorical"])
    with t_num:
        nd = df.describe(include='number').round(2)
        if not nd.empty: _show_df(nd, use_container_width=True)
        else: st.info("No numeric columns.")
    with t_cat:
        cd = df.describe(include='object')
        if not cd.empty: _show_df(cd, use_container_width=True)
        else: st.info("No categorical columns.")

    st.markdown('<div class="section-header">Column Deep Dive</div>', unsafe_allow_html=True)
    chosen_col = st.selectbox("Pick a column", df.columns, key='dd_col')
    d1,d2,d3 = st.columns(3)
    d1.metric("Unique Values", df[chosen_col].nunique())
    d2.metric("Missing", int(df[chosen_col].isna().sum()))
    d3.metric("Dtype", str(df[chosen_col].dtype))
    fig, ax = plt.subplots(figsize=(9,3)); _style_fig(fig); _style_ax(ax)
    if pd.api.types.is_numeric_dtype(df[chosen_col]):
        ax.hist(df[chosen_col].dropna(), bins=40, color='#667eea', alpha=0.85, edgecolor='none')
        ax.set_title(f'Distribution — {chosen_col}', color='#1e1b4b')
    else:
        vc = df[chosen_col].value_counts().head(15)
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color='#f093fb', edgecolor='none')
        ax.set_title(f'Top Values — {chosen_col}', color='#1e1b4b')
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    eda_choice = st.radio("Analysis",["📊 Categorical","📈 Numerical Distributions","👫 Pairplot"],
                          horizontal=True)
    if eda_choice == "📊 Categorical":
        with st.spinner("Rendering…"): show_textual_analysis(df)
    elif eda_choice == "📈 Numerical Distributions":
        with st.spinner("Rendering…"): show_numerical_analysis(df)
    elif eda_choice == "👫 Pairplot":
        st.info("Up to 5 numeric columns, 300 random rows.")
        with st.spinner("Rendering…"): show_pair_plot(df)

    st.divider()
    st.markdown('<div class="section-header">Custom Plot Builder</div>', unsafe_allow_html=True)
    x_col = st.selectbox("X axis", df.columns, key='x_col')
    y_options = [None]+[c for c in df.columns if c!=x_col]
    y_col = st.selectbox("Y axis (optional)", y_options, key='y_col')
    plot_type = st.selectbox("Plot type",["Histogram","Bar","Box","Scatter","Line","Violin"])
    if st.button("🎨 Generate Plot"):
        fig,ax = plt.subplots(figsize=(10,4)); _style_fig(fig); _style_ax(ax)
        try:
            if plot_type=="Histogram": ax.hist(df[x_col].dropna(),bins=40,color='#667eea',alpha=0.85)
            elif plot_type=="Bar" and y_col: df.groupby(x_col)[y_col].mean().plot(kind='bar',ax=ax,color='#f093fb')
            elif plot_type=="Box":
                if y_col: sns.boxplot(data=df,x=x_col,y=y_col,ax=ax)
                else:     sns.boxplot(data=df,y=x_col,ax=ax,color='#4facfe')
            elif plot_type=="Scatter" and y_col:
                ax.scatter(df[x_col],df[y_col],color='#43e97b',alpha=0.4,s=20)
                ax.set_xlabel(x_col,color='#555'); ax.set_ylabel(y_col,color='#555')
            elif plot_type=="Line" and y_col: ax.plot(df[x_col],df[y_col],color='#667eea',lw=1.5)
            elif plot_type=="Violin":
                if y_col: sns.violinplot(data=df,x=x_col,y=y_col,ax=ax)
                else:     sns.violinplot(data=df,y=x_col,ax=ax,color='#fa709a')
            ax.set_title(f'{plot_type}: {x_col}', color='#1e1b4b')
            st.pyplot(fig)
        except Exception as e: st.error(f"Could not render: {e}")
        plt.close()


# ─────────────────────────────────────────────
# TAB 3 — CORRELATIONS
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols_list = df.select_dtypes('number').columns.tolist()
    if len(num_cols_list) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        sel_corr = st.multiselect("Columns", num_cols_list,
                                  default=num_cols_list[:min(12,len(num_cols_list))])
        if len(sel_corr) >= 2:
            corr_result = show_corr(df[sel_corr])
            if corr_result is not None:
                st.markdown('<div class="section-header">Top Correlated Pairs</div>',
                            unsafe_allow_html=True)
                pairs = (corr_result.abs()
                         .where(np.tril(np.ones(corr_result.shape),k=-1).astype(bool))
                         .stack().sort_values(ascending=False).head(10).reset_index())
                pairs.columns = ['Feature A','Feature B','|r|']
                pairs['|r|'] = pairs['|r|'].round(3)
                _show_df(pairs, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4 — PREPROCESSING
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Data Preprocessing Pipeline</div>',
                unsafe_allow_html=True)
    st.info("Select your **target column first** — it will be excluded from scaling.")

    work_df = df.copy()

    st.markdown("**🎯 Step 0 — Select Target Column**")
    prep_target = st.selectbox("Target column (excluded from scaling)",
                               ["— None —"]+work_df.columns.tolist(), key='prep_target_sel')
    prep_target = None if prep_target == "— None —" else prep_target
    st.session_state['prep_target_col'] = prep_target

    if prep_target and pd.api.types.is_object_dtype(work_df[prep_target]):
        st.warning(f"⚠️ **{prep_target}** is categorical → will be Label-Encoded automatically.")

    st.markdown("**Step 1 — Drop Columns**")
    cols_to_drop = st.multiselect("Columns to remove",
                                  [c for c in work_df.columns if c != prep_target],
                                  key='cols_to_drop_selector')

    st.markdown("**Step 2 — Handle Missing Values**")
    miss_thresh = st.slider("Drop columns with missing % above", 0, 100, 50)
    fill_method = st.radio("Fill strategy", ["median","mean","drop rows"], horizontal=True)

    st.markdown("**Step 3 — Encode Categorical Features**")
    enc_method = st.radio("Encoding", ["Label Encoding","One-Hot Encoding","None"], horizontal=True)

    st.markdown("**Step 4 — Feature Scaling** *(target column excluded)*")
    scale_method = st.radio("Scaling", ["Min-Max","Standard Scaler","None"], horizontal=True)

    if st.button("⚙️ Apply Pipeline"):
        with st.spinner("Processing…"):
            result = work_df.copy()

            # Track actually dropped columns so Train tab can exclude them
            actually_dropped = list(cols_to_drop) if cols_to_drop else []

            if cols_to_drop:
                result.drop([c for c in cols_to_drop if c in result.columns], axis=1, inplace=True)

            high_miss = [c for c in result.columns
                         if c != prep_target and result[c].isna().mean()*100 > miss_thresh]
            if high_miss:
                result.drop(high_miss, axis=1, inplace=True)
                actually_dropped += high_miss
                st.warning(f"Dropped {len(high_miss)} high-missing col(s): {high_miss}")

            if fill_method in ['median','mean']:
                result = fill_all_na(result, method=fill_method)
            elif fill_method == 'drop rows':
                result.dropna(inplace=True)

            # Always encode target if it's categorical
            if prep_target and prep_target in result.columns and \
               pd.api.types.is_object_dtype(result[prep_target]):
                from sklearn.preprocessing import LabelEncoder
                result[prep_target] = LabelEncoder().fit_transform(
                    result[prep_target].astype(str))

            if enc_method == "Label Encoding":
                result = encode_df(result, method='le',
                                   skip_cols=[prep_target] if prep_target else [])
            elif enc_method == "One-Hot Encoding":
                result = encode_df(result, method='ohe',
                                   skip_cols=[prep_target] if prep_target else [])

            if scale_method != "None":
                skip = [prep_target] if prep_target else []
                try:
                    result = scale_df(result,
                                      method='min_max' if scale_method=='Min-Max' else 'ss',
                                      skip_cols=skip)
                except Exception as e:
                    st.warning(f"Scaling skipped: {e}")

            st.session_state['processed_df'] = result
            st.session_state['prep_dropped_cols'] = actually_dropped
            # Reset model results since data changed
            for _k in ['model_results','trained_models','train_feature_cols',
                       'train_target_col','train_problem_type','best_model_name',
                       'prediction_result']:
                st.session_state[_k] = None

            st.success(f"✅ Pipeline done! {result.shape[0]:,} rows × {result.shape[1]} cols")
            _show_df(result.head(10), use_container_width=True)
            csv = result.to_csv(index=False).encode()
            st.download_button("⬇️ Download Processed CSV", csv,
                               "processed_data.csv","text/csv")


# ─────────────────────────────────────────────
# TAB 5 — TRAIN MODELS
# ─────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">AutoML — Train & Compare Models</div>',
                unsafe_allow_html=True)

    _proc = st.session_state.get('processed_df')
    source_df = _proc if _proc is not None else df.copy()
    if _proc is not None:
        st.success("✅ Using preprocessed dataset")
    else:
        st.warning("⚠️ Using raw dataset — run Preprocessing first for best results.")

    prep_tgt_hint = st.session_state.get('prep_target_col')
    num_cols_src  = source_df.select_dtypes('number').columns.tolist()

    if len(num_cols_src) < 2:
        st.warning("Need at least 2 numeric columns. Run Preprocessing (encode categoricals) first.")
    else:
        default_tgt_idx = (num_cols_src.index(prep_tgt_hint)
                           if prep_tgt_hint and prep_tgt_hint in num_cols_src
                           else len(num_cols_src)-1)
        target_col   = st.selectbox("🎯 Target Column", num_cols_src, index=default_tgt_idx,
                                    key='train_tgt_sel')
        problem_type = st.radio("Problem Type", ["classification","regression"], horizontal=True,
                                key='train_prob_sel')

        n_u = source_df[target_col].nunique()
        if problem_type=='classification' and n_u>20:
            st.warning(f"⚠️ Target has {n_u} unique values — sure this is classification?")
        if problem_type=='regression' and n_u<10:
            st.warning(f"⚠️ Target has only {n_u} unique values — consider classification.")

        # FIX: feature options must exclude target AND previously dropped columns
        dropped_in_prep = st.session_state.get('prep_dropped_cols') or []
        feat_options = [c for c in num_cols_src
                        if c != target_col and c not in dropped_in_prep]
        feature_cols = st.multiselect("Feature Columns", feat_options,
                                      default=feat_options, key='train_feat_sel')

        cv_folds = st.slider("Cross-validation folds", 2, 10, 3, key='cv_slider')

        st.markdown("**Models:**")
        if problem_type == 'classification':
            st.markdown("`Logistic Regression` · `KNN` · `Decision Tree` · `Random Forest` · "
                        "`Gradient Boosting` · `SVM` · `Naive Bayes` · `XGBoost`*(if installed)*")
        else:
            st.markdown("`Linear Regression` · `Ridge` · `Lasso` · `KNN` · `Decision Tree` · "
                        "`Random Forest` · `Gradient Boosting` · `SVR` · `XGBoost`*(if installed)*")

        if st.button("🚀 Train All Models", key='train_btn'):
            if not feature_cols:
                st.error("Select at least one feature column.")
            else:
                clean_src = source_df[feature_cols+[target_col]].dropna()
                X = clean_src[feature_cols].values
                y = clean_src[target_col].values
                if problem_type == 'classification':
                    y = y.astype(int)

                with st.spinner("Training…"):
                    results, metric_name, trained_models = run_models(
                        X, y, problem_type, cv_folds)

                st.session_state['model_results']      = results
                st.session_state['trained_models']     = trained_models
                st.session_state['train_feature_cols'] = feature_cols
                st.session_state['train_target_col']   = target_col
                st.session_state['train_problem_type'] = problem_type
                st.session_state['source_df_for_pred'] = source_df
                st.session_state['train_metric_name']  = metric_name
                st.session_state['prediction_result']  = None   # reset old prediction

                valid = {k:v for k,v in results.items() if v is not None}
                if valid:
                    best_name = max(valid, key=valid.get)
                    st.session_state['best_model_name'] = best_name

        # ── Show results if available ──────────────────────────────────────
        if st.session_state.get('model_results'):
            results     = st.session_state['model_results']
            metric_name = st.session_state.get('train_metric_name', 'score')
            valid       = {k:v for k,v in results.items() if v is not None}
            best_name   = st.session_state.get('best_model_name','')

            # ── Results table ──────────────────────────────────────────────
            # For R²: show actual value (can be negative); for accuracy: show %
            def fmt_score(name, v):
                if metric_name == 'accuracy':
                    return f"{v*100:.2f}%"
                else:
                    # R² — show raw value, mark negative clearly
                    marker = " ⚠️ (poor fit)" if v < 0 else ""
                    return f"{v:.4f}{marker}"

            rows = [{'Model': k,
                     f'{metric_name.upper()} Score': fmt_score(k,v) if v is not None else "Failed ❌"}
                    for k,v in results.items()]
            res_df = pd.DataFrame(rows)
            # Sort: put failures last, sort valid by raw score
            valid_rows   = [(k,v) for k,v in results.items() if v is not None]
            failed_rows  = [k for k,v in results.items() if v is None]
            sorted_keys  = [k for k,_ in sorted(valid_rows, key=lambda x:x[1], reverse=True)] + failed_rows
            res_df = res_df.set_index('Model').loc[sorted_keys].reset_index()
            _show_df(res_df, use_container_width=True)

            # ── Leaderboard chart (valid only) ─────────────────────────────
            if valid:
                sorted_r = dict(sorted(valid.items(), key=lambda x:x[1]))
                fig, ax  = plt.subplots(figsize=(10, max(3, len(sorted_r)*0.6+1)))
                _style_fig(fig); _style_ax(ax)
                bar_colors = sns.color_palette("rocket", len(sorted_r))

                if metric_name == 'accuracy':
                    bar_vals = [v*100 for v in sorted_r.values()]
                    xlabel   = "Accuracy (%)"
                    bar_labels = [f"{v*100:.1f}%" for v in sorted_r.values()]
                else:
                    bar_vals = list(sorted_r.values())
                    xlabel   = "R² Score"
                    bar_labels = [f"{v:.4f}" for v in sorted_r.values()]

                bars = ax.barh(list(sorted_r.keys()), bar_vals,
                               color=bar_colors, edgecolor='none', height=0.55)
                for bar, lbl in zip(bars, bar_labels):
                    xpos = bar.get_width()
                    ax.text(xpos + abs(max(bar_vals))*0.01,
                            bar.get_y()+bar.get_height()/2,
                            lbl, va='center', color='#333', fontsize=9)

                # Vertical line at 0 for R² to show negative clearly
                if metric_name == 'r2':
                    ax.axvline(0, color='#ef4444', lw=1.2, linestyle='--', alpha=0.7)

                ax.set_xlabel(xlabel, color='#555')
                ax.set_title("Model Leaderboard", color='#1e1b4b', fontsize=13)
                st.pyplot(fig); plt.close()

                if best_name:
                    bv = valid[best_name]
                    bv_str = f"{bv*100:.2f}%" if metric_name=='accuracy' else f"{bv:.4f}"
                    st.success(f"🏆 Best: **{best_name}** — {bv_str} {metric_name}")

            # ── PREDICTION WIDGET ─────────────────────────────────────────
            feat_cols    = st.session_state.get('train_feature_cols')
            tgt_col      = st.session_state.get('train_target_col')
            prob_type    = st.session_state.get('train_problem_type')
            trained_mds  = st.session_state.get('trained_models', {})
            src_for_pred = st.session_state.get('source_df_for_pred', source_df)

            if feat_cols and tgt_col and trained_mds and valid:
                st.divider()
                st.markdown('<div class="section-header">🔮 Predict on New Input</div>',
                            unsafe_allow_html=True)

                valid_model_names = [k for k,v in results.items() if v is not None]
                # FIX: use index+on_change pattern to avoid page-refresh-on-select
                if 'pred_model_idx' not in st.session_state:
                    st.session_state['pred_model_idx'] = 0

                model_choice = st.selectbox(
                    "Choose model for prediction",
                    valid_model_names,
                    key='pred_model_sel')

                st.markdown(f"**Enter feature values** *(model: {model_choice})*")
                st.caption("Sliders default to column mean. Selectboxes for low-cardinality features.")

                input_vals = {}
                cols_per_row = 3
                feat_chunks = [feat_cols[i:i+cols_per_row]
                               for i in range(0, len(feat_cols), cols_per_row)]

                for chunk in feat_chunks:
                    row_cols = st.columns(len(chunk))
                    for rc, fc in zip(row_cols, chunk):
                        col_data = src_for_pred[fc].dropna()
                        col_min  = float(col_data.min())
                        col_max  = float(col_data.max())
                        col_mean = float(col_data.mean())
                        n_unique = int(col_data.nunique())

                        with rc:
                            if n_unique <= 15:
                                unique_vals = sorted(col_data.unique().tolist())
                                closest = min(unique_vals, key=lambda x: abs(x-col_mean))
                                input_vals[fc] = st.selectbox(
                                    fc, unique_vals,
                                    index=unique_vals.index(closest),
                                    key=f'inp_{fc}')
                            else:
                                rng = col_max - col_min
                                step = round(rng/200, 6) if rng > 0 else 0.01
                                input_vals[fc] = st.slider(
                                    fc,
                                    min_value=float(round(col_min,4)),
                                    max_value=float(round(col_max,4)),
                                    value=float(round(col_mean,4)),
                                    step=float(step),
                                    key=f'inp_{fc}')

                # FIX: Predict button stores result in session_state so
                # changing the model selectbox/sliders doesn't wipe the output.
                pred_col1, pred_col2 = st.columns([1,4])
                if pred_col1.button("🔮 Predict", key='predict_btn', use_container_width=True):
                    model_to_use = trained_mds.get(model_choice)
                    if model_to_use is None:
                        st.error("Selected model not available.")
                    else:
                        input_array = np.array([[input_vals[fc] for fc in feat_cols]])
                        try:
                            prediction = model_to_use.predict(input_array)[0]
                            st.session_state['prediction_result'] = {
                                'model': model_choice,
                                'prediction': prediction,
                                'prob_type': prob_type,
                                'tgt_col': tgt_col,
                                'input_array': input_array,
                                'model_obj': model_to_use,
                            }
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            st.session_state['prediction_result'] = None

                # Display stored prediction result (persists across interactions)
                pred_res = st.session_state.get('prediction_result')
                if pred_res:
                    prediction   = pred_res['prediction']
                    p_prob_type  = pred_res['prob_type']
                    p_tgt_col    = pred_res['tgt_col']
                    p_arr        = pred_res['input_array']
                    p_model      = pred_res['model_obj']
                    p_model_name = pred_res['model']

                    st.markdown(f"*Result from **{p_model_name}***")

                    if p_prob_type == 'classification':
                        try:
                            proba   = p_model.predict_proba(p_arr)[0]
                            classes = p_model.classes_
                            st.success(f"**Predicted Class: `{int(prediction)}`**")
                            prob_df = pd.DataFrame({
                                'Class': [str(c) for c in classes],
                                'Probability (%)': [round(p*100,2) for p in proba]
                            }).sort_values('Probability (%)', ascending=False)
                            _show_df(prob_df, use_container_width=True)
                            fig, ax = plt.subplots(figsize=(8,2.5))
                            _style_fig(fig); _style_ax(ax,'Class Probabilities')
                            ax.bar(prob_df['Class'], prob_df['Probability (%)'],
                                   color=sns.color_palette("rocket",len(classes)),
                                   edgecolor='none')
                            ax.set_ylabel("Probability (%)", color='#555')
                            st.pyplot(fig); plt.close()
                        except Exception:
                            st.success(f"**Predicted Class: `{int(prediction)}`**")
                    else:
                        st.success(f"**Predicted Value: `{prediction:.4f}`**")
                        tgt_data = src_for_pred[p_tgt_col].dropna()
                        fig, ax  = plt.subplots(figsize=(8,3))
                        _style_fig(fig); _style_ax(ax,f"Prediction vs {p_tgt_col} Distribution")
                        ax.hist(tgt_data, bins=40, color='#c7d2fe', edgecolor='none', alpha=0.8)
                        ax.axvline(prediction, color='#4f46e5', lw=2.5,
                                   label=f'Predicted: {prediction:.3f}')
                        ax.legend(fontsize=9)
                        st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────
# TAB 6 — ASK YOUR DATA (pure statistical)
# ─────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">💬 Ask Your Data</div>', unsafe_allow_html=True)
    st.caption("Pure statistical Q&A — fully offline, no external API.")

    suggestions = generate_suggestions(df, 6)
    st.markdown("**💡 Quick questions:**")
    sug_cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        if sug_cols[i%3].button(sug, key=f'sug_{i}', use_container_width=True):
            st.session_state['chat_history'].append({"role":"user","content":sug})

    st.divider()
    user_q = st.chat_input("Ask something about your data…")
    if user_q:
        st.session_state['chat_history'].append({"role":"user","content":user_q})

    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            content = msg['content']
            if isinstance(content, pd.DataFrame):   _show_df(content, use_container_width=True)
            elif isinstance(content, plt.Figure):   st.pyplot(content); plt.close()
            else:                                   st.markdown(str(content))

    # ── Q&A engine ────────────────────────────────────────────────────────
    def find_col(q_lower, df):
        for col in sorted(df.columns, key=len, reverse=True):
            if col.lower() in q_lower: return col
        return None

    def qa_answer(q, df):
        ql = q.lower().strip()
        col = find_col(ql, df)

        if any(w in ql for w in ['average','mean']):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return (f"📊 Mean of **{col}**: **{df[col].mean():.4f}**\n\n"
                        f"*(Median: {df[col].median():.4f} | Std: {df[col].std():.4f})*")
            means = df.mean(numeric_only=True).round(4).reset_index()
            means.columns = ['Column','Mean']; return means

        if 'median' in ql:
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"📊 Median of **{col}**: **{df[col].median():.4f}**"
            meds = df.median(numeric_only=True).round(4).reset_index()
            meds.columns = ['Column','Median']; return meds

        if any(w in ql for w in ['std','standard deviation','deviation']):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"📊 Std Dev of **{col}**: **{df[col].std():.4f}**"
            stds = df.std(numeric_only=True).round(4).reset_index()
            stds.columns = ['Column','Std Dev']; return stds

        if any(w in ql for w in ['missing','null','nan','na','empty']):
            if col:
                m = int(df[col].isna().sum())
                return f"🔍 **{col}** has **{m}** missing values ({df[col].isna().mean()*100:.1f}%)."
            miss = df.isna().sum().reset_index()
            miss.columns = ['Column','Missing']
            miss['%'] = (miss['Missing']/len(df)*100).round(2)
            miss = miss[miss['Missing']>0].sort_values('Missing',ascending=False)
            return miss if not miss.empty else "✅ No missing values!"

        if any(w in ql for w in ['shape','size','dimension']):
            return f"📐 **{df.shape[0]:,} rows × {df.shape[1]} columns**"
        if 'row' in ql and 'column' not in ql:
            return f"📐 **{df.shape[0]:,} rows**"
        if 'column' in ql and 'row' not in ql:
            return f"📋 **{df.shape[1]} columns:** {', '.join(df.columns.tolist())}"

        if any(w in ql for w in ['distribution','histogram','distrib','hist']):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                fig,ax = plt.subplots(figsize=(8,3)); _style_fig(fig); _style_ax(ax,f'Distribution of {col}')
                data = df[col].dropna()
                ax.hist(data,bins=40,color='#667eea',alpha=0.85,edgecolor='none')
                ax.axvline(data.mean(),color='#ef4444',lw=1.5,label=f'Mean:{data.mean():.2f}')
                ax.axvline(data.median(),color='#22c55e',lw=1.5,linestyle='--',
                           label=f'Median:{data.median():.2f}')
                ax.legend(fontsize=8); return fig
            elif col:
                fig,ax = plt.subplots(figsize=(8,3)); _style_fig(fig); _style_ax(ax,f'Value Counts — {col}')
                vc = df[col].value_counts().head(15)
                ax.barh(vc.index.astype(str)[::-1],vc.values[::-1],color='#f093fb',edgecolor='none')
                return fig
            return "Please specify a column name — e.g. *Show distribution of Age*"

        if any(w in ql for w in ['summary','describe','statistics','stats']):
            if col:
                s = df[col].describe()
                lines = [f"**{col} Summary:**"]
                for k,v in s.items():
                    lines.append(f"- **{k}**: {v:.4f}" if isinstance(v,float) else f"- **{k}**: {v}")
                return '\n'.join(lines)
            return df.describe(include='number').round(3)

        if any(w in ql for w in ['correlation','corr','relate']):
            num_df = df.select_dtypes('number')
            if num_df.shape[1] < 2: return "Need at least 2 numeric columns."
            corr = num_df.corr().round(3)
            if col and col in corr:
                cc = corr[col].drop(col).sort_values(key=abs,ascending=False).reset_index()
                cc.columns = ['Feature',f'Correlation with {col}']; return cc
            return corr

        if any(w in ql for w in ['max','maximum','highest','largest']):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"📈 Max of **{col}**: **{df[col].max():.4f}** (row {df[col].idxmax()})"
            maxs = df.max(numeric_only=True).round(4).reset_index()
            maxs.columns = ['Column','Max']; return maxs

        if any(w in ql for w in ['min','minimum','lowest','smallest']):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"📉 Min of **{col}**: **{df[col].min():.4f}** (row {df[col].idxmin()})"
            mins = df.min(numeric_only=True).round(4).reset_index()
            mins.columns = ['Column','Min']; return mins

        if any(w in ql for w in ['unique','distinct','different']):
            if col:
                n = df[col].nunique()
                top = df[col].value_counts().head(5)
                out = f"🔢 **{col}** has **{n}** unique values.\n\nTop 5:\n"
                for val,cnt in top.items(): out += f"- `{val}` → {cnt}\n"
                return out
            uniq = pd.DataFrame({'Column':df.columns,
                                  'Unique':[df[c].nunique() for c in df.columns]})
            return uniq.sort_values('Unique',ascending=False)

        if any(w in ql for w in ['count','value count','frequency','freq']):
            if col:
                vc = df[col].value_counts().reset_index()
                vc.columns = [col,'Count']
                vc['%'] = (vc['Count']/len(df)*100).round(2)
                return vc.head(20)
            counts = pd.DataFrame({'Column':df.columns,
                                    'Non-Null':[df[c].notna().sum() for c in df.columns]})
            return counts

        if any(w in ql for w in ['outlier','anomaly','skew']):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                data = df[col].dropna()
                q1,q3 = data.quantile(0.25),data.quantile(0.75); iqr = q3-q1
                out = data[(data<q1-1.5*iqr)|(data>q3+1.5*iqr)]
                return (f"📦 **{col}** — IQR outlier analysis:\n"
                        f"- Q1:{q1:.4f} | Q3:{q3:.4f} | IQR:{iqr:.4f}\n"
                        f"- Fences: [{q1-1.5*iqr:.4f}, {q3+1.5*iqr:.4f}]\n"
                        f"- **{len(out)} outliers** ({len(out)/len(data)*100:.1f}%)\n"
                        f"- Skewness: {data.skew():.4f}")
            num_df = df.select_dtypes('number')
            if not num_df.empty:
                fig,ax = plt.subplots(figsize=(10,4)); _style_fig(fig)
                _style_ax(ax,"Boxplot — All Numeric Columns")
                num_df.boxplot(ax=ax,vert=False); return fig
            return "No numeric columns for outlier analysis."

        if any(w in ql for w in ['dtype','type','datatype']):
            if col: return f"🔠 **{col}** dtype: **{df[col].dtype}**"
            return pd.DataFrame({'Column':df.dtypes.index,'Dtype':df.dtypes.astype(str).values})

        if any(w in ql for w in ['duplicate','duplicates']):
            n = df.duplicated().sum()
            return f"🔁 **{n}** duplicate rows ({n/len(df)*100:.2f}%)"

        if 'variance' in ql:
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"📊 Variance of **{col}**: **{df[col].var():.4f}**"
            v = df.var(numeric_only=True).round(4).reset_index()
            v.columns = ['Column','Variance']; return v.sort_values('Variance',ascending=False)

        if any(w in ql for w in ['info','overview','about','tell me']):
            nc = df.select_dtypes('number').columns.tolist()
            cc = df.select_dtypes('object').columns.tolist()
            return (f"📊 **Dataset Overview:**\n"
                    f"- **{df.shape[0]:,}** rows, **{df.shape[1]}** columns\n"
                    f"- **{len(nc)}** numeric: {', '.join(nc[:8])}{'...' if len(nc)>8 else ''}\n"
                    f"- **{len(cc)}** categorical: {', '.join(cc[:8])}{'...' if len(cc)>8 else ''}\n"
                    f"- **{df.isna().sum().sum():,}** missing cells\n"
                    f"- **{df.duplicated().sum():,}** duplicate rows")

        return ("🤔 I can answer:\n"
                "- *average/mean/median/std of [column]*\n"
                "- *distribution of [column]*\n"
                "- *missing values in [column]*\n"
                "- *max/min of [column]*\n"
                "- *unique values in [column]*\n"
                "- *value counts of [column]*\n"
                "- *outliers in [column]*\n"
                "- *correlation / correlation with [column]*\n"
                "- *shape / columns / dtype*\n"
                "- *dataset overview*\n"
                "- *duplicates / variance of [column]*")

    if (st.session_state['chat_history'] and
        st.session_state['chat_history'][-1]['role'] == 'user'):
        q = st.session_state['chat_history'][-1]['content']
        try:
            answer = qa_answer(q, df)
        except Exception as e:
            answer = f"⚠️ Error: {e}"
        st.session_state['chat_history'].append({"role":"assistant","content":answer})
        with st.chat_message("assistant"):
            if isinstance(answer, pd.DataFrame):   _show_df(answer, use_container_width=True)
            elif isinstance(answer, plt.Figure):   st.pyplot(answer); plt.close()
            else:                                  st.markdown(str(answer))


# ── Footer ───────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#9ca3af;font-size:.82rem'>"
    "✨ Designed & Developed by <b>Ankit Mishra</b> · ML Studio · Powered by scikit-learn"
    "</center>", unsafe_allow_html=True)
