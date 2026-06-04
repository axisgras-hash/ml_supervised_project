import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math, random, warnings
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

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── DAY MODE THEME ── */

/* Gradient header */
.main-title {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
}
/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(79,70,229,0.08);
}
.metric-card h2 { color: #4f46e5; margin: 0; font-size: 2rem; font-weight: 800; }
.metric-card p  { color: #6b7280; margin: 0; font-size: 0.82rem; letter-spacing: 1px; text-transform: uppercase; }
/* Section headers */
.section-header {
    border-left: 4px solid #4f46e5;
    padding-left: 12px;
    margin: 1.2rem 0 0.8rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e1b4b;
}
/* Suggestion chip */
.chip {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.82rem;
    color: #4f46e5;
    cursor: pointer;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f7ff 0%, #eef2ff 100%);
    border-right: 1px solid #e0e7ff;
}
/* Tab content */
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ai.py FUNCTIONS (inline — no file dependency)
# ──────────────────────────────────────────────

def load_dataset(path):
    encodings = ["utf-8","utf-8-sig","cp1252","latin1","ISO-8859-1","utf-16"]
    file_name = path.name if hasattr(path, "name") else str(path)
    ext = file_name.split('.')[-1].lower()
    for enc in encodings:
        try:
            if ext == 'csv':    return pd.read_csv(path, encoding=enc)
            elif ext in ['xlsx','xls']: return pd.read_excel(path)
            elif ext == 'json': return pd.read_json(path)
            elif ext == 'xml':  return pd.read_xml(path)
        except: continue
    return pd.DataFrame()

def generate_suggestions(df, n=5):
    cols = df.columns.tolist()
    if not cols: return []
    pool = []
    for col in random.sample(cols, min(n, len(cols))):
        pool += [
            f"What is the average of **{col}**?",
            f"Show distribution of **{col}**",
            f"Any missing values in **{col}**?",
            f"Give a summary of **{col}**",
        ]
    random.shuffle(pool)
    return pool[:n]

def show_textual_analysis(df):
    palettes = ["deep","muted","pastel","bright","rocket","mako","coolwarm","Spectral"]
    text_df = df.select_dtypes('object')
    col_len = len(text_df.columns)
    if col_len == 0:
        st.info("No categorical columns found.")
        return
    fixed_col = 3
    row_fixed = math.ceil(col_len / fixed_col)
    fig, axes = plt.subplots(row_fixed, fixed_col, figsize=(18, row_fixed * 5))
    fig.patch.set_facecolor('white')
    axes = axes.flatten() if row_fixed > 1 else [axes] if fixed_col == 1 else axes.flatten()
    for idx, col in enumerate(text_df.columns):
        ax = axes[idx]
        ax.set_facecolor('#f8f9ff')
        series = text_df[col].value_counts().head(10)
        x = [str(n)[:20] for n in series.index]
        y = series.values
        palette = sns.color_palette(random.choice(palettes), len(x))
        bars = ax.bar(x, y, color=palette, edgecolor='none', width=0.6)
        ax.set_title(f'{col}', fontsize=10, color='#1e1b4b', pad=8)
        ax.tick_params(colors='#555', labelsize=7)
        ax.set_xticklabels(x, rotation=20, ha='right')
        for bar, v in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(y)*0.01,
                    str(v), ha='center', va='bottom', fontsize=7, color='#1e1b4b')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.set_visible(False)
    for j in range(col_len, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

def show_numerical_analysis(df):
    colors = ['#667eea','#f093fb','#4facfe','#43e97b','#fa709a','#f6d365']
    num_df = df.select_dtypes('number')
    imp_cols = [c for c in num_df.columns if num_df[c].nunique() <= 1000]
    num_df = num_df[imp_cols]
    col_len = len(num_df.columns)
    if col_len == 0:
        st.info("No numeric columns found.")
        return
    fixed_col = 3
    row_fixed = math.ceil(col_len / fixed_col)
    fig, axes = plt.subplots(row_fixed, fixed_col, figsize=(18, row_fixed * 4))
    fig.patch.set_facecolor('white')
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for idx, col in enumerate(num_df.columns):
        ax = axes[idx]
        ax.set_facecolor('#f8f9ff')
        color = colors[idx % len(colors)]
        ax.hist(num_df[col].dropna(), bins=40, color=color, alpha=0.8, edgecolor='none')
        try:
            from scipy.stats import gaussian_kde
            data = num_df[col].dropna()
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(x_range, kde(x_range), color='#4f46e5', lw=1.5, alpha=0.7)
            ax2.set_yticks([])
            ax2.spines['right'].set_visible(False)
        except: pass
        ax.set_title(f'{col}', fontsize=9, color='#1e1b4b', pad=6)
        ax.tick_params(colors='#555', labelsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.set_visible(False)
    for j in range(col_len, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(pad=2)
    st.pyplot(fig); plt.close()

def show_corr(df):
    num_df = df.select_dtypes('number')
    if num_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns for correlation."); return
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(min(16, num_df.shape[1]*1.2+2), min(14, num_df.shape[1]*1.0+2)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9ff')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax,
                cmap='coolwarm', linewidths=0.5, linecolor='#e0e7ff',
                annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='#555', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    return corr

def show_pair_plot(df):
    num_df = df.select_dtypes('number').iloc[:, :5]
    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns for pairplot."); return
    sample_df = num_df.sample(min(300, len(num_df)))
    with st.spinner("Rendering pairplot…"):
        g = sns.pairplot(sample_df, corner=True, diag_kind='hist',
                         plot_kws={'alpha':0.5, 'color':'#667eea'},
                         diag_kws={'color':'#f093fb'})
        g.fig.patch.set_facecolor('white')
        st.pyplot(g.fig); plt.close()


# ──────────────────────────────────────────────
# ML PIPELINE HELPERS (from notebook)
# ──────────────────────────────────────────────

def identify_nan_cols(df, threshold=0.5):
    return [c for c, v in df.isna().mean().items() if v > threshold]

def fill_na(series, method='median'):
    if pd.api.types.is_numeric_dtype(series):
        if method == 'mean':   return series.fillna(series.mean())
        elif method == 'median': return series.fillna(series.median())
        else: return series.fillna(0)
    elif pd.api.types.is_object_dtype(series):
        return series.fillna(series.mode()[0] if not series.mode().empty else 'Unknown')
    return series

def fix_fill_na(df, thresh=0.5, method='mean'):
    df = df.copy()
    for col in identify_nan_cols(df, thresh):
        df[col] = fill_na(df[col], method)
    return df

def drop_cols(df, cols):
    valid = [c for c in cols if c in df.columns]
    return df.drop(valid, axis=1)

def textual_to_numerical(df, method='le'):
    from sklearn.preprocessing import LabelEncoder
    tdf = df.copy()
    text_cols = tdf.select_dtypes('object').columns
    if method == 'le':
        le = LabelEncoder()
        for col in text_cols:
            tdf[col] = le.fit_transform(tdf[col].astype(str))
    elif method == 'get_dummies':
        dummies = pd.get_dummies(tdf[text_cols], dtype=int, drop_first=True)
        tdf = pd.concat([tdf.drop(text_cols, axis=1), dummies], axis=1)
    return tdf

def change_data_scale(df, method='min_max'):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    tdf = df.drop(date_cols, axis=1) if date_cols else df.copy()
    num_cols = tdf.select_dtypes('number').columns
    scaler = MinMaxScaler() if method == 'min_max' else StandardScaler()
    tdf[num_cols] = scaler.fit_transform(tdf[num_cols])
    if date_cols:
        for dc in date_cols: tdf[dc] = df[dc]
    return tdf

def run_models(X, y, problem_type='classification'):
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(max_depth=6),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50),
        }
        metric = 'accuracy'
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(max_depth=6),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50),
        }
        metric = 'r2'

    results = {}
    progress = st.progress(0, text="Training models…")
    for i, (name, model) in enumerate(models.items()):
        try:
            score = cross_val_score(model, X, y, cv=3, scoring=metric).mean()
            results[name] = round(score * 100, 2)
        except Exception as e:
            results[name] = 0.0
        progress.progress((i+1)/len(models), text=f"Trained: {name}")
    progress.empty()
    return results, metric


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if 'df' not in st.session_state:          st.session_state['df'] = None
if 'cleaned_df' not in st.session_state:  st.session_state['cleaned_df'] = None
if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
if 'model_results' not in st.session_state: st.session_state['model_results'] = None
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Studio")
    st.markdown("**by Ankit** · Powered by sklearn + ai.py")
    st.divider()

    st.markdown("### 📁 Load Dataset")
    source = st.radio("Source", ["Upload File", "Paste URL"], horizontal=True)

    df = None
    if source == "Upload File":
        file = st.file_uploader("CSV / Excel / JSON / XML", type=["csv","xlsx","xls","json","xml"])
        if file:
            with st.spinner("Loading…"):
                df = load_dataset(file)
    else:
        url = st.text_input("Dataset URL")
        if url:
            try:
                df = pd.read_csv(url) if 'csv' in url else pd.read_json(url)
            except: st.error("Could not load from URL.")

    if df is not None and not df.empty:
        st.session_state['df'] = df
        st.success(f"✅ Loaded — {df.shape[0]:,} rows × {df.shape[1]} cols")

    st.divider()
    if st.session_state['df'] is not None:
        st.markdown("### 📊 Dataset Info")
        d = st.session_state['df']
        st.markdown(f"- **Rows:** {d.shape[0]:,}")
        st.markdown(f"- **Cols:** {d.shape[1]}")
        st.markdown(f"- **Numeric:** {len(d.select_dtypes('number').columns)}")
        st.markdown(f"- **Categorical:** {len(d.select_dtypes('object').columns)}")
        miss = d.isna().sum().sum()
        st.markdown(f"- **Missing cells:** {miss:,}")

    st.divider()
    st.markdown("### ⚙️ Settings")
    palette_theme = st.selectbox("Plot palette", ["deep","muted","bright","pastel","rocket","mako"])
    sns.set_palette(palette_theme)


# ──────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────
st.markdown('<h1 class="main-title">🤖 ML Studio</h1>', unsafe_allow_html=True)
st.markdown("**Explore · Analyze · Preprocess · Train** — all in one interactive dashboard")

if st.session_state['df'] is None:
    st.info("👈 Upload a dataset or paste a URL in the sidebar to get started.")
    st.stop()

df = st.session_state['df']

# ──────────────────────────────────────────────
# TOP METRIC CARDS
# ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (df.shape[0], "Rows"),
    (df.shape[1], "Columns"),
    (df.isna().sum().sum(), "Missing Cells"),
    (len(df.select_dtypes('number').columns), "Numeric Cols"),
]
for col, (val, label) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <h2>{val:,}</h2>
        <p>{label}</p>
    </div>""", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tabs = st.tabs([
    "🗂️ Data Explorer",
    "📊 EDA",
    "🔗 Correlations",
    "🛠️ Preprocessing",
    "🤖 Train Models",
    "💬 Ask Your Data",
])


# ═══════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ═══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)

    view_mode = st.radio("View", ["Head","Tail","Random Sample","Full"], horizontal=True)
    n_rows = st.slider("Rows to display", 5, 50, 10)

    if view_mode == "Head":        preview = df.head(n_rows)
    elif view_mode == "Tail":      preview = df.tail(n_rows)
    elif view_mode == "Random Sample": preview = df.sample(min(n_rows, len(df)))
    else:                          preview = df

    st.dataframe(preview, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Shape</div>', unsafe_allow_html=True)
        st.write(f"**{df.shape[0]:,}** rows × **{df.shape[1]}** columns")

    with col_b:
        st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({'Column': df.dtypes.index, 'Type': df.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True, height=200)

    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    miss = df.isna().sum().reset_index()
    miss.columns = ['Column','Missing']
    miss['Percent'] = (miss['Missing'] / len(df) * 100).round(2)
    miss = miss[miss['Missing'] > 0].sort_values('Missing', ascending=False)
    if miss.empty:
        st.success("🎉 No missing values!")
    else:
        st.dataframe(miss, use_container_width=True)

    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    tab_num, tab_cat = st.tabs(["Numerical", "Categorical"])
    with tab_num:
        num_desc = df.describe(include='number').round(2)
        if not num_desc.empty: st.dataframe(num_desc, use_container_width=True)
        else: st.info("No numeric columns.")
    with tab_cat:
        cat_desc = df.describe(include='object')
        if not cat_desc.empty: st.dataframe(cat_desc, use_container_width=True)
        else: st.info("No categorical columns.")

    st.markdown('<div class="section-header">Column-level Deep Dive</div>', unsafe_allow_html=True)
    chosen_col = st.selectbox("Pick a column", df.columns)
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Values", df[chosen_col].nunique())
    col2.metric("Missing", int(df[chosen_col].isna().sum()))
    col3.metric("Dtype", str(df[chosen_col].dtype))

    if pd.api.types.is_numeric_dtype(df[chosen_col]):
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9ff')
        ax.hist(df[chosen_col].dropna(), bins=40, color='#667eea', alpha=0.85, edgecolor='none')
        ax.set_title(f'Distribution — {chosen_col}', color='#1e1b4b')
        ax.tick_params(colors='#555')
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig); plt.close()
    else:
        vc = df[chosen_col].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9ff')
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color='#f093fb', edgecolor='none')
        ax.set_title(f'Top Values — {chosen_col}', color='#1e1b4b')
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values(): spine.set_visible(False)
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda_choice = st.radio("Choose analysis",
        ["📊 Categorical Analysis","📈 Numerical Distributions","👫 Pairplot"],
        horizontal=True)

    if eda_choice == "📊 Categorical Analysis":
        if df.select_dtypes('object').shape[1] == 0:
            st.info("No categorical columns found.")
        else:
            with st.spinner("Generating charts…"):
                show_textual_analysis(df)

    elif eda_choice == "📈 Numerical Distributions":
        with st.spinner("Generating distributions…"):
            show_numerical_analysis(df)

    elif eda_choice == "👫 Pairplot":
        n_cols = min(5, len(df.select_dtypes('number').columns))
        st.info(f"Showing up to **{n_cols}** numeric columns (sampled 300 rows for speed)")
        show_pair_plot(df)

    # Dynamic per-column plot
    st.divider()
    st.markdown('<div class="section-header">Custom Plot Builder</div>', unsafe_allow_html=True)
    x_col = st.selectbox("X axis", df.columns, key='x_col')
    y_options = [None] + [c for c in df.columns if c != x_col]
    y_col = st.selectbox("Y axis (optional)", y_options, key='y_col')
    plot_type = st.selectbox("Plot type", ["Histogram","Bar","Box","Scatter","Line","Violin"])

    if st.button("🎨 Generate Plot"):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9ff')
        try:
            if plot_type == "Histogram":
                ax.hist(df[x_col].dropna(), bins=40, color='#667eea', alpha=0.85)
            elif plot_type == "Bar" and y_col:
                df.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax, color='#f093fb')
            elif plot_type == "Box":
                if y_col: sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                else: sns.boxplot(data=df, y=x_col, ax=ax, color='#4facfe')
            elif plot_type == "Scatter" and y_col:
                ax.scatter(df[x_col], df[y_col], color='#43e97b', alpha=0.4, s=20)
                ax.set_xlabel(x_col, color='#555'); ax.set_ylabel(y_col, color='#555')
            elif plot_type == "Line" and y_col:
                ax.plot(df[x_col], df[y_col], color='#667eea', lw=1.5)
            elif plot_type == "Violin":
                if y_col: sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
                else: sns.violinplot(data=df, y=x_col, ax=ax, color='#fa709a')
            ax.set_title(f'{plot_type}: {x_col}', color='#1e1b4b')
            ax.tick_params(colors='#555')
            for spine in ax.spines.values(): spine.set_visible(False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not render: {e}")
        plt.close()


# ═══════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ═══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes('number').columns.tolist()
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns.")
    else:
        selected_corr_cols = st.multiselect("Select columns (default = all numeric)",
                                            num_cols, default=num_cols[:min(10, len(num_cols))])
        if len(selected_corr_cols) >= 2:
            corr_result = show_corr(df[selected_corr_cols])

            st.markdown('<div class="section-header">Top Correlated Pairs</div>', unsafe_allow_html=True)
            corr_pairs = (corr_result.abs()
                          .where(np.tril(np.ones(corr_result.shape), k=-1).astype(bool))
                          .stack().sort_values(ascending=False).head(10).reset_index())
            corr_pairs.columns = ['Feature A','Feature B','|Correlation|']
            corr_pairs['|Correlation|'] = corr_pairs['|Correlation|'].round(3)
            st.dataframe(corr_pairs, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 4 — PREPROCESSING
# ═══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.info("Configure each step and click **Apply Pipeline** to create a processed dataset.")

    work_df = df.copy()

    # Step 1 — Drop columns
    st.markdown("**Step 1 — Drop Columns**")
    cols_to_drop = st.multiselect("Select columns to drop", work_df.columns)

    # Step 2 — Missing value treatment
    st.markdown("**Step 2 — Handle Missing Values**")
    miss_thresh = st.slider("Drop columns with missing % above", 0, 100, 50)
    fill_method = st.radio("Fill strategy for remaining NaNs", ["mean","median","drop rows"], horizontal=True)

    # Step 3 — Encoding
    st.markdown("**Step 3 — Encode Categorical Columns**")
    enc_method = st.radio("Encoding method", ["Label Encoding","One-Hot (get_dummies)","None"], horizontal=True)

    # Step 4 — Scaling
    st.markdown("**Step 4 — Feature Scaling**")
    scale_method = st.radio("Scaling", ["Min-Max","Standard Scaler","None"], horizontal=True)

    if st.button("⚙️ Apply Pipeline"):
        with st.spinner("Processing…"):
            result = work_df.copy()

            # Drop selected cols
            if cols_to_drop:
                result = drop_cols(result, cols_to_drop)

            # Drop high-missing cols
            high_miss = [c for c in result.columns if result[c].isna().mean()*100 > miss_thresh]
            if high_miss:
                result = drop_cols(result, high_miss)
                st.warning(f"Dropped {len(high_miss)} high-missing column(s): {high_miss}")

            # Fill NaNs
            if fill_method in ['mean','median']:
                result = fix_fill_na(result, thresh=-1, method=fill_method)  # fill all
            elif fill_method == 'drop rows':
                result = result.dropna()

            # Encode
            if enc_method == "Label Encoding":
                result = textual_to_numerical(result, method='le')
            elif enc_method == "One-Hot (get_dummies)":
                result = textual_to_numerical(result, method='get_dummies')

            # Scale
            if scale_method != "None":
                try:
                    method_key = 'min_max' if scale_method == 'Min-Max' else 'ss'
                    result = change_data_scale(result, method=method_key)
                except Exception as e:
                    st.warning(f"Scaling skipped: {e}")

            st.session_state['processed_df'] = result
            st.success(f"✅ Pipeline done! Shape: {result.shape[0]:,} rows × {result.shape[1]} cols")
            st.dataframe(result.head(10), use_container_width=True)

            # Download
            csv = result.to_csv(index=False).encode()
            st.download_button("⬇️ Download Processed CSV", csv, "processed_data.csv", "text/csv")


# ═══════════════════════════════════════════════
# TAB 5 — TRAIN MODELS
# ═══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">AutoML — Train & Compare Models</div>', unsafe_allow_html=True)

    _proc = st.session_state.get('processed_df')
    source_df = _proc if _proc is not None else df
    src_label = "✅ Using preprocessed dataset" if st.session_state.get('processed_df') is not None else "⚠️ Using raw dataset (consider preprocessing first)"
    st.info(src_label)

    num_only = source_df.select_dtypes('number')
    if num_only.shape[1] < 2:
        st.warning("Need at least 2 numeric columns (features + target).")
    else:
        target_col = st.selectbox("🎯 Target Column", num_only.columns)
        problem_type = st.radio("Problem Type", ["classification","regression"], horizontal=True)
        cv_folds = st.slider("Cross-validation folds", 2, 10, 3)

        feature_cols = st.multiselect("Feature Columns (default = all except target)",
                                      [c for c in num_only.columns if c != target_col],
                                      default=[c for c in num_only.columns if c != target_col])

        if st.button("🚀 Train All Models"):
            if not feature_cols:
                st.error("Select at least one feature column.")
            else:
                X = num_only[feature_cols].dropna()
                y = num_only.loc[X.index, target_col]

                with st.spinner("Training in progress…"):
                    results, metric_name = run_models(X, y, problem_type)
                st.session_state['model_results'] = results

                st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
                res_df = pd.DataFrame({'Model': list(results.keys()),
                                       f'{metric_name.capitalize()} (%)': list(results.values())})
                res_df = res_df.sort_values(f'{metric_name.capitalize()} (%)', ascending=False).reset_index(drop=True)
                st.dataframe(res_df, use_container_width=True)

                # Bar chart
                fig, ax = plt.subplots(figsize=(9, 4))
                fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9ff')
                colors_bar = sns.color_palette("rocket", len(res_df))
                bars = ax.barh(res_df['Model'], res_df[f'{metric_name.capitalize()} (%)'],
                               color=colors_bar, edgecolor='none', height=0.55)
                for bar, v in zip(bars, res_df[f'{metric_name.capitalize()} (%)']):
                    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                            f'{v:.1f}%', va='center', color='#333', fontsize=9)
                ax.set_xlim(0, max(res_df[f'{metric_name.capitalize()} (%)']) * 1.15)
                ax.set_xlabel(f'{metric_name.capitalize()} (%)', color='#555')
                ax.tick_params(colors='#555')
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.set_title("Model Leaderboard", color='#1e1b4b', fontsize=13)
                st.pyplot(fig); plt.close()

                best = res_df.iloc[0]
                st.success(f"🏆 Best Model: **{best['Model']}** with **{best[f'{metric_name.capitalize()} (%)']:.2f}%** {metric_name}")


# ═══════════════════════════════════════════════
# TAB 6 — ASK YOUR DATA
# ═══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">💬 Ask Your Data</div>', unsafe_allow_html=True)
    st.markdown("Explore your dataset with natural language queries.")

    # Suggestions
    suggestions = generate_suggestions(df, 5)
    st.markdown("**💡 Try these questions:**")
    cols_sug = st.columns(len(suggestions))
    for i, (col, sug) in enumerate(zip(cols_sug, suggestions)):
        if col.button(sug.replace("**",""), key=f"sug_{i}"):
            st.session_state['chat_history'].append({"role":"user","content": sug.replace("**","")})

    st.divider()

    # Chat input
    user_q = st.chat_input("Ask something about your data…")
    if user_q:
        st.session_state['chat_history'].append({"role":"user","content": user_q})

    # Render chat history
    for msg in st.session_state['chat_history']:
        with st.chat_message(msg['role']):
            if msg['role'] == 'user':
                st.write(msg['content'])
            else:
                if isinstance(msg['content'], pd.DataFrame):
                    st.dataframe(msg['content'], use_container_width=True)
                elif isinstance(msg['content'], plt.Figure):
                    st.pyplot(msg['content'])
                else:
                    st.write(msg['content'])

    # Auto-answer last user message
    if st.session_state['chat_history'] and st.session_state['chat_history'][-1]['role'] == 'user':
        q = st.session_state['chat_history'][-1]['content'].lower()
        answer = None

        # Rule-based Q&A engine
        if 'average' in q or 'mean' in q:
            for col in df.columns:
                if col.lower() in q and pd.api.types.is_numeric_dtype(df[col]):
                    answer = f"📊 The average of **{col}** is **{df[col].mean():.4f}**"
                    break
            if not answer:
                answer = "📊 **Means of all numeric columns:**\n\n" + df.mean(numeric_only=True).round(3).to_markdown()

        elif 'missing' in q or 'null' in q or 'na' in q:
            miss = df.isna().sum()
            answer = miss[miss > 0].reset_index().rename(columns={'index':'Column',0:'Missing'})
            if isinstance(answer, pd.DataFrame) and answer.empty:
                answer = "✅ No missing values found!"

        elif 'shape' in q or 'size' in q or 'rows' in q or 'columns' in q:
            answer = f"📐 Dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**."

        elif 'distribution' in q or 'histogram' in q or 'distrib' in q:
            for col in df.columns:
                if col.lower() in q and pd.api.types.is_numeric_dtype(df[col]):
                    fig, ax = plt.subplots(figsize=(7,3))
                    fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9ff')
                    ax.hist(df[col].dropna(), bins=40, color='#667eea', alpha=0.85)
                    ax.set_title(f'Distribution — {col}', color='#1e1b4b')
                    ax.tick_params(colors='#555')
                    for sp in ax.spines.values(): sp.set_visible(False)
                    answer = fig
                    break
            if not answer:
                answer = "Please specify a numeric column name, e.g. *Show distribution of Age*"

        elif 'summary' in q or 'describe' in q or 'statistics' in q:
            for col in df.columns:
                if col.lower() in q:
                    s = df[col].describe()
                    answer = f"**{col} Summary:**\n" + s.round(3).to_markdown()
                    break
            if not answer:
                answer = df.describe(include='number').round(2)

        elif 'correlation' in q or 'corr' in q:
            answer = df.corr(numeric_only=True).round(3)

        elif 'column' in q or 'feature' in q or 'fields' in q:
            answer = f"📋 Columns ({df.shape[1]}): **{', '.join(df.columns.tolist())}**"

        elif 'unique' in q:
            for col in df.columns:
                if col.lower() in q:
                    answer = f"🔢 **{col}** has **{df[col].nunique()}** unique values."
                    break
            if not answer:
                uniq = pd.Series({c: df[c].nunique() for c in df.columns}, name='Unique Values')
                answer = uniq.reset_index().rename(columns={'index':'Column'})

        elif 'max' in q or 'maximum' in q:
            for col in df.columns:
                if col.lower() in q and pd.api.types.is_numeric_dtype(df[col]):
                    answer = f"📈 Maximum of **{col}** = **{df[col].max():.4f}**"; break
            if not answer:
                answer = "Please specify a numeric column, e.g. *What is the max of Age?*"

        elif 'min' in q or 'minimum' in q:
            for col in df.columns:
                if col.lower() in q and pd.api.types.is_numeric_dtype(df[col]):
                    answer = f"📉 Minimum of **{col}** = **{df[col].min():.4f}**"; break
            if not answer:
                answer = "Please specify a numeric column, e.g. *What is the min of Salary?*"

        else:
            answer = ("🤔 I can answer questions like:\n"
                      "- *What is the average of [column]?*\n"
                      "- *Show distribution of [column]*\n"
                      "- *Any missing values?*\n"
                      "- *What is the shape?*\n"
                      "- *Unique values in [column]*\n"
                      "- *Max/min of [column]*\n"
                      "- *Give a summary of [column]*\n"
                      "- *Show correlation*")

        if answer is not None:
            st.session_state['chat_history'].append({"role":"assistant","content": answer})
            with st.chat_message("assistant"):
                if isinstance(answer, pd.DataFrame):
                    st.dataframe(answer, use_container_width=True)
                elif isinstance(answer, plt.Figure):
                    st.pyplot(answer); plt.close()
                else:
                    st.markdown(str(answer))

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#555; font-size:0.8rem'>✨ Designed & Developed by <b>Ankit</b> · ML Studio</center>",
    unsafe_allow_html=True
)