import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(path):
    '''Load dataset from path, URL, or uploaded file'''

    encodings = [
        "utf-8","utf-8-sig","cp1252","latin1","ISO-8859-1",
        "ISO-8859-15","cp1250","cp1251","cp1256",
        "utf-16","utf-16-le","utf-16-be"
    ]

    # ✅ Handle uploaded file
    if hasattr(path, "name"):
        file_name = path.name
    else:
        file_name = path

    extension = file_name.split('.')[-1].lower()

    for enc in encodings:
        try:
            # CSV
            if extension == 'csv':
                return pd.read_csv(path, encoding=enc)

            # Excel
            elif extension in ['xlsx', 'xls']:
                return pd.read_excel(path)

            # JSON
            elif extension == 'json':
                return pd.read_json(path)

            # XML
            elif extension == 'xml':
                return pd.read_xml(path)

            # URL handling
            elif isinstance(path, str) and path.startswith('https'):
                if 'github' in path:
                    return pd.read_csv(path + '?raw=true')
                elif 'docs.google' in path:
                    final_path = path[:path.rfind('/')] + '/' + \
                        'export?format=csv&gid=' + path[path.rfind('gid=')+4:]
                    return pd.read_csv(final_path)
                else:
                    return pd.read_html(path)[0]

        except:
            continue

    return pd.DataFrame()



def show_shape(df):
    return df.shape

def show_missing_values(df):
    return df.isna().sum().reset_index()

def cleaned_df(df):
    df = df.dropna()
    return df

def show_head(df):
    return df.head()

def show_tail(df):
    return df.tail()

def show_samples(df, v=5):
    return df.sample(v)

def describe_numerical(df):
    return df.describe(include='number').round(2)

def describe_textual(df):
    return df.describe(include='object').round(2)

def show_corr(df):
    plt.figure(figsize=(15,15))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True)
    
    # st.pyplot(plt.gcf())   # ✅ Updated
    plt.show()
    return corr

def show_pair_plot(df):
    num_df = df.select_dtypes('number')

    if num_df.shape[1] < 2:
        st.warning("Not enough numerical columns for pairplot.")
        return

    num_df = num_df.iloc[:, :5]
    sample_df = num_df.sample(min(200, len(num_df)))

    sns.pairplot(sample_df, corner=True, diag_kind='hist')
    # st.pyplot(plt.gcf())
    plt.show()

    
def show_columns(df):
    return df.columns

def show_textual_analysis(df):
    import random
    import math
    
    palettes = [
        "deep","muted","pastel","bright","dark","colorblind",
        "hsv","rainbow",
        "Blues","Greens","Reds","Purples","Oranges","Greys",
        "rocket","mako","flare","crest",
        "coolwarm","vlag","icefire","Spectral","RdBu","PiYG"
    ]
    
    text_df = df.select_dtypes('object')
    col_len = len(text_df.columns)

    fixed_col = 3
    row_fixed = math.ceil(col_len/fixed_col) 
    
    plt.figure(figsize=(40,row_fixed*10))
    
    for index,i in enumerate(text_df):
        plt.subplot(row_fixed,fixed_col, index+1)
        
        series = text_df[i].value_counts().head(10)
        x = [name[:25] for name in series.index]
        y = series.values
        
        plt.title(f'Analysis by {i}')
        
        chart = sns.barplot(x=x, y=y,
                            palette=sns.color_palette(random.choice(palettes)))
        
        for j in chart.containers:
            plt.bar_label(j)
        
        plt.xticks(rotation=15)
    
    # st.pyplot(plt.gcf())   # ✅ Updated
    plt.show()

def show_numerical_analysis(df):
    import random
    import math

    num_df = df.select_dtypes('number')
    col_len = len(num_df.columns)

    fixed_col = 3
    row_fixed = math.ceil(col_len/fixed_col) 
    
    plt.figure(figsize=(30,row_fixed*9))

    num_df_imp_cols = [i for i in num_df.columns if len(num_df[i].value_counts()) <= 1000]
    num_df = num_df[num_df_imp_cols]
    
    for index,i in enumerate(num_df):
        plt.subplot(row_fixed,fixed_col, index+1)
        
        plt.title(f'Distibution Analysis by {i}')
        
        sns.histplot(num_df[i], bins=100, kde=True,
                     color=random.choice(['r','orange','blue','green']))
        
        plt.xticks(rotation=15)
    
    # st.pyplot(plt.gcf())   # ✅ Updated
    plt.show()

def mannual_summary(df):
    desc = df.describe()
    
    st.markdown("## 📊 Dynamic Insights and Summary")   # Header
    
    for col in desc.columns:
        summary = f"""
### 🔹 {col}

- **Count:** {desc.loc['count', col]:.2f}  
- **Mean:** {desc.loc['mean', col]:.2f}  
- **Std Dev:** {desc.loc['std', col]:.2f}  
- **Minimum:** {desc.loc['min', col]:.2f}  
- **25th Percentile:** {desc.loc['25%', col]:.2f}  
- **Median (50%):** {desc.loc['50%', col]:.2f}  
- **75th Percentile:** {desc.loc['75%', col]:.2f}  
- **Maximum:** {desc.loc['max', col]:.2f}  

---
"""
        st.markdown(summary)

    st.markdown("### ✨ Design and Developed by Ankit")

# Generate suggestions based on columns


def generate_suggestions(df, n=5):
    import random
    suggestions = []
    cols = df.columns.tolist()

    if not cols:
        return suggestions

    # Pick up to n random columns (without repetition)
    sample_cols = random.sample(cols, min(n, len(cols)))

    # Generate natural language questions for each picked column
    for col in sample_cols:
        suggestions.append(f"What is the average of {col}?")
        suggestions.append(f"Show distribution of {col}?")
        suggestions.append(f"Are there any missing values in {col}?")
        suggestions.append(f"Give a summary of {col}.")

    # Shuffle and return top n suggestions
    random.shuffle(suggestions)
    return suggestions[:n]