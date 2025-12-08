import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# === Настройка страницы ===
st.set_page_config(
    page_title="Монитор воздействия на окружающую среду",
    page_icon="🌍",
    layout="wide"
)

# === Кэшированная загрузка данных ===
@st.cache_data
def load_data():
    DATA_FILENAME = Path(__file__).parent / "data" / "GlobalTemperatures_Optimized_Half2_English.csv"
    df = pd.read_csv(DATA_FILENAME, header=None)
    df.columns = ["Date", "AverageTemperature", "UncertaintyAverageTemperature", "City", "Country", "Latitude", "Longitude"]

    # Обработка широты и долготы
    def parse_lat(lat_str):
        if 'N' in lat_str:
            return float(lat_str.replace('N', ''))
        elif 'S' in lat_str:
            return -float(lat_str.replace('S', ''))
        return float(lat_str)

    def parse_lon(lon_str):
        if 'E' in lon_str:
            return float(lon_str.replace('E', ''))
        elif 'W' in lon_str:
            return -float(lon_str.replace('W', ''))
        return float(lon_str)

    df['Latitude'] = df['Latitude'].apply(parse_lat)
    df['Longitude'] = df['Longitude'].apply(parse_lon)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hemisphere'] = df['Latitude'].apply(lambda x: 'Northern' if x >= 0 else 'Southern')
    df['LatZone'] = pd.cut(
        df['Latitude'],
        bins=[-90, -60, -30, 0, 30, 60, 90],
        labels=['Antarctic', 'South Temperate', 'Tropics South', 'Tropics North', 'North Temperate', 'Arctic']
    )
    return df

df = load_data()

# === Навигация ===
st.title("🌍 Монитор воздействия на окружающую среду")
st.markdown("Анализ глобальных температурных трендов по историческим данным.")
page = st.sidebar.radio("Навигация", ["1. Исходные данные", "2. Результаты анализа"])

# === СТРАНИЦА 1: Исследование данных ===
if page == "1. Исходные данные":
    st.header("🔍 Исследование исходных данных")

    # --- KPI ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего записей", df.shape[0])
    col2.metric("Города", df["City"].nunique())
    col3.metric("Страны", df["Country"].nunique())

    # Пропущенные значения
    st.write("**Пропущенные значения по колонкам:**")
    st.write(df.isnull().sum().to_dict())

    # --- Таблица ---
    st.subheader("Первые 10 строк датасета")
    st.dataframe(df.head(10), use_container_width=True)

    # --- Фильтры ---
    st.sidebar.subheader("Фильтры")
    countries = st.sidebar.multiselect("Выберите страны", options=df["Country"].unique(), default=[])
    years = st.sidebar.slider("Годы", int(df["Year"].min()), int(df["Year"].max()), (1900, 2010))

    # Применение фильтров
    filtered_df = df[
        (df["Year"] >= years[0]) & (df["Year"] <= years[1])
    ]
    if countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(countries)]

    # --- Распределения ---
    st.subheader("Распределения")
    fig1 = px.histogram(filtered_df, x="AverageTemperature", nbins=50, title="Распределение средней температуры")
    st.plotly_chart(fig1, use_container_width=True)

    country_counts = filtered_df["Country"].value_counts().head(20)
    fig2 = px.bar(country_counts, x=country_counts.index, y=country_counts.values, title="Топ-20 стран по числу записей")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Корреляция ---
    st.subheader("Корреляционная матрица")
    numeric_cols = filtered_df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig3 = px.imshow(corr, text_auto=True, title="Корреляция числовых признаков")
    st.plotly_chart(fig3, use_container_width=True)

    # --- Scatter plot ---
    st.subheader("Scatter: Температура vs Широта")
    fig4 = px.scatter(filtered_df, x="Latitude", y="AverageTemperature", color="Country", title="Зависимость температуры от широты")
    st.plotly_chart(fig4, use_container_width=True)

# === СТРАНИЦА 2: Анализ и экологические тренды ===
elif page == "2. Результаты анализа":
    st.header("📊 Результаты анализа: экологические тренды")

    # --- Фильтры анализа ---
    st.sidebar.subheader("Фильтры для анализа")
    countries_analysis = st.sidebar.multiselect("Страны", options=df["Country"].unique(), default=[])
    years_analysis = st.sidebar.slider("Годы анализа", int(df["Year"].min()), int(df["Year"].max()), (1850, 2010))

    # Фильтрация
    analysis_df = df[
        (df["Year"] >= years_analysis[0]) & (df["Year"] <= years_analysis[1])
    ]
    if countries_analysis:
        analysis_df = analysis_df[analysis_df["Country"].isin(countries_analysis)]

    # === 1. Глобальный тренд температуры ===
    st.subheader("1. Глобальный тренд температуры")
    yearly = analysis_df.groupby("Year")["AverageTemperature"].mean().reset_index()
    fig = px.line(yearly, x="Year", y="AverageTemperature", title="Средняя глобальная температура по годам")
    st.plotly_chart(fig, use_container_width=True)

    # === 2. Сезонность по месяцам ===
    st.subheader("2. Сезонность температуры по месяцам")
    monthly = analysis_df.groupby("Month")["AverageTemperature"].mean().reset_index()
    fig = px.line(monthly, x="Month", y="AverageTemperature", title="Средняя температура по месяцам")
    st.plotly_chart(fig, use_container_width=True)

    # === 3. Средняя температура по странам ===
    st.subheader("3. Средняя температура по странам")
    country_avg = analysis_df.groupby("Country")["AverageTemperature"].mean().sort_values(ascending=False).head(20).reset_index()
    fig = px.bar(country_avg, x="AverageTemperature", y="Country", orientation='h', title="Топ-20 стран по средней температуре")
    st.plotly_chart(fig, use_container_width=True)

    # === 4. Температура по полушариям ===
    st.subheader("4. Температура по полушариям")
    hemi_avg = analysis_df.groupby("Hemisphere")["AverageTemperature"].mean().reset_index()
    fig = px.bar(hemi_avg, x="Hemisphere", y="AverageTemperature", title="Средняя температура по полушариям")
    st.plotly_chart(fig, use_container_width=True)

    # === 5. Тепловая карта: Годы × Страны ===
    st.subheader("5. Тепловая карта: Годы × Страны")
    heatmap_data = analysis_df.groupby(["Year", "Country"])["AverageTemperature"].mean().unstack(fill_value=0)
    fig = px.imshow(heatmap_data.T, labels=dict(x="Год", y="Страна", color="Температура"), title="Тепловая карта: страны × годы")
    st.plotly_chart(fig, use_container_width=True)

    # === 6. Температура по широтным зонам ===
    st.subheader("6. Температура по широтным зонам")
    latzone_avg = analysis_df.groupby("LatZone")["AverageTemperature"].mean().reset_index()
    fig = px.bar(latzone_avg, x="LatZone", y="AverageTemperature", title="Средняя температура по широтным зонам")
    st.plotly_chart(fig, use_container_width=True)

    # === 7. Тепловая карта: Месяцы × Широтные зоны ===
    st.subheader("7. Тепловая карта: Месяцы × Широтные зоны")
    month_lat = analysis_df.groupby(["Month", "LatZone"])["AverageTemperature"].mean().unstack(fill_value=0)
    fig = px.imshow(month_lat.T, labels=dict(x="Месяц", y="Широтная зона", color="Температура"), title="Тепловая карта: месяцы × широтные зоны")
    st.plotly_chart(fig, use_container_width=True)

    # === Интерпретация ===
    st.info("""
    **Ключевые инсайты:**
    - Наблюдается устойчивый рост глобальной средней температуры с XIX века.
    - Четкая сезонность: пик — в июле (северное полушарие), минимум — в январе.
    - Тропические регионы имеют стабильно высокие температуры.
    - Арктическая зона демонстрирует наибольший рост температур за последние 50 лет.
    """)

# === Конец ===
