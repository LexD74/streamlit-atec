# Перед запуском установите необходимые библиотеки:
# pip install streamlit pandas numpy plotly statsmodels requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import requests
import io
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы ДОЛЖНА БЫТЬ ПЕРВОЙ
st.set_page_config(
    page_title="АТЭЦ Аналитика",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Текстовые данные из документа (для дефолта) - выносим в отдельную переменную
TEXT_DATA = """
row3: ТГ-1,0.7328471198156682,0.8120683673469389,0.775663133640553,0.7069816666666666,0.6136569124423963,0.23684261904761905,0.5779423963133641,0,0,0,0,0,0.3691946281800391
row4: ТГ-2,0.8102637096774193,0.944525,0.8871923963133641,0.3771666666666667,0.5168207373271889,0.28462190476190474,0.5132366359447005,0,0,0,0,0,0.3585021232876712
row5: ТГ-3,0.3024148233486943,0.9418595238095238,0.6418884792626729,0.4908720634920635,0.3746497695852534,0.6992368253968254,0.6340451612903226,0,0,0,0,0,0.3359404892367906
row6: ТГ-4,0.7797910497091485,0.057489510148321626,0.37599365415124275,0.8467934426229508,0.5802938921205711,0.7388691256830602,0.5944280803807509,0,0,0,0,0,0.3326720282206752
row7: ТГ-5,0,0,0,0,0,0.00047583333333333337,0.0006647849462365591,0,0,0,0,0,0.00009557077625570777
row8: ТГ-6,0.012135618279569892,0.0005714285714285714,0,0,0,0,0,0,0,0,0,0,0.0010745319634703197
row9: ТГ-7,0.8838499999999999,0.9378366071428571,0.7883255376344086,0.38926083333333333,0.5079059139784946,0.2152377777777778,0,0,0,0,0,0,0.3067860502283105
row12: ТГ-1,19083.339,19099.848,20198.268,17815.938,15979.626,5968.434,15049.62,,,,,,35
row13: ТГ-2,21099.267,22215.228,23102.49,9504.6,13458.012,7172.472,13364.682,,,,,,35
row14: ТГ-3,7874.882,22152.536,16714.776,12369.976,9755.88,17620.768,16510.536,,,,,,35
row15: ТГ-4,35390.037,2356.61,17064.096,37191.168,26336.058,32451.132,26977.524,,,,,,61
row16: ТГ-5,0,0,0,0,0,13.704,19.784,,,,,,40
row17: ТГ-6,180.578,7.68,0,0,0,0,0,,,,,,20
row18: ТГ-7,19727.532,18906.786,17595.426,8408.034,11336.46,4649.136,0,,,,,,30
"""

months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

# Функция для расчета КИУМ из выработки
def calculate_kium_from_generation(gen_df, hours_df):
    """Рассчитывает КИУМ на основе данных выработки и установленной мощности"""
    kium_data = []
    
    for _, row in gen_df.iterrows():
        tg = row['ТГ']
        n_ust = row['N_уст']
        kium_row = {'ТГ': tg}
        
        monthly_kium = []
        for i, month in enumerate(months):
            generation = row[month]
            hours = hours_per_month[i]
            # КИУМ = Выработка / (N_уст × Часы)
            if n_ust > 0 and hours > 0:
                kium = generation / (n_ust * hours)
            else:
                kium = 0.0
            kium_row[month] = kium
            monthly_kium.append(kium)
        
        # Общий КИУМ как среднее за год
        kium_row['КИУМ_общий'] = np.mean(monthly_kium)
        kium_data.append(kium_row)
    
    return pd.DataFrame(kium_data)

# Функция для парсинга данных из текстового представления
def parse_data_from_text(text_data):
    lines = text_data.strip().split('\n')
    
    # Парсим ВЫРАБОТКУ МВт*ч: rows 12-18 (это первичные данные)
    generation_data = {}
    for line in lines:
        if ':' in line:
            row_part = line.split(':')[0]
            if row_part.startswith('row') and row_part[3:].isdigit():
                row_num = int(row_part[3:])
                if 12 <= row_num <= 18:
                    content = line.split(':')[1].strip()
                    parts = [p.strip() for p in content.split(',')]
                    if len(parts) > 0 and parts[0].startswith('ТГ-'):
                        tg = parts[0]
                        values = []
                        for i in range(1, 13):
                            if i < len(parts):
                                try:
                                    values.append(float(parts[i]))
                                except ValueError:
                                    values.append(0.0)
                            else:
                                values.append(0.0)
                        generation_data[tg] = {months[i]: values[i] for i in range(12)}
                        if len(parts) > 13:
                            generation_data[tg]['N_уст'] = float(parts[13])
                        else:
                            generation_data[tg]['N_уст'] = 0
    
    # DataFrame для выработки (первичные данные)
    gen_data = []
    for tg, data in generation_data.items():
        row = {'ТГ': tg}
        for month in months:
            row[month] = data.get(month, 0)
        row['N_уст'] = data.get('N_уст', 0)
        gen_data.append(row)
    gen_df = pd.DataFrame(gen_data)
    
    # Часы в месяцах
    hours_df = pd.DataFrame({'Месяц': months, 'Часы': hours_per_month})
    
    # РАССЧИТЫВАЕМ КИУМ из выработки (вместо загрузки из строк 3-9)
    kium_df = calculate_kium_from_generation(gen_df, hours_df)
    
    return kium_df, gen_df, hours_df

# Кэшируем функцию парсинга
@st.cache_data
def get_parsed_data():
    return parse_data_from_text(TEXT_DATA)

# Профессиональный CSS для строгого интерфейса
st.markdown("""
    <style>
    /* Основные настройки */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Заголовки */
    h1 {
        color: #1f1f1f;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }
    
    h2, h3 {
        color: #2c2c2c;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Сайдбар */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Кнопки */
    .stButton > button {
        width: 100%;
        border-radius: 4px;
        border: 1px solid #d0d0d0;
        background-color: white;
        color: #1f1f1f;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f0f0f0;
        border-color: #a0a0a0;
    }
    
    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(0,0,0,0.1);
    }
    
    /* Селекты и инпуты */
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        background-color: white;
    }
    
    /* Метрики */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 600;
        color: #1f1f1f;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #666;
    }
    
    /* Таблицы */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #f8f9fa;
        font-weight: 600;
        text-align: left;
        padding: 0.75rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .dataframe td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .dataframe tr:hover {
        background-color: #f8f9fa;
    }
    
    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f1f1f;
        color: white;
    }
    
    /* Графики контейнеры */
    .plotly-chart {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 1rem;
        background-color: white;
    }
    
    /* Уведомления */
    .stAlert {
        border-radius: 6px;
        border: 1px solid;
    }
    
    /* Мобильная адаптация */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .stButton > button {
            margin: 0.25rem 0;
        }
        
        [data-testid="stMetric"] {
            margin: 0.5rem 0;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Инициализация session state
if 'kium_df' not in st.session_state:
    kium_df, gen_df, hours_df = get_parsed_data()
    st.session_state.kium_df = kium_df
    st.session_state.gen_df = gen_df
    st.session_state.hours_df = hours_df

# Заголовок с логотипом
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("### ⚡")
with col2:
    st.title("АТЭЦ - Анализ КИУМ")
    st.caption("КИУМ рассчитывается на основе данных выработки и установленной мощности")

# Боковая панель - настройки
with st.sidebar:
    st.header("Настройки данных")
    
    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузить Excel файл", type=['xlsx'], 
                                   help="Загрузите файл с данными за 2025 год")
    
    if uploaded_file is not None:
        try:
            # Загружаем данные выработки (первичные)
            gen_df = pd.read_excel(uploaded_file, sheet_name='2025', skiprows=10, nrows=7, usecols='A:N')
            gen_df.columns = ['ТГ'] + months + ['N_уст']
            
            # РАССЧИТЫВАЕМ КИУМ из выработки
            kium_df = calculate_kium_from_generation(gen_df, pd.DataFrame({'Месяц': months, 'Часы': hours_per_month}))
            
            hours_df = pd.DataFrame({'Месяц': months, 'Часы': hours_per_month})
            
            st.session_state.kium_df = kium_df
            st.session_state.gen_df = gen_df
            st.session_state.hours_df = hours_df
            st.success("Данные успешно загружены и КИУМ пересчитан")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    
    st.divider()
    st.header("Фильтры")
    
    # Фильтры
    selected_tgs = st.multiselect(
        "Выберите ТГ", 
        st.session_state.kium_df['ТГ'].unique(), 
        default=st.session_state.kium_df['ТГ'].unique(),
        help="Выберите турбогенераторы для анализа"
    )
    
    selected_months = st.multiselect(
        "Выберите месяцы", 
        months, 
        default=months,
        help="Выберите месяцы для анализа"
    )

# Основной контент
kium_df = st.session_state.kium_df
gen_df = st.session_state.gen_df
hours_df = st.session_state.hours_df

# Фильтрация данных
filtered_kium = kium_df[kium_df['ТГ'].isin(selected_tgs)][['ТГ'] + selected_months + ['КИУМ_общий']]
filtered_gen = gen_df[gen_df['ТГ'].isin(selected_tgs)][['ТГ'] + selected_months + ['N_уст']]

# Навигация через вкладки
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 КИУМ", 
    "⚡ Выработка", 
    "📊 Анализ", 
    "🔮 Прогноз", 
    "🎯 What-if", 
    "⚙️ Настройки"
])

with tab1:
    st.header("Коэффициент использования установленной мощности")
    st.info("💡 КИУМ рассчитывается автоматически: КИУМ = Выработка / (N_уст × Часы)")
    
    # Быстрые метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_kium = filtered_kium['КИУМ_общий'].mean()
        st.metric("Средний КИУМ", f"{avg_kium:.2%}")
    with col2:
        max_kium = filtered_kium['КИУМ_общий'].max()
        max_tg = filtered_kium.loc[filtered_kium['КИУМ_общий'].idxmax(), 'ТГ']
        st.metric("Максимальный КИУМ", f"{max_kium:.2%}", delta=f"{max_tg}")
    with col3:
        min_kium = filtered_kium['КИУМ_общий'].min()
        min_tg = filtered_kium.loc[filtered_kium['КИУМ_общий'].idxmin(), 'ТГ']
        st.metric("Минимальный КИУМ", f"{min_kium:.2%}", delta=f"{min_tg}")
    
    # Данные и график в колонках
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Расчетные данные КИУМ")
        display_kium = filtered_kium.copy()
        for month in selected_months:
            display_kium[month] = display_kium[month].apply(lambda x: f"{x:.2%}")
        display_kium['КИУМ_общий'] = display_kium['КИУМ_общий'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(display_kium, use_container_width=True)
        
        csv = filtered_kium.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Скачать CSV", 
            csv, 
            "kium_data.csv",
            help="Скачать данные КИУМ в формате CSV"
        )
    
    with col2:
        st.subheader("Визуализация")
        
        # Выбор типа графика
        chart_type = st.radio(
            "Тип графика:",
            ["Линейный", "Столбчатый", "Тепловая карта"],
            horizontal=True
        )
        
        if chart_type == "Линейный":
            melted = filtered_kium.melt(id_vars=['ТГ'], value_vars=selected_months, 
                                      var_name='Месяц', value_name='КИУМ')
            fig = px.line(melted, x='Месяц', y='КИУМ', color='ТГ',
                         title="Динамика КИУМ по месяцам")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Столбчатый":
            fig = px.bar(filtered_kium, x='ТГ', y='КИУМ_общий',
                        title="Общий КИУМ по ТГ")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            fig = px.imshow(filtered_kium.set_index('ТГ')[selected_months].T, 
                           color_continuous_scale='RdYlGn',
                           title="Тепловая карта КИУМ")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Выработка электроэнергии")
    st.info("💡 Это первичные данные, на основе которых рассчитывается КИУМ")
    
    # Метрики выработки
    total_per_tg = filtered_gen[selected_months].sum(axis=1)
    total_gen = total_per_tg.sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Общая выработка", f"{total_gen:,.0f} МВт*ч")
    with col2:
        avg_gen_per_tg = total_per_tg.mean()
        st.metric("Средняя на ТГ", f"{avg_gen_per_tg:,.0f} МВт*ч")
    with col3:
        max_gen_tg = filtered_gen.loc[total_per_tg.idxmax(), 'ТГ']
        st.metric("Лидер", max_gen_tg)
    
    # Данные и график
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Данные выработки")
        display_gen = filtered_gen.copy()
        for month in selected_months:
            display_gen[month] = display_gen[month].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        
        st.dataframe(display_gen, use_container_width=True)
        
        csv = filtered_gen.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Скачать CSV", 
            csv, 
            "generation_data.csv",
            help="Скачать данные выработки в формате CSV"
        )
    
    with col2:
        st.subheader("Визуализация выработки")
        
        chart_type = st.radio(
            "Тип графика:",
            ["Суммарная по ТГ", "Помесячная"],
            horizontal=True,
            key="gen_chart"
        )
        
        if chart_type == "Суммарная по ТГ":
            fig = px.bar(x=filtered_gen['ТГ'], y=total_per_tg,
                        title="Суммарная выработка по ТГ",
                        labels={'x': 'Турбогенератор', 'y': 'Выработка, МВт*ч'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            monthly_totals = filtered_gen[selected_months].sum()
            fig = px.line(x=selected_months, y=monthly_totals,
                         title="Общая выработка по месяцам",
                         labels={'x': 'Месяц', 'y': 'Выработка, МВт*ч'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Анализ эффективности")
    st.info("💡 Все показатели рассчитываются на основе согласованных данных")
    
    # Расчет показателей
    total_per_tg = filtered_gen[selected_months].sum(axis=1)
    avg_kium = filtered_kium['КИУМ_общий'].mean()
    total_gen = total_per_tg.sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Средний КИУМ", f"{avg_kium:.2%}")
    with col2:
        st.metric("Общая выработка", f"{total_gen:,.0f} МВт*ч")
    with col3:
        total_hours = hours_df[hours_df['Месяц'].isin(selected_months)]['Часы'].sum()
        capacity_utilization = total_gen / (filtered_gen['N_уст'].sum() * total_hours)
        st.metric("Использование мощностей", f"{capacity_utilization:.2%}")
    with col4:
        best_tg = filtered_kium.loc[filtered_kium['КИУМ_общий'].idxmax(), 'ТГ']
        st.metric("Лучший КИУМ", best_tg)
    
    # Детальный анализ
    st.subheader("Сводка по ТГ")
    
    analysis_df = pd.DataFrame({
        'ТГ': filtered_gen['ТГ'].values,
        'Выработка, МВт*ч': total_per_tg.values,
        'КИУМ, %': (filtered_kium['КИУМ_общий'].values * 100).round(2),
        'N уст, МВт': filtered_gen['N_уст'].values,
        'Эффективность': (filtered_kium['КИУМ_общий'].values * 100).round(2)
    })
    
    st.dataframe(analysis_df, use_container_width=True)
    
    # Топ-3 ТГ
    st.subheader("Топ-3 по выработке")
    top_3 = total_per_tg.nlargest(3)
    for i, (idx, value) in enumerate(top_3.items(), 1):
        tg_name = filtered_gen.loc[idx, 'ТГ']
        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric(f"#{i}", f"{value:,.0f} МВт*ч")
        with col2:
            progress_value = float(value / top_3.max()) if top_3.max() > 0 else 0
            st.progress(progress_value, text=f"{tg_name}")

with tab4:
    st.header("Прогнозирование выработки")
    
    if len(selected_tgs) == 0:
        st.warning("Выберите хотя бы один ТГ для прогнозирования")
    else:
        tg = st.selectbox("Выберите ТГ для прогноза", selected_tgs)
        
        if tg:
            data = filtered_gen[filtered_gen['ТГ'] == tg][selected_months].T.squeeze()
            data.index = pd.date_range(start='2025-01-01', periods=len(data), freq='M')
            
            col1, col2 = st.columns(2)
            with col1:
                forecast_steps = st.slider("Период прогноза (месяцы)", 1, 6, 3)
            with col2:
                confidence = st.slider("Доверительный интервал", 0.8, 0.99, 0.95)
            
            if st.button("Рассчитать прогноз", type="primary"):
                with st.spinner("Построение прогноза..."):
                    try:
                        model = ARIMA(data, order=(1,1,1)).fit()
                        forecast = model.get_forecast(steps=forecast_steps)
                        forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), 
                                                     periods=forecast_steps, freq='M')
                        
                        fig = go.Figure()
                        
                        # Исторические данные
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data, 
                            name='Исторические данные',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        # Прогноз
                        fig.add_trace(go.Scatter(
                            x=forecast_index, y=forecast.predicted_mean,
                            name='Прогноз',
                            line=dict(color='#ff7f0e', width=2, dash='dash')
                        ))
                        
                        # Доверительный интервал
                        ci = forecast.conf_int(alpha=1-confidence)
                        fig.add_trace(go.Scatter(
                            x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                            y=ci.iloc[:, 0].tolist() + ci.iloc[:, 1].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(255, 127, 14, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'Доверительный интервал {confidence:.0%}'
                        ))
                        
                        fig.update_layout(
                            title=f"Прогноз выработки для {tg}",
                            xaxis_title="Дата",
                            yaxis_title="Выработка, МВт*ч",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Прогноз в таблице
                        st.subheader("Численные значения прогноза")
                        forecast_df = pd.DataFrame({
                            'Месяц': forecast_index.strftime('%B %Y'),
                            'Прогноз, МВт*ч': forecast.predicted_mean.round(1),
                            'Нижняя граница': ci.iloc[:, 0].round(1),
                            'Верхняя граница': ci.iloc[:, 1].round(1)
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Ошибка при построении прогноза: {e}")

with tab5:
    st.header("Анализ сценариев")
    st.info("💡 Изменения применяются к выработке, КИУМ пересчитывается автоматически")
    
    if len(selected_tgs) == 0:
        st.warning("Выберите хотя бы один ТГ для анализа")
    else:
        tg = st.selectbox("Выберите ТГ", selected_tgs, key="whatif_tg")
        
        if tg:
            orig_row_kium = filtered_kium[filtered_kium['ТГ'] == tg].iloc[0]
            orig_row_gen = filtered_gen[filtered_gen['ТГ'] == tg].iloc[0]
            orig_n_уст = orig_row_gen['N_уст']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Параметры сценария")
                percent_change_n = st.number_input(
                    "Изменение N_уст (%)", 
                    value=0.0, 
                    min_value=-50.0, 
                    max_value=100.0,
                    help="Изменение установленной мощности в процентах"
                )
                
                st.subheader("Изменение выработки по месяцам (%)")
                change_gen = {}
                for month in selected_months[:6]:  # Показываем первые 6 месяцев для компактности
                    change_gen[month] = st.number_input(
                        f"{month} - выработка", 
                        value=0.0, 
                        min_value=-100.0, 
                        max_value=100.0,
                        key=f"gen_{month}"
                    )
            
            with col2:
                # Расчет новых значений
                new_n_уст = orig_n_уст * (1 + percent_change_n / 100)
                
                # Применяем изменения к выработке
                new_gen = pd.Series(index=selected_months)
                for month in selected_months:
                    change = change_gen.get(month, 0.0)
                    new_gen[month] = max(0, orig_row_gen[month] * (1 + change / 100))
                
                new_gen_total = new_gen.sum()
                orig_gen_total = orig_row_gen[selected_months].sum()
                
                # ПЕРЕСЧИТЫВАЕМ КИУМ на основе новой выработки
                new_kium = pd.Series(index=selected_months)
                for i, month in enumerate(selected_months):
                    hours = hours_per_month[months.index(month)]
                    if new_n_уст > 0 and hours > 0:
                        new_kium[month] = new_gen[month] / (new_n_уст * hours)
                    else:
                        new_kium[month] = 0.0
                
                new_kium_avg = new_kium.mean()
                
                # Отображение результатов
                st.subheader("Результаты сценария")
                
                results_df = pd.DataFrame({
                    'Параметр': ['N_уст, МВт', 'Средний КИУМ', 'Общая выработка, МВт*ч'],
                    'Базовый': [orig_n_уст, f"{orig_row_kium['КИУМ_общий']:.2%}", f"{orig_gen_total:,.0f}"],
                    'Сценарий': [f"{new_n_уст:.1f}", f"{new_kium_avg:.2%}", f"{new_gen_total:,.0f}"],
                    'Изменение': [
                        f"{percent_change_n:+.1f}%",
                        f"{(new_kium_avg - orig_row_kium['КИУМ_общий']) / orig_row_kium['КИУМ_общий'] * 100:+.1f}%" if orig_row_kium['КИУМ_общий'] > 0 else "N/A",
                        f"{(new_gen_total - orig_gen_total) / orig_gen_total * 100:+.1f}%" if orig_gen_total > 0 else "N/A"
                    ]
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Визуализация сравнения
                compare_df = pd.DataFrame({
                    'Месяц': selected_months,
                    'Базовый сценарий': orig_row_gen[selected_months],
                    'Новый сценарий': new_gen
                }).melt(id_vars='Месяц', var_name='Сценарий', value_name='Выработка')
                
                fig = px.bar(compare_df, x='Месяц', y='Выработка', color='Сценарий', 
                            barmode='group', title="Сравнение выработки по месяцам")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("Настройки и данные")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Добавить новые данные")
        st.info("💡 Вводите данные выработки - КИУМ рассчитается автоматически")
        
        data_type = st.selectbox("Тип данных", ['Выработка'])
        tg_name = st.text_input("Наименование ТГ")
        
        if tg_name:
            st.write("Введите данные ВЫРАБОТКИ по месяцам (МВт*ч):")
            col1, col2, col3 = st.columns(3)
            new_data = {}
            
            with col1:
                for month in months[:4]:
                    new_data[month] = st.number_input(month, value=0.0, key=f"new_{month}")
            with col2:
                for month in months[4:8]:
                    new_data[month] = st.number_input(month, value=0.0, key=f"new_{month}")
            with col3:
                for month in months[8:]:
                    new_data[month] = st.number_input(month, value=0.0, key=f"new_{month}")
            
            n_ust = st.number_input("N уст (МВт)", value=0.0)
            
            if st.button("Добавить данные", type="primary"):
                # Добавляем данные выработки
                new_gen_row = pd.DataFrame([{'ТГ': tg_name, **new_data, 'N_уст': n_ust}])
                st.session_state.gen_df = pd.concat([st.session_state.gen_df, new_gen_row], ignore_index=True)
                
                # ПЕРЕСЧИТЫВАЕМ КИУМ для всех данных
                st.session_state.kium_df = calculate_kium_from_generation(
                    st.session_state.gen_df, 
                    st.session_state.hours_df
                )
                
                st.success(f"Данные выработки для {tg_name} добавлены, КИУМ пересчитан")
                st.rerun()
    
    with col2:
        st.subheader("Текущие данные")
        
        st.write("**Турбогенераторы в системе:**")
        for tg in kium_df['ТГ'].unique():
            st.write(f"• {tg}")
        
        # Кнопка для сброса к исходным данным
        if st.button("Сбросить к исходным данным"):
            kium_df, gen_df, hours_df = get_parsed_data()
            st.session_state.kium_df = kium_df
            st.session_state.gen_df = gen_df
            st.session_state.hours_df = hours_df
            st.success("Данные сброшены к исходным")
            st.rerun()
        
        st.divider()
        st.subheader("О системе")
        st.write("""
        **АТЭЦ Аналитика** - система мониторинга и анализа работы турбогенераторов.
        
        **Принцип работы:**
        • Ввод данных выработки и установленной мощности
        • Автоматический расчет КИУМ по формуле:
          КИУМ = Выработка / (N_уст × Часы)
        • Согласованный анализ всех показателей
        
        Для связи: example@company.com
        """)

# Футер
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "АТЭЦ Аналитика • v2.0 • 2024 • Данные КИУМ рассчитываются автоматически"
    "</div>",
    unsafe_allow_html=True
)
