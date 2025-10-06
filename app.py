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

# Текстовые данные из документа (для дефолта)
text_data = """
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

# Функция для парсинга данных из текстового представления
@st.cache_data
def parse_data_from_text(text_data):
    lines = text_data.strip().split('\n')
    
    # Парсим строки с данными ТГ для КИУМ %: rows 3-9
    tg_data = {}
    for line in lines:
        if ':' in line:
            row_part = line.split(':')[0]
            if row_part.startswith('row') and row_part[3:].isdigit():
                row_num = int(row_part[3:])
                if 3 <= row_num <= 9:
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
                        tg_data[tg] = {months[i]: values[i] for i in range(12)}
                        if len(parts) > 13:
                            tg_data[tg]['КИУМ_общий'] = float(parts[13])
                        else:
                            tg_data[tg]['КИУМ_общий'] = np.mean(values)
    
    # Создаем DataFrame для КИУМ
    kium_data = []
    for tg, data in tg_data.items():
        row = {'ТГ': tg}
        for month in months:
            row[month] = data.get(month, 0)
        row['КИУМ_общий'] = data.get('КИУМ_общий', np.mean([data.get(m,0) for m in months]))
        kium_data.append(row)
    kium_df = pd.DataFrame(kium_data)
    
    # Парсим выработку МВт*ч: rows 12-18
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
    
    # DataFrame для выработки
    gen_data = []
    for tg, data in generation_data.items():
        row = {'ТГ': tg}
        for month in months:
            row[month] = data.get(month, 0)
        row['N_уст'] = data.get('N_уст', 0)
        gen_data.append(row)
    gen_df = pd.DataFrame(gen_data)
    
    # Часы в месяцах
    hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    hours_df = pd.DataFrame({'Месяц': months, 'Часы': hours_per_month})
    
    return kium_df, gen_df, hours_df

# Добавляем CSS для мобильной адаптации
st.markdown("""
    <style>
        /* Основной контейнер */
        [data-testid="stAppViewContainer"] {
            max-width: 100%;
            padding: 1rem;
        }
        /* Таблицы */
        .dataframe {
            width: 100% !important;
            overflow-x: auto;
        }
        /* Графики */
        .plotly-chart {
            width: 100% !important;
            height: auto !important;
        }
        /* Блоки */
        .block-container {
            padding: 0.5rem;
        }
        /* Мобильные корректировки */
        @media (max-width: 768px) {
            .stButton > button {
                width: 100%;
            }
            .stTextInput > div > div > input {
                width: 100%;
            }
            .stNumberInput > div > div > input {
                width: 100%;
            }
            .stSelectbox > div > div > select {
                width: 100%;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit интерфейс
st.title("КИУМ АТЭЦ 2025")

# Загрузка данных: дефолт + uploader
uploaded_file = st.sidebar.file_uploader("Excel", type=['xlsx'])
if uploaded_file:
    try:
        kium_df = pd.read_excel(uploaded_file, sheet_name='2025', skiprows=1, nrows=7, usecols='A:N')
        kium_df.columns = ['ТГ'] + months + ['КИУМ_общий']
        gen_df = pd.read_excel(uploaded_file, sheet_name='2025', skiprows=10, nrows=7, usecols='A:N')
        gen_df.columns = ['ТГ'] + months + ['N_уст']
        hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        hours_df = pd.DataFrame({'Месяц': months, 'Часы': hours_per_month})
    except Exception as e:
        st.error(f"Ошибка: {e}")
else:
    kium_df, gen_df, hours_df = parse_data_from_text(text_data)

# Сохраняем в session_state
st.session_state.kium_df = kium_df
st.session_state.gen_df = gen_df
st.session_state.hours_df = hours_df

kium_df = st.session_state.kium_df
gen_df = st.session_state.gen_df
hours_df = st.session_state.hours_df

# Фильтры
selected_tgs = st.sidebar.multiselect("ТГ", kium_df['ТГ'].unique(), default=kium_df['ТГ'].unique())
selected_months = st.sidebar.multiselect("Месяцы", months, default=months)

filtered_kium = kium_df[kium_df['ТГ'].isin(selected_tgs)][['ТГ'] + selected_months + ['КИУМ_общий']]
filtered_gen = gen_df[gen_df['ТГ'].isin(selected_tgs)][['ТГ'] + selected_months + ['N_уст']]

# Навигация
page = st.sidebar.selectbox("Раздел", ["КИУМ", "Выработка", "Анализ", "Графики", "Прогноз", "Добавить", "Нагрузка", "What-if", "Погода", "Цены энергии"])

if page == "КИУМ":
    st.dataframe(filtered_kium)
    csv = filtered_kium.to_csv(index=False).encode('utf-8')
    st.download_button("CSV", csv, "kium.csv")

elif page == "Выработка":
    st.dataframe(filtered_gen)
    csv = filtered_gen.to_csv(index=False).encode('utf-8')
    st.download_button("CSV", csv, "gen.csv")

elif page == "Анализ":
    total_per_tg = filtered_gen[selected_months].sum(axis=1)
    avg_kium = filtered_kium['КИУМ_общий'].mean()
    total_gen = total_per_tg.sum()
    st.metric("Средний КИУМ", f"{avg_kium:.4f}%")
    st.metric("Выработка", f"{total_gen:.2f} МВт*ч")
    
    top_indices = total_per_tg.nlargest(3).index
    for idx in top_indices:
        tg = filtered_gen.loc[idx, 'ТГ']
        val = total_per_tg.loc[idx]
        st.write(f"{tg}: {val:.2f} МВт*ч")
    
    total_table = pd.DataFrame({
        'ТГ': filtered_gen['ТГ'].values,
        'Выработка': total_per_tg.values,
        'КИУМ': filtered_kium['КИУМ_общий'].values,
        'N уст': filtered_gen['N_уст'].values
    })
    st.dataframe(total_table)
    csv = total_table.to_csv(index=False).encode('utf-8')
    st.download_button("CSV", csv, "analysis.csv")

elif page == "Графики":
    plot_type = st.selectbox("Тип", ['КИУМ/мес', 'Heatmap КИУМ', 'Выработка/ТГ', 'КИУМ/ТГ'])
    
    if plot_type == 'КИУМ/мес':
        melted = filtered_kium.melt(id_vars=['ТГ'], value_vars=selected_months, var_name='Мес', value_name='КИУМ')
        fig = px.line(melted, x='Мес', y='КИУМ', color='ТГ')
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == 'Heatmap КИУМ':
        fig = px.imshow(filtered_kium.set_index('ТГ')[selected_months].T, color_continuous_scale='RdYlGn')
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == 'Выработка/ТГ':
        total_per_tg = filtered_gen[selected_months].sum(axis=1)
        fig = px.bar(x=filtered_gen['ТГ'], y=total_per_tg)
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == 'КИУМ/ТГ':
        fig = px.bar(x=filtered_kium['ТГ'], y=filtered_kium['КИУМ_общий'])
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Прогноз":
    tg = st.selectbox("ТГ", filtered_gen['ТГ'].unique())
    data = filtered_gen[filtered_gen['ТГ'] == tg][selected_months].T.squeeze()
    data.index = pd.date_range(start='2025-01-01', periods=len(data), freq='M')
    try:
        model = ARIMA(data, order=(1,1,1)).fit()
        forecast_steps = st.slider("Мес", 1, 12, 3)
        forecast = model.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data, name='Ист'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Прог', line=dict(dash='dash')))
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Ошибка: {e}")

elif page == "Добавить":
    data_type = st.selectbox("Тип", ['КИУМ', 'Выработка'])
    tg_name = st.text_input("ТГ")
    if tg_name:
        new_data = {}
        for month in months:
            new_data[month] = st.number_input(month, value=0.0)
        n_ust = 0.0
        if data_type == 'Выработка':
            n_ust = st.number_input("N уст", value=0.0)
        if st.button("Добавить"):
            if data_type == 'КИУМ':
                new_row = pd.DataFrame([{'ТГ': tg_name, **new_data, 'КИУМ_общий': np.mean(list(new_data.values()))}])
                st.session_state.kium_df = pd.concat([st.session_state.kium_df, new_row], ignore_index=True)
            else:
                new_row = pd.DataFrame([{'ТГ': tg_name, **new_data, 'N_уст': n_ust}])
                st.session_state.gen_df = pd.concat([st.session_state.gen_df, new_row], ignore_index=True)
            st.rerun()

elif page == "Нагрузка":
    hours_monthly = pd.Series(hours_df.set_index('Месяц')['Часы'][selected_months].values, index=selected_months)
    gen_monthly = filtered_gen.set_index('ТГ')[selected_months]
    load_from_gen = gen_monthly.div(hours_monthly, axis=1)
    load_from_gen['Средняя'] = load_from_gen.mean(axis=1)
    load_from_gen['Макс'] = load_from_gen.max(axis=1)
    load_from_gen = load_from_gen.reset_index()
    
    kium_monthly = filtered_kium.set_index('ТГ')[selected_months]
    n_ust = filtered_gen.set_index('ТГ')['N_уст']
    load_from_kium = kium_monthly.mul(n_ust, axis=0)
    load_from_kium['Средняя'] = load_from_kium.mean(axis=1)
    load_from_kium['Макс'] = load_from_kium.max(axis=1)
    load_from_kium = load_from_kium.reset_index()
    
    st.dataframe(load_from_gen.round(2))
    st.dataframe(load_from_kium.round(2))
    
    avg_load = load_from_gen['Средняя'].mean()
    st.metric("Средняя", f"{avg_load:.2f} МВт")
    
    fig = px.bar(load_from_gen, x='ТГ', y='Средняя')
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

elif page == "What-if":
    tg = st.selectbox("ТГ", filtered_gen['ТГ'].unique())
    if tg:
        orig_row_kium = filtered_kium[filtered_kium['ТГ'] == tg].iloc[0]
        orig_row_gen = filtered_gen[filtered_gen['ТГ'] == tg].iloc[0]
        orig_n_уст = orig_row_gen['N_уст']
        
        percent_change_n = st.number_input("Изменение N_уст %", value=0.0)
        new_n_уст = orig_n_уст * (1 + percent_change_n / 100)
        
        change_kium = {}
        for month in selected_months:
            change_kium[month] = st.number_input(month, value=0.0)
        
        new_kium = orig_row_kium.copy()
        for month in selected_months:
            new_kium[month] = orig_row_kium[month] * (1 + change_kium[month] / 100)
        new_kium['КИУМ_общий'] = new_kium[selected_months].mean()
        
        hours_monthly = hours_df.set_index('Месяц')['Часы']
        new_gen = pd.Series(index=selected_months)
        for month in selected_months:
            new_gen[month] = new_kium[month] * new_n_уст * hours_monthly[month]
        new_gen_total = new_gen.sum()
        
        orig_gen = orig_row_gen[selected_months]
        orig_gen_total = orig_gen.sum()
        
        st.dataframe(pd.DataFrame({
            'Параметр': ['N_уст', 'КИУМ', 'Выработка'],
            'Ориг': [orig_n_уст, orig_row_kium['КИУМ_общий'], orig_gen_total],
            'Новое': [new_n_уст, new_kium['КИУМ_общий'], new_gen_total]
        }))
        
        compare_df = pd.DataFrame({
            'Мес': selected_months,
            'Ориг': orig_gen,
            'Новое': new_gen
        }).melt(id_vars='Мес', var_name='Тип', value_name='Выработка')
        fig = px.bar(compare_df, x='Мес', y='Выработка', color='Тип', barmode='group')
        fig.update_layout(autosize=True)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Погода":
    api_key = st.text_input("API OpenWeather", type="password")
    city = st.text_input("Город", value="Moscow")
    weather_type = st.selectbox("Тип", ["Текущая", "Прогноз"])
    
    if api_key and city and st.button("Получить"):
        try:
            if weather_type == "Текущая":
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=ru"
                data = requests.get(url).json()
                if data['cod'] == 200:
                    st.write(f"{data['weather'][0]['description']}, {data['main']['temp']}°C")
                    st.write(f"Влажность: {data['main']['humidity']}%, Ветер: {data['wind']['speed']} м/с")
                else:
                    st.error(data['message'])
            else:
                url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric&lang=ru"
                data = requests.get(url).json()
                if data['cod'] == '200':
                    forecast_df = pd.DataFrame([{
                        'Дата': item['dt_txt'],
                        'T': item['main']['temp']
                    } for item in data['list']])
                    st.dataframe(forecast_df)
                    fig = px.line(forecast_df, x='Дата', y='T')
                    fig.update_layout(autosize=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(data['message'])
        except Exception as e:
            st.error(e)

elif page == "Цены энергии":
    api_key = st.text_input("API EIA", type="password")
    series_id = st.text_input("ID серии", value="ELEC.PRICE.US-ALL.M")
    start_year = st.number_input("Нач год", value=2025)
    end_year = st.number_input("Кон год", value=2025)
    
    if api_key and series_id and st.button("Получить"):
        try:
            url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={api_key}&start={start_year}-01&end={end_year}-12&data[]=value"
            data = requests.get(url).json()
            if 'response' in data and 'data' in data['response']:
                price_df = pd.DataFrame(data['response']['data'])[['period', 'value']]
                price_df.columns = ['Мес', 'Цена']
                st.dataframe(price_df)
                
                fig = px.line(price_df, x='Мес', y='Цена')
                fig.update_layout(autosize=True)
                st.plotly_chart(fig, use_container_width=True)
                
                total_gen_monthly = filtered_gen[selected_months].sum()
                if len(total_gen_monthly) == len(price_df):
                    corr_df = pd.DataFrame({
                        'Мес': selected_months,
                        'Выработка': total_gen_monthly.values,
                        'Цена': price_df['Цена'].values
                    })
                    corr = corr_df['Выработка'].corr(corr_df['Цена'])
                    st.metric("Корреляция", f"{corr:.2f}")
                    fig_corr = px.scatter(corr_df, x='Выработка', y='Цена', trendline='ols')
                    fig_corr.update_layout(autosize=True)
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.error("Ошибка EIA")
        except Exception as e:
            st.error(e)