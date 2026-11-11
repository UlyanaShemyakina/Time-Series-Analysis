
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.title('Прогнозирование временных рядов')
st.write('Загрузите файл с временным рядом и получите прогноз.')

# Загрузка файла
uploaded_file = st.file_uploader(
    'Загрузите CSV или Excel файл',
    type=['csv', 'xlsx', 'xls']
)

if uploaded_file is not None:

    # Чтение файла
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader('Исходные данные')
    st.dataframe(df.head())

    # Выбор столбцов
    date_col = st.selectbox(
        'Выберите столбец с датой',
        df.columns
    )

    target_col = st.selectbox(
        'Выберите целевой столбец',
        df.columns
    )

    # Частота временного ряда
    freq = st.selectbox(
        'Выберите частоту данных',
        ['D', 'W', 'MS']
    )

    periods = st.number_input(
        'Количество периодов для прогноза',
        min_value=1,
        max_value=365,
        value=52
    )

    if st.button('Построить прогноз'):

        # Подготовка данных
        df[date_col] = pd.to_datetime(
            df[date_col],
            errors='coerce'
        )

        df[target_col] = pd.to_numeric(
            df[target_col],
            errors='coerce'
        )

        data = df[[date_col, target_col]].dropna()

        data = data.set_index(date_col).sort_index()

        data = data[target_col].resample(freq).sum()

        data = data.dropna()

        data_prophet = data.rename_axis('ds').reset_index(name='y')

        if len(data_prophet) < 20:
            st.error('Недостаточно данных для построения модели.')
            st.stop()

        # Разделение на обучающую и тестовую выборки
        split_index = int(len(data_prophet) * 0.8)

        train = data_prophet.iloc[:split_index]
        test = data_prophet.iloc[split_index:]

        # Обучение модели
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=1.0,
            seasonality_mode='additive'
        )

        model.fit(train)

        # Прогноз на тестовую выборку
        future_test = model.make_future_dataframe(
            periods=len(test),
            freq=freq,
            include_history=False
        )

        forecast_test = model.predict(future_test)

        y_pred = forecast_test['yhat'].values
        y_test = test['y'].values

        # Метрики
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.subheader('Метрики качества')

        col1, col2 = st.columns(2)

        col1.metric('MAE', f'{mae:.2f}')
        col2.metric('RMSE', f'{rmse:.2f}')

        # Интерпретация
        mean_sales = np.mean(np.abs(y_test))

        if mean_sales != 0:
            mae_percent = mae / mean_sales * 100
        else:
            mae_percent = np.nan

        st.subheader('Интерпретация результатов')

        st.write(
            f'Средняя абсолютная ошибка прогноза составляет '
            f'{mae:.2f}.'
        )

        st.write(
            f'RMSE составляет {rmse:.2f}. '
            f'Эта метрика сильнее реагирует на большие ошибки.'
        )

        if np.isfinite(mae_percent):
            st.write(
                f'MAE составляет примерно {mae_percent:.1f}% '
                f'от среднего абсолютного значения фактических данных.'
            )

        if mae < rmse:
            st.write(
                'RMSE больше MAE: вероятно, отдельные большие ошибки '
                'заметно влияют на качество прогноза.'
            )

        # График на тестовой выборке
        st.subheader('Фактические значения и прогноз')

        comparison = pd.DataFrame({
            'Actual': y_test,
            'Prophet': y_pred
        }, index=test['ds'])

        st.line_chart(comparison)

        # Обучение на всех данных
        final_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=1.0,
            seasonality_mode='additive'
        )

        final_model.fit(data_prophet)

        # Прогноз будущих значений
        future = final_model.make_future_dataframe(
            periods=int(periods),
            freq=freq
        )

        forecast = final_model.predict(future)

        st.subheader('Прогноз будущих значений')

        future_forecast = forecast.tail(int(periods))

        st.dataframe(
            future_forecast[
                ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            ]
        )

        # График прогноза
        fig = final_model.plot(forecast)

        plt.title('Prophet Forecast')
        plt.xlabel('Date')
        plt.ylabel(target_col)

        st.pyplot(fig)
        plt.close(fig)

        st.success('Прогноз успешно построен!')
