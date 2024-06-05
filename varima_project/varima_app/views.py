import pandas as pd
from statsmodels.tsa.api import VAR
from django.shortcuts import render
from .models import ParfumData

def load_data():
    data = ParfumData.objects.all().values('date', 'pendapatan', 'modal')
    df = pd.DataFrame(data)
    
    # Pastikan kolom 'date' menjadi index dan pastikan formatnya benar
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Konversi kolom ke tipe numerik
    df['pendapatan'] = pd.to_numeric(df['pendapatan'], errors='coerce')
    df['modal'] = pd.to_numeric(df['modal'], errors='coerce')
    
    return df

def analyze_data(request):
    df = load_data()
    df_diff = df.diff().dropna()
    model = VAR(df_diff)
    model_fitted = model.fit(2)  # asumsikan lag optimal adalah 2
    forecast_input = df_diff.values[-2:]
    fc = model_fitted.forecast(y=forecast_input, steps=10)
    df_forecast = pd.DataFrame(fc, index=pd.date_range(start=df_diff.index[-1], periods=10, freq='D'), columns=df.columns)

    def invert_transformation(df_train, df_forecast, diff_df):
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
        return df_fc

    df_results = invert_transformation(df, df_forecast, df_diff)
    context = {'forecast': df_results.to_html()}
    return render(request, 'varima_app/results.html', context)
