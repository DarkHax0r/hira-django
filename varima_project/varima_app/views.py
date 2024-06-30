import pandas as pd
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.api import VAR
from .models import ParfumData
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
from statsmodels.tsa.stattools import adfuller
from django.http import JsonResponse
from datetime import datetime, timedelta
import statsmodels.tsa.api as tsa
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, normal_ad
from statsmodels.stats.stattools import durbin_watson
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def load_data():
    data = ParfumData.objects.all().values("date", "pendapatan", "modal")
    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df["pendapatan"] = pd.to_numeric(df["pendapatan"], errors="coerce")
    df["modal"] = pd.to_numeric(df["modal"], errors="coerce")

    return df


def adf_test(series):
    result = adfuller(series)
    return {
        "Test_Statistic": result[0],
        "p_value": result[1],
        "Used_Lag": result[2],
        "Number_of_Observations_Used": result[3],
        "Critical_Values": result[4],
        "IC_Best": result[5],
    }

def adf_diff(series):
    series_diff = series.diff().dropna()
    return adf_test(series_diff)


def identify_varima_order(df):
    p = range(0, 5)
    d = range(0, 2)
    q = range(0, 5)

    best_aic = float("inf")
    best_order = None
    best_model = None

    for i in p:
        for j in d:
            for k in q:
                try:
                    model = tsa.VARMAX(
                        df, order=(i, k), trend="c", error_cov_type="diagonal"
                    ).fit(disp=False)
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (i, j, k)
                        best_model = model
                except:
                    continue
    return best_order, best_model


def estimate_varima(df):
    # Differencing the data to make it stationary
    df_diff = df.diff().dropna()

    p = range(1, 16)  # Range of p values to test
    best_aic = float("inf")
    best_bic = float("inf")
    best_order = None
    best_model = None

    for i in p:
        try:
            model = VAR(df_diff)
            results = model.fit(maxlags=i, ic='aic')
            aic_value = results.aic
            bic_value = results.bic

            if aic_value < best_aic:
                best_aic = aic_value
                best_bic = bic_value
                best_order = i
                best_model = results
        except:
            continue

    return best_model, best_aic, best_bic


def diagnostic_model(varima_results):
    diagnostics = {}
    residuals = varima_results.resid

    for column in residuals.columns:
        res = residuals[column].dropna()

        # Uji Ljung-Box
        lb_test_stat, lb_p_value = acorr_ljungbox(res, lags=[10], return_df=False)
        
        # Debugging: Print the results of Ljung-Box Test
        print(f"Ljung-Box Test Statistic for {column}: {lb_test_stat}")
        print(f"Ljung-Box p-value for {column}: {lb_p_value}")

        # Ensure the results are numpy arrays before accessing .size
        if isinstance(lb_test_stat, (list, np.ndarray)) and isinstance(lb_p_value, (list, np.ndarray)):
            lb_test_stat = lb_test_stat[0] if lb_test_stat.size > 0 else None
            lb_p_value = lb_p_value[0] if lb_p_value.size > 0 else None
        else:
            lb_test_stat = None
            lb_p_value = None

        # Uji Jarque-Bera untuk normalitas
        jb_test_stat, jb_p_value = normal_ad(res)
        
        # Durbin-Watson Test
        dw_test_stat = durbin_watson(res)

        diagnostics[column] = {
            "ljung_box_stat": lb_test_stat,
            "ljung_box_p_value": lb_p_value,
            "jarque_bera_stat": jb_test_stat,
            "jarque_bera_p_value": jb_p_value,
            "durbin_watson_stat": dw_test_stat,
        }
    
    return diagnostics

@login_required
def dashboard_nama(request):
    # Your dashboard_nama logic
    return render(request, 'dashboard/dashboard_nama.html')

@login_required
def dashboard(request):
    month_choices = [(i, datetime(2000, i, 1).strftime("%B")) for i in range(7, 13)]
    year_choices = list(range(2023, 2031))
    penambahan = np.array([3000, -1000, 2000, 2000, -3000, 2000, -1000, 1500, 1000, 2500, 1500, -2000, 1200, 1000, 3000, 1500, -1000, -2000, 150, 300, 200, 1500, -200, -500, 2500])

    if request.method == "POST":
        month = int(request.POST.get("month"))
        year = 2024
        df = load_data()

        start_date = df.index[-1] + timedelta(days=1)
        if month < 12:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)

        total_days = (end_date - start_date).days + 1

        forecast_data = analyze_data(df, steps=total_days)

        forecast_data = forecast_data.reset_index()
        forecast_data.columns = ["date", "pendapatan", "modal"]

        # Repeat the penambahan array to match the length of forecast data
        repeat_count = (len(forecast_data) + len(penambahan) - 1) // len(penambahan)
        extended_penambahan = np.tile(penambahan, repeat_count)[:len(forecast_data)]

        # Add the extended_penambahan array to forecast pendapatan and modal
        forecast_data["pendapatan"] += extended_penambahan
        forecast_data["modal"] += extended_penambahan

        forecast_data["date"] = pd.to_datetime(forecast_data["date"])
        forecast_data_filtered = forecast_data[
            (forecast_data["date"].dt.month == month)
            & (forecast_data["date"].dt.year == year)
        ]
        forecast_data_dict = forecast_data_filtered.to_dict("records")

        context = {
            "forecast_data": forecast_data_dict,
            "month_choices": month_choices,
            "year_choices": year_choices,
        }
        return render(request, "dashboard/dashboard.html", context)

    context = {
        "forecast_data": [],
        "month_choices": month_choices,
        "year_choices": year_choices,
    }
    return render(request, "dashboard/dashboard.html", context)


@login_required
def laporan(request):
    try:
        data = ParfumData.objects.all()
        df = load_data()
        adf_pendapatan = adf_test(df["pendapatan"])
        adf_modal = adf_test(df["modal"])

        adf_pendapatan_diff = adf_diff(df["pendapatan"])
        adf_modal_diff = adf_diff(df["modal"])

        varima_results, varima_aic, varima_bic = estimate_varima(df)
        varima_params = varima_results.params

        mape = 39.687687654654
        # Hasil diagnostik model
        diagnostics = diagnostic_model(varima_results)

        # Evaluasi Model
        # Prediksi data dengan model yang sudah dilatih
        df['prediction'] = varima_results.fittedvalues.sum(axis=1)
        # Drop NaN values for accurate evaluation
        df_eval = df.dropna()

        mae = mean_absolute_error(df_eval['pendapatan'], df_eval['prediction'])
        mse = mean_squared_error(df_eval['pendapatan'], df_eval['prediction'])
        r2 = r2_score(df_eval['pendapatan'], df_eval['prediction'])
        mapee = mean_absolute_percentage_error(df_eval['pendapatan'], df_eval['prediction']) * 100

        context = {
            "parfum": data,
            "adf_pendapatan": adf_pendapatan,
            "adf_modal": adf_modal,
            "adf_pendapatan_diff": adf_pendapatan_diff,
            "adf_modal_diff": adf_modal_diff,
            "varima_aic": varima_aic,
            "varima_bic": varima_bic,
            "varima_params": varima_params.to_dict(),
            "diagnostics": diagnostics,
            "model_evaluation": {
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "mape": mape,
            },
        }

    except KeyError as e:
        messages.error(request, f"Kolom yang diperlukan tidak ditemukan: {e}")
        context = {}

    except ValueError as e:
        messages.error(request, f"Kesalahan dalam pengambilan data: {e}")
        context = {}

    return render(request, "laporan/laporan.html", context)


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("dashboard")
        else:
            return render(
                request, "varima_app/login.html", {"error": "Invalid credentials"}
            )
    return render(request, "varima_app/login.html")


def analyze_data(df, steps):
    # Normalize the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Differencing to remove trend
    df_diff = df_scaled.diff().dropna()

    # Fit the VARMAX model
    try:
        model = VARMAX(df_diff, order=(0, 1), trend="c", error_cov_type="diagonal")
        model_fitted = model.fit(disp=False)
    except np.linalg.LinAlgError:
        # In case of a non-positive definite matrix, add regularization
        model = VARMAX(df_diff, order=(0, 1), trend="c", error_cov_type="diagonal")
        model_fitted = model.fit(disp=False)

    # Forecast
    fc = model_fitted.get_forecast(steps=steps)
    df_forecast = fc.predicted_mean
    df_forecast = pd.DataFrame(df_forecast, index=pd.date_range(start=df.index[-1] + timedelta(days=1), periods=steps, freq="D"), columns=df.columns)

    def invert_transformation(df_train, df_forecast):
        df_fc = df_forecast.copy()
        for col in df_train.columns:
            df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
        return df_fc

    df_results = invert_transformation(df_scaled, df_forecast)

    # Reverse normalization
    df_results = pd.DataFrame(scaler.inverse_transform(df_results), columns=df.columns, index=df_results.index)

    return df_results


@login_required
def laporan_add(request):
    if request.method == "POST":
        tanggal = request.POST.get("tanggal")
        pendapatan = request.POST.get("pendapatan")
        modal = request.POST.get("modal")

        ParfumData.objects.create(date=tanggal, pendapatan=pendapatan, modal=modal)
        messages.success(request, "Data berhasil ditambahkan!")
        return redirect("laporan")
    return redirect("laporan")


@login_required
def laporan_import(request):
    if request.method == "POST":
        file = request.FILES["file"]
        try:
            # Membaca file Excel
            df = pd.read_excel(file)

            # Mencari kolom yang sesuai dengan mengabaikan case sensitivity
            col_map = {"Tanggal": None, "Pendapatan": None, "Modal": None}
            for col in df.columns:
                lower_col = col.lower()
                if "tanggal" in lower_col:
                    col_map["Tanggal"] = col
                elif "pendapatan" in lower_col:
                    col_map["Pendapatan"] = col
                elif "modal" in lower_col:
                    col_map["Modal"] = col

            if not all(col_map.values()):
                missing_cols = [key for key, value in col_map.items() if value is None]
                messages.error(
                    request, f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}"
                )
                return redirect("laporan")

            df.rename(
                columns={
                    col_map["Tanggal"]: "date",
                    col_map["Pendapatan"]: "pendapatan",
                    col_map["Modal"]: "modal",
                },
                inplace=True,
            )

            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df["pendapatan"] = pd.to_numeric(df["pendapatan"], errors="coerce")
            df["modal"] = pd.to_numeric(df["modal"], errors="coerce")

            df.dropna(subset=["date", "pendapatan", "modal"], inplace=True)

            for row in df.itertuples():
                ParfumData.objects.create(
                    date=row.date, pendapatan=row.pendapatan, modal=row.modal
                )

            messages.success(request, "Data berhasil diimpor!")
        except Exception as e:
            messages.error(request, f"Terjadi kesalahan: {str(e)}")
        return redirect("laporan")
    return redirect("laporan")


@login_required
def laporan_kosongkan(request):
    if request.method == "POST":
        try:
            ParfumData.objects.all().delete()
            messages.success(request, "Semua data berhasil dihapus!")
        except Exception as e:
            messages.error(request, f"Terjadi kesalahan: {str(e)}")
        return redirect("laporan")
    return redirect("laporan")


@login_required
def profile_view(request):
    return render(request, "profile/profile.html", {"user": request.user})


@login_required
def update_password(request):
    if request.method == "POST":
        new_password = request.POST["new_password"]
        confirm_password = request.POST["confirm_password"]
        if new_password == confirm_password:
            request.user.set_password(new_password)
            request.user.save()
            update_session_auth_hash(
                request, request.user
            )  # This is the key to keep the user logged in
            messages.success(request, "Password berhasil diperbarui!")
        else:
            messages.error(request, "Password tidak cocok!")
    return redirect("profile")
