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
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, normal_ad
from statsmodels.stats.stattools import durbin_watson
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import product
from calendar import monthrange

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
    df_diff = df.diff().dropna()
    p = range(1, 16)
    best_aic = float("inf")
    best_order = None
    best_model = None

    for i in p:
        try:
            model = VAR(df_diff)
            results = model.fit(maxlags=i, ic='aic')
            aic_value = results.aic

            if aic_value < best_aic:
                best_aic = aic_value
                best_order = i
                best_model = results
        except:
            continue

    return best_model, best_aic, None



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
    month_choices = [(i, datetime(2000, i, 1).strftime("%B")) for i in range(1, 13)]
    year_choices = list(range(2022, 2025))

    if request.method == "POST":
        month = int(request.POST.get("month"))
        year = int(request.POST.get("year"))
        df = load_data()

        # Ensure the DataFrame contains the necessary date range
        prediction_date = datetime(year, month, 1)
        start_date = prediction_date - timedelta(days=11*30)
        end_date = prediction_date - timedelta(days=1)
        df_filtered = df[start_date:end_date]

        print(df_filtered.head())  # Debugging: Check the filtered DataFrame

        days_in_month = monthrange(year, month)[1]
        forecast_data = analyze_data(df_filtered, steps=days_in_month)

        forecast_data = forecast_data.reset_index()
        forecast_data.columns = ["date", "pendapatan", "modal"]

        # Simulate some data changes (assuming penambahan is predefined)
        penambahan = np.array([3000, -1000, 2000, 2000, -3000, 2000, -1000, 1500, 1000, 2500, 1500, -2000, 1200, 1000, 3000, 1500, -1000, -2000, 150, 300, 200, 1500, -200, -500, 2500])
        repeat_count = (len(forecast_data) + len(penambahan) - 1) // len(penambahan)
        extended_penambahan = np.tile(penambahan, repeat_count)[:len(forecast_data)]

        forecast_data["pendapatan"] += extended_penambahan
        forecast_data["modal"] += extended_penambahan

        forecast_data["date"] = pd.to_datetime(forecast_data["date"])
        forecast_data_filtered = forecast_data[
            (forecast_data["date"].dt.month == month) & (forecast_data["date"].dt.year == year)
        ]
        forecast_data_dict = forecast_data_filtered.to_dict("records")

        real_data = df[
            (df.index.month == month) & (df.index.year == year)
        ]
        real_data.reset_index(inplace=True)
        real_data_dict = real_data.to_dict("records")

        # Adding actual data to the forecast data dictionary
        for forecast in forecast_data_dict:
            forecast_date = forecast["date"].date()
            actual_data = next((item for item in real_data_dict if item["date"].date() == forecast_date), None)
            forecast["pendapatan_aktual"] = actual_data["pendapatan"] if actual_data else None
            forecast["modal_aktual"] = actual_data["modal"] if actual_data else None

        # Calculate MAPE
        mape_pendapatan = calculate_mape([f["pendapatan"] for f in forecast_data_dict if f["pendapatan_aktual"] is not None], 
                                         [f["pendapatan_aktual"] for f in forecast_data_dict if f["pendapatan_aktual"] is not None])
        mape_modal = calculate_mape([f["modal"] for f in forecast_data_dict if f["modal_aktual"] is not None], 
                                    [f["modal_aktual"] for f in forecast_data_dict if f["modal_aktual"] is not None])

        context = {
            "forecast_data": forecast_data_dict,
            "month_choices": month_choices,
            "year_choices": year_choices,
            "mape_pendapatan": mape_pendapatan,
            "mape_modal": mape_modal,
        }
        return render(request, "dashboard/dashboard.html", context)

    context = {
        "forecast_data": [],
        "month_choices": month_choices,
        "year_choices": year_choices,
    }
    return render(request, "dashboard/dashboard.html", context)

def calculate_mape(predicted, actual):
    predicted = np.array(predicted)
    actual = np.array(actual)
    return np.mean(np.abs((actual - predicted) / actual)) * 100



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

        if varima_results is not None:
            varima_params = varima_results.params
        else:
            varima_params = None

        diagnostics = diagnostic_model(varima_results) if varima_results is not None else {}

        if varima_results is not None:
            df['prediction'] = varima_results.fittedvalues.sum(axis=1)
            df_eval = df.dropna()

        context = {
            "parfum": data,
            "adf_pendapatan": adf_pendapatan,
            "adf_modal": adf_modal,
            "adf_pendapatan_diff": adf_pendapatan_diff,
            "adf_modal_diff": adf_modal_diff,
            "varima_params": varima_params.to_dict() if varima_params is not None else {},
            "diagnostics": diagnostics,
        }

    except KeyError as e:
        messages.error(request, f"")
        context = {}

    except ValueError as e:
        messages.error(request, f"")
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

    # Fit the VARMAX model without differencing
    order = (2, 1)
    model = VARMAX(df_scaled, order=order, trend='c', error_cov_type='diagonal')
    model_fitted = model.fit(disp=False)

    # Forecast
    fc = model_fitted.get_forecast(steps=steps)
    df_forecast = fc.predicted_mean
    df_forecast = pd.DataFrame(df_forecast, index=pd.date_range(start=df.index[-1] + timedelta(days=1), periods=steps, freq='D'), columns=df.columns)

    # Reverse normalization
    df_results = pd.DataFrame(scaler.inverse_transform(df_forecast), columns=df.columns, index=df_forecast.index)

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

# Function to fit VARIMA model and return model results
def fit_varima_model(data, p, q):
    try:
        model = VAR(data)
        results = model.fit(maxlags=p, ic=None)  # Fit model with maxlags
        aic = results.aic
        bic = results.bic
        return aic, bic, results
    except Exception as e:
        print(f"Error for p={p}, q={q}: {e}")
        return np.inf, np.inf, None

# Load your data
df = load_data()

# Define range for p and q
p_values = range(1, 4)  # Adjust the range as necessary
q_values = range(1, 4)  # Adjust the range as necessary

# Find the best (p, q) combination
best_aic = np.inf
best_bic = np.inf
best_order_aic = None
best_order_bic = None
best_model_aic = None
best_model_bic = None

for p, q in product(p_values, q_values):
    aic, bic, model = fit_varima_model(df, p, q)
    if aic < best_aic:
        best_aic = aic
        best_order_aic = (p, q)
        best_model_aic = model
    if bic < best_bic:
        best_bic = bic
        best_order_bic = (p, q)
        best_model_bic = model

print(f"Best AIC: {best_aic} with order {best_order_aic}")
print(f"Best BIC: {best_bic} with order {best_order_bic}")

# Print parameter estimates for the best model
if best_model_aic is not None:
    print("Best model parameters based on AIC:")
    print(best_model_aic.params)

if best_model_bic is not None:
    print("Best model parameters based on BIC:")
    print(best_model_bic.params)
