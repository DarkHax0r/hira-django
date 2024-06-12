import pandas as pd
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
import statsmodels.api as sm
import statsmodels.tsa.api as tsa


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


@login_required
def dashboard(request):
    month_choices = [(i, datetime(2000, i, 1).strftime("%B")) for i in range(1, 13)]
    year_choices = list(range(2023, 2031))

    if request.method == "POST":
        month = int(request.POST.get("month"))
        year = int(request.POST.get("year"))

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
        best_order, best_model = identify_varima_order(df)

        context = {
            "parfum": data,
            "adf_pendapatan": adf_pendapatan,
            "adf_modal": adf_modal,
            "varima_order": best_order,
            "varima_aic": best_model.aic if best_model else None,
            "varima_bic": best_model.bic if best_model else None,
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


# The analyze_data function was not intended to be a view, it's used internally
def analyze_data(df, steps):
    df_diff = df.diff().dropna()
    model = VAR(df_diff)
    model_fitted = model.fit(2)  # Assume the optimal lag is 2
    forecast_input = df_diff.values[-2:]
    fc = model_fitted.forecast(y=forecast_input, steps=steps)
    df_forecast = pd.DataFrame(
        fc,
        index=pd.date_range(
            start=df.index[-1] + timedelta(days=1), periods=steps, freq="D"
        ),
        columns=df.columns,
    )

    def invert_transformation(df_train, df_forecast):
        df_fc = df_forecast.copy()
        for col in df_train.columns:
            df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
        return df_fc

    df_results = invert_transformation(df, df_forecast)
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
