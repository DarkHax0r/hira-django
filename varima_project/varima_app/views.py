import pandas as pd
from statsmodels.tsa.api import VAR
from .models import ParfumData
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash
import openpyxl

def load_data():
    data = ParfumData.objects.all().values("date", "pendapatan", "modal")
    df = pd.DataFrame(data)

    # Pastikan kolom 'date' menjadi index dan pastikan formatnya benar
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Konversi kolom ke tipe numerik
    df["pendapatan"] = pd.to_numeric(df["pendapatan"], errors="coerce")
    df["modal"] = pd.to_numeric(df["modal"], errors="coerce")

    return df

def analyze_data(request):
    df = load_data()
    df_diff = df.diff().dropna()
    model = VAR(df_diff)
    model_fitted = model.fit(2)  # asumsikan lag optimal adalah 2
    forecast_input = df_diff.values[-2:]
    fc = model_fitted.forecast(y=forecast_input, steps=10)
    df_forecast = pd.DataFrame(
        fc,
        index=pd.date_range(start=df_diff.index[-1], periods=10, freq="D"),
        columns=df.columns,
    )

    def invert_transformation(df_train, df_forecast, diff_df):
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            df_fc[col] = df_train[col].iloc[-1] + df_fc[col].cumsum()
        return df_fc

    df_results = invert_transformation(df, df_forecast, df_diff)
    context = {"forecast": df_results.to_html()}
    return render(request, "pages/results.html", context)

@login_required
def dashboard(request):
    return render(request, "layout/base.html")

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

@login_required
def laporan(request):
    data = ParfumData.objects.all()
    return render(request, "laporan/laporan.html", {"parfum": data})

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
        file = request.FILES['file']
        try:
            # Membaca file Excel
            df = pd.read_excel(file)
            
            # Mencari kolom yang sesuai dengan mengabaikan case sensitivity
            col_map = {
                'Tanggal': None,
                'Pendapatan': None,
                'Modal': None
            }
            for col in df.columns:
                lower_col = col.lower()
                if 'tanggal' in lower_col:
                    col_map['Tanggal'] = col
                elif 'pendapatan' in lower_col:
                    col_map['Pendapatan'] = col
                elif 'modal' in lower_col:
                    col_map['Modal'] = col
            
            # Pastikan kolom-kolom yang dibutuhkan ada
            if not all(col_map.values()):
                missing_cols = [key for key, value in col_map.items() if value is None]
                messages.error(request, f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
                return redirect("laporan")
            
            # Mengubah nama kolom agar sesuai dengan model
            df.rename(columns={
                col_map['Tanggal']: 'date',
                col_map['Pendapatan']: 'pendapatan',
                col_map['Modal']: 'modal'
            }, inplace=True)
            
            # Mengonversi kolom tanggal ke format datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Memastikan semua kolom lainnya ke tipe numerik
            df['pendapatan'] = pd.to_numeric(df['pendapatan'], errors='coerce')
            df['modal'] = pd.to_numeric(df['modal'], errors='coerce')
            
            # Menghapus baris dengan nilai yang hilang
            df.dropna(subset=['date', 'pendapatan', 'modal'], inplace=True)
            
            # Menyimpan data ke dalam database
            for row in df.itertuples():
                ParfumData.objects.create(date=row.date, pendapatan=row.pendapatan, modal=row.modal)
                
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
    return render(request, 'profile/profile.html', {'user': request.user})

@login_required
def update_password(request):
    if request.method == 'POST':
        new_password = request.POST['new_password']
        confirm_password = request.POST['confirm_password']
        if new_password == confirm_password:
            request.user.set_password(new_password)
            request.user.save()
            update_session_auth_hash(request, request.user)  # This is the key to keep the user logged in
            messages.success(request, 'Password berhasil diperbarui!')
        else:
            messages.error(request, 'Password tidak cocok!')
    return redirect('profile')
