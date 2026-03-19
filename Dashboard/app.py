
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard Adopsi Fintech Gen Z", page_icon="📊", layout="wide")

COLORS = {
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "line": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
    "blue": "#2563EB",
    "blue2": "#DBEAFE",
    "good": "#DCFCE7",
    "warn": "#FEF3C7",
}

CONSTRUCT_COLUMNS = [
    "Accessibility",
    "Customer_Support",
    "Security",
    "Perceived_Usefulness",
    "Perceived_Ease_of_Use",
    "Subjective_Norm",
    "Perceived_Behavioral_Control",
    "Behavioral_Intention",
]

PERCEPTION_COLUMNS = [
    "Accessibility",
    "Customer_Support",
    "Security",
    "Perceived_Usefulness",
    "Perceived_Ease_of_Use",
    "Subjective_Norm",
    "Perceived_Behavioral_Control",
]

PREDICTOR_COLUMNS = [
    "Accessibility",
    "Customer_Support",
    "Security",
    "Perceived_Usefulness",
    "Perceived_Ease_of_Use",
    "Subjective_Norm",
    "Perceived_Behavioral_Control",
]

EXPECTED_FILES = {
    "data_dashboard.csv": "Data utama dashboard",
    "model_results_dashboard.csv": "Hasil akurasi model",
    "Hasil_Final_Overfitting_Semua_Algoritma.csv": "Train vs test accuracy",
    "Rangking_Fitur_SMOTE_RFE_SVM_BI.csv": "Ranking fitur",
    "Subset_SMOTE_RFE_SVM_Linear_BI.csv": "Fitur terpilih",
}

BUSINESS_NAMES = {
    "Accessibility": "kemudahan akses layanan",
    "Customer_Support": "dukungan layanan pelanggan",
    "Security": "rasa aman saat menggunakan fintech",
    "Perceived_Usefulness": "manfaat yang dirasakan",
    "Perceived_Ease_of_Use": "kemudahan penggunaan",
    "Subjective_Norm": "pengaruh lingkungan sosial",
    "Perceived_Behavioral_Control": "rasa mampu dan percaya diri untuk menggunakan fintech",
    "Behavioral_Intention": "niat mengadopsi fintech",
}

DISPLAY_NAMES = {
    "Accessibility": "Accessibility",
    "Customer_Support": "Customer Support",
    "Security": "Security",
    "Perceived_Usefulness": "Perceived Usefulness",
    "Perceived_Ease_of_Use": "Perceived Ease of Use",
    "Subjective_Norm": "Subjective Norm",
    "Perceived_Behavioral_Control": "Perceived Behavioral Control",
    "Behavioral_Intention": "Behavioral Intention",
}

def add_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background: {COLORS["bg"]};
    }}
    .block-container {{
        max-width: 1450px;
        padding-top: 5.2rem;
        padding-bottom: 2rem;
    }}
    .stAppDeployButton {{
        top: 0.75rem;
    }}
    .title {{
        font-size: 2rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-top: 0.5rem;
        margin-bottom: 0.35rem;
        line-height: 1.25;
    }}
    .subtitle {{
        font-size: 0.95rem;
        color: {COLORS["muted"]};
        margin-bottom: 1rem;
    }}
    .section {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {COLORS["text"]};
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }}
    .card {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["line"]};
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }}
    .kpi {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["line"]};
        border-radius: 18px;
        padding: 1rem;
        min-height: 110px;
    }}
    .kpi-label {{
        color: {COLORS["muted"]};
        font-size: 0.85rem;
    }}
    .kpi-value {{
        color: {COLORS["text"]};
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }}
    .note {{
        color: {COLORS["muted"]};
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }}
    .insight {{
        background: {COLORS["blue2"]};
        border: 1px solid #BFDBFE;
        color: {COLORS["text"]};
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.93rem;
    }}
    .takeaway {{
        background: {COLORS["good"]};
        border: 1px solid #86EFAC;
        color: {COLORS["text"]};
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.93rem;
    }}
    .risk {{
        background: {COLORS["warn"]};
        border: 1px solid #FCD34D;
        color: {COLORS["text"]};
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.93rem;
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def find_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd(), here, here.parent]
    for base in candidates:
        if (base / "data_dashboard.csv").exists():
            return base
    return here

@st.cache_data
def load_csv(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None

def load_all() -> Dict[str, Optional[pd.DataFrame]]:
    root = find_root()
    return {name: load_csv(str(root / name)) for name in EXPECTED_FILES}

def fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "-"

def build_df(data_df: pd.DataFrame) -> pd.DataFrame:
    df = data_df.copy()
    if "Label" in df.columns and "Pseudo_Label" not in df.columns:
        df["Pseudo_Label"] = df["Label"]
    if "Tingkat_Adopsi" not in df.columns and "Pseudo_Label" in df.columns:
        mapper = {
            "BI-Low": "Rendah",
            "BI-Moderate": "Menengah",
            "BI-High": "Tinggi",
        }
        df["Tingkat_Adopsi"] = df["Pseudo_Label"].map(mapper).fillna("Tidak diketahui")
    return df

def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter")
    out = df.copy()

    if "Pseudo_Label" in out.columns:
        segs = ["Semua"] + sorted(out["Pseudo_Label"].dropna().astype(str).unique().tolist())
        seg = st.sidebar.selectbox("Pilih segmen", segs)
        if seg != "Semua":
            out = out[out["Pseudo_Label"].astype(str) == seg]

    if "Tingkat_Adopsi" in out.columns:
        levels = ["Semua"] + sorted(out["Tingkat_Adopsi"].dropna().astype(str).unique().tolist())
        level = st.sidebar.selectbox("Pilih tingkat adopsi", levels)
        if level != "Semua":
            out = out[out["Tingkat_Adopsi"].astype(str) == level]

    st.sidebar.markdown("### File terbaca")
    root = find_root()
    for name in EXPECTED_FILES:
        status = "✅" if (root / name).exists() else "❌"
        st.sidebar.caption(f"{status} {name}")

    return out

def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
    )
    return fig

def kpi_box(label, value, note=""):
    st.markdown(f"""
    <div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)

def insight_box(text, kind="insight"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)

def overview(df: pd.DataFrame, model_df: Optional[pd.DataFrame]):
    section_title("1. Ringkasan Utama")

    total_resp = len(df)
    cluster_count = int(df["Cluster"].nunique()) if "Cluster" in df.columns else 0
    bi_mean = df["Behavioral_Intention"].mean() if "Behavioral_Intention" in df.columns else np.nan
    best_model = "-"
    best_acc = None

    if model_df is not None and {"Algoritma", "Akurasi"}.issubset(model_df.columns):
        top = model_df.sort_values("Akurasi", ascending=False).iloc[0]
        best_model = str(top["Algoritma"])
        best_acc = float(top["Akurasi"])

    a, b, c, d = st.columns(4)
    with a:
        kpi_box("Jumlah responden", fmt_int(total_resp), "Data yang tampil setelah filter")
    with b:
        kpi_box("Jumlah cluster", fmt_int(cluster_count), "Segmen hasil pengolahan Python")
    with c:
        kpi_box("Rata-rata Behavioral Intention", f"{bi_mean:.2f}/4" if pd.notna(bi_mean) else "-", "Makin tinggi makin baik")
    with d:
        note = f"Akurasi {best_acc:.2f}%" if best_acc is not None else "Menunggu file model"
        kpi_box("Model terbaik", best_model, note)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    available = [c for c in PERCEPTION_COLUMNS if c in df.columns]
    means = df[available].mean().sort_values(ascending=True).reset_index()
    means.columns = ["Variabel", "Rata-rata"]
    fig = px.bar(
        means,
        x="Rata-rata",
        y="Variabel",
        orientation="h",
        text="Rata-rata",
        color="Rata-rata",
        color_continuous_scale="Blues"
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    chart_layout(fig, 380)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    strongest = means.sort_values("Rata-rata", ascending=False).iloc[0]
    weakest = means.sort_values("Rata-rata", ascending=True).iloc[0]
    insight_box(
        f"Temuan utama. Persepsi paling kuat ada pada <b>{strongest['Variabel']}</b> dengan rata-rata <b>{strongest['Rata-rata']:.2f}</b>. "
        f"Artinya responden paling positif pada aspek {BUSINESS_NAMES.get(strongest['Variabel'], strongest['Variabel'])}. "
        f"Sebaliknya, area terlemah ada pada <b>{weakest['Variabel']}</b> dengan rata-rata <b>{weakest['Rata-rata']:.2f}</b>, sehingga ini layak menjadi prioritas perbaikan."
    )
    if pd.notna(bi_mean):
        if bi_mean >= 3.4:
            insight_box("Makna bisnis. Niat adopsi sudah relatif kuat. Fokus strategi bisa diarahkan ke konversi, cross sell, dan retensi.", "takeaway")
        elif bi_mean >= 2.8:
            insight_box("Makna bisnis. Niat adopsi berada di level menengah. Pasar masih potensial, tetapi butuh dorongan yang lebih jelas pada faktor penghambat.", "risk")
        else:
            insight_box("Makna bisnis. Niat adopsi masih lemah. Perusahaan perlu memperbaiki value proposition, trust, dan pengalaman awal pengguna.", "risk")
    st.markdown('</div>', unsafe_allow_html=True)

def perception(df: pd.DataFrame):
    section_title("2. Perbandingan Persepsi Konsumen")
    available = [c for c in PERCEPTION_COLUMNS if c in df.columns]

    left, right = st.columns(2)

    means = df[available].mean().sort_values(ascending=False).reset_index()
    means.columns = ["Variabel", "Rata-rata"]
    strongest = means.iloc[0]
    weakest = means.iloc[-1]

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.bar(
            means,
            x="Variabel",
            y="Rata-rata",
            text="Rata-rata",
            color="Rata-rata",
            color_continuous_scale="Blues"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        chart_layout(fig, 360)
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        insight_box(
            f"Variabel dengan skor tertinggi adalah <b>{strongest['Variabel']}</b>. Ini menunjukkan kekuatan utama persepsi konsumen ada pada aspek "
            f"{BUSINESS_NAMES.get(strongest['Variabel'], strongest['Variabel'])}. "
            f"Variabel dengan skor terendah adalah <b>{weakest['Variabel']}</b>. Area ini paling berpotensi menahan peningkatan adopsi."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        long_df = df[available].melt(var_name="Variabel", value_name="Skor")
        fig = px.box(long_df, x="Variabel", y="Skor")
        chart_layout(fig, 360)
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        stds = df[available].std().sort_values(ascending=False)
        varied = stds.index[0]
        insight_box(
            f"Sebaran jawaban paling beragam terdapat pada <b>{varied}</b>. Artinya persepsi responden pada aspek ini belum seragam. "
            f"Ini bisa menandakan adanya kelompok pengguna yang sangat menerima, tetapi ada juga yang masih ragu."
        )
        st.markdown('</div>', unsafe_allow_html=True)

def drivers(df: pd.DataFrame, feature_df: Optional[pd.DataFrame]):
    section_title("3. Faktor yang Berkaitan dengan Niat Adopsi")

    left, right = st.columns([1.1, 0.9])

    drivers = [c for c in CONSTRUCT_COLUMNS if c in df.columns and c != "Behavioral_Intention"]
    corr = (
        df[drivers + ["Behavioral_Intention"]]
        .corr(numeric_only=True)["Behavioral_Intention"]
        .drop("Behavioral_Intention")
        .sort_values(ascending=True)
        .reset_index()
    )
    corr.columns = ["Variabel", "Korelasi"]
    top_driver = corr.sort_values("Korelasi", ascending=False).iloc[0]
    weak_driver = corr.sort_values("Korelasi", ascending=True).iloc[0]

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.bar(
            corr,
            x="Korelasi",
            y="Variabel",
            orientation="h",
            text="Korelasi",
            color="Korelasi",
            color_continuous_scale="Blues"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        chart_layout(fig, 360)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        insight_box(
            f"Faktor yang paling kuat berkaitan dengan niat adopsi adalah <b>{top_driver['Variabel']}</b> dengan korelasi <b>{top_driver['Korelasi']:.2f}</b>. "
            f"Artinya peningkatan pada aspek ini paling berpotensi diikuti kenaikan niat penggunaan fintech. "
            f"Hubungan paling lemah terdapat pada <b>{weak_driver['Variabel']}</b>."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if feature_df is not None and {"Fitur Asli", "Ranking (1 = Terpilih)"}.issubset(feature_df.columns):
            temp = feature_df.copy()
            temp["Ranking (1 = Terpilih)"] = pd.to_numeric(temp["Ranking (1 = Terpilih)"], errors="coerce")
            temp = temp.sort_values("Ranking (1 = Terpilih)", ascending=True).head(10)
            fig = px.bar(
                temp.iloc[::-1],
                x="Ranking (1 = Terpilih)",
                y="Fitur Asli",
                orientation="h",
                text="Ranking (1 = Terpilih)"
            )
            chart_layout(fig, 360)
            st.plotly_chart(fig, use_container_width=True)
            top_features = ", ".join(temp["Fitur Asli"].astype(str).tolist()[:3])
            insight_box(
                f"Hasil seleksi fitur menunjukkan fitur yang paling diprioritaskan model antara lain <b>{top_features}</b>. "
                f"Ini penting karena model tidak memilih fitur secara acak, tetapi berdasarkan kontribusinya terhadap prediksi."
            )
        else:
            st.info("File ranking fitur belum tersedia.")
        st.markdown('</div>', unsafe_allow_html=True)

def segmentation(df: pd.DataFrame):
    section_title("4. Segmentasi Konsumen")
    if "Cluster" not in df.columns:
        st.info("Kolom Cluster belum tersedia.")
        return

    left, right = st.columns(2)

    seg = df["Cluster"].value_counts().sort_index().reset_index()
    seg.columns = ["Cluster", "Jumlah Responden"]

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if "Pseudo_Label" in df.columns:
            label_map = df.groupby("Cluster")["Pseudo_Label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).reset_index()
            seg = seg.merge(label_map, on="Cluster", how="left")
            fig = px.bar(seg, x="Cluster", y="Jumlah Responden", color="Pseudo_Label", text="Jumlah Responden")
        else:
            fig = px.bar(seg, x="Cluster", y="Jumlah Responden", text="Jumlah Responden")
        chart_layout(fig, 340)
        st.plotly_chart(fig, use_container_width=True)
        dominant = seg.sort_values("Jumlah Responden", ascending=False).iloc[0]
        label_text = f" atau segmen <b>{dominant['Pseudo_Label']}</b>" if "Pseudo_Label" in seg.columns else ""
        insight_box(
            f"Cluster dengan jumlah responden terbesar adalah <b>{int(dominant['Cluster'])}</b>{label_text}. "
            f"Segmen ini paling representatif dalam sampel, sehingga layak menjadi fokus utama strategi awal."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        prof = df.groupby("Cluster")[[c for c in CONSTRUCT_COLUMNS if c in df.columns]].mean().reset_index()
        prof_long = prof.melt(id_vars="Cluster", var_name="Variabel", value_name="Rata-rata")
        fig = px.bar(
            prof_long,
            x="Variabel",
            y="Rata-rata",
            color="Cluster",
            barmode="group",
            text="Rata-rata"
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        chart_layout(fig, 380)
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        if "Behavioral_Intention" in prof.columns:
            top_seg = prof.sort_values("Behavioral_Intention", ascending=False).iloc[0]
            low_seg = prof.sort_values("Behavioral_Intention", ascending=True).iloc[0]
            insight_box(
                f"Cluster dengan niat adopsi tertinggi adalah <b>{int(top_seg['Cluster'])}</b> dengan rata-rata Behavioral Intention <b>{top_seg['Behavioral_Intention']:.2f}</b>. "
                f"Sementara cluster dengan niat terendah adalah <b>{int(low_seg['Cluster'])}</b>. "
                f"Ini menunjukkan bahwa strategi tidak bisa disamaratakan untuk semua kelompok."
            )
        st.markdown('</div>', unsafe_allow_html=True)

def model_section(model_df: Optional[pd.DataFrame], overfit_df: Optional[pd.DataFrame]):
    section_title("5. Hasil Model Prediksi")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if model_df is not None and {"Algoritma", "Akurasi"}.issubset(model_df.columns):
            temp = model_df.sort_values("Akurasi", ascending=True)
            fig = px.bar(
                temp,
                x="Akurasi",
                y="Algoritma",
                orientation="h",
                text="Akurasi",
                color="Akurasi",
                color_continuous_scale="Blues"
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            chart_layout(fig, 350)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            best = temp.sort_values("Akurasi", ascending=False).iloc[0]
            worst = temp.sort_values("Akurasi", ascending=True).iloc[0]
            insight_box(
                f"Model dengan akurasi tertinggi adalah <b>{best['Algoritma']}</b> sebesar <b>{best['Akurasi']:.2f}%</b>. "
                f"Model dengan akurasi terendah adalah <b>{worst['Algoritma']}</b>. "
                f"Ini menunjukkan bahwa performa model berbeda, sehingga pemilihan model terbaik perlu berbasis hasil evaluasi."
            )
        else:
            st.info("File model_results_dashboard.csv belum tersedia.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if overfit_df is not None and {"Model", "Train Acc", "Test Acc"}.issubset(overfit_df.columns):
            temp = overfit_df.copy()
            temp["Train Acc Num"] = pd.to_numeric(temp["Train Acc"].astype(str).str.replace("%", "", regex=False), errors="coerce")
            temp["Test Acc Num"] = pd.to_numeric(temp["Test Acc"].astype(str).str.replace("%", "", regex=False), errors="coerce")
            long_df = temp.melt(
                id_vars="Model",
                value_vars=["Train Acc Num", "Test Acc Num"],
                var_name="Tipe",
                value_name="Akurasi"
            )
            long_df["Tipe"] = long_df["Tipe"].replace({
                "Train Acc Num": "Training Accuracy",
                "Test Acc Num": "Testing Accuracy"
            })
            fig = px.bar(
                long_df,
                x="Model",
                y="Akurasi",
                color="Tipe",
                barmode="group",
                text="Akurasi"
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            chart_layout(fig, 350)
            fig.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
            if "Gap" in temp.columns:
                gap_num = pd.to_numeric(temp["Gap"].astype(str).str.replace("%", "", regex=False), errors="coerce")
                safe = temp.loc[gap_num.idxmin()] if gap_num.notna().any() else None
                risky = temp.loc[gap_num.idxmax()] if gap_num.notna().any() else None
                if safe is not None and risky is not None:
                    insight_box(
                        f"Model paling stabil adalah <b>{safe['Model']}</b> karena selisih train dan test paling kecil. "
                        f"Model yang paling perlu dicermati adalah <b>{risky['Model']}</b> karena gap train dan test paling besar."
                    )
        else:
            st.info("File overfitting belum tersedia.")
        st.markdown('</div>', unsafe_allow_html=True)



def prediction_form(df: pd.DataFrame):
    section_title("6. Customer Scenario Simulator")
    if "Cluster" not in df.columns:
        st.info("Fitur ini butuh kolom Cluster di data_dashboard.csv")
        return

    cluster_profile = df.groupby("Cluster")[[c for c in CONSTRUCT_COLUMNS if c in df.columns]].mean().reset_index()

    if "Pseudo_Label" in df.columns:
        label_map = df.groupby("Cluster")["Pseudo_Label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).reset_index()
        cluster_profile = cluster_profile.merge(label_map, on="Cluster", how="left")
    if "Tingkat_Adopsi" in df.columns:
        level_map = df.groupby("Cluster")["Tingkat_Adopsi"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).reset_index()
        cluster_profile = cluster_profile.merge(level_map, on="Cluster", how="left")

    business_questions = {
        "Accessibility": "Seberapa mudah target pengguna mengakses layanan fintech yang ditawarkan.",
        "Customer_Support": "Seberapa baik target pengguna memandang dukungan layanan pelanggan.",
        "Security": "Seberapa tinggi rasa aman target pengguna terhadap transaksi dan data di fintech.",
        "Perceived_Usefulness": "Seberapa besar target pengguna melihat manfaat nyata dari penggunaan fintech.",
        "Perceived_Ease_of_Use": "Seberapa mudah target pengguna merasa dapat memahami dan memakai layanan fintech.",
        "Subjective_Norm": "Seberapa kuat pengaruh lingkungan sosial target pengguna untuk memakai fintech.",
        "Perceived_Behavioral_Control": "Seberapa yakin target pengguna bahwa mereka mampu menggunakan fintech secara mandiri.",
    }

    likert_labels = {
        1: "Sangat Rendah",
        2: "Rendah",
        3: "Tinggi",
        4: "Sangat Tinggi",
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Gunakan simulator ini untuk memodelkan profil persepsi calon target pengguna. Nilai 1 sampai 4 menggambarkan kondisi target pasar yang ingin diuji.")
    st.caption("1 = Sangat Rendah, 2 = Rendah, 3 = Tinggi, 4 = Sangat Tinggi")

    with st.form("business_scenario_form"):
        inputs = {}
        for i, col in enumerate(PREDICTOR_COLUMNS, start=1):
            st.markdown(f"#### {i}. {business_questions[col]}")
            inputs[col] = st.radio(
                label=f"Nilai untuk {DISPLAY_NAMES[col]}",
                options=[1, 2, 3, 4],
                horizontal=True,
                format_func=lambda x: f"{x}. {likert_labels[x]}",
                key=f"radio_{col}",
                label_visibility="collapsed",
            )
            st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Simulasikan profil target", use_container_width=True)

    if submitted:
        user_vector = pd.Series(inputs)
        usable_cols = [c for c in PREDICTOR_COLUMNS if c in cluster_profile.columns]
        temp = cluster_profile.copy()
        temp["distance"] = temp[usable_cols].apply(
            lambda row: float(np.linalg.norm(row[usable_cols].values - user_vector[usable_cols].values)),
            axis=1
        )
        best_match = temp.sort_values("distance", ascending=True).iloc[0]

        strengths = user_vector.sort_values(ascending=False).head(2).index.tolist()
        weaknesses = user_vector.sort_values(ascending=True).head(2).index.tolist()

        segment_label = str(best_match["Pseudo_Label"]) if "Pseudo_Label" in best_match.index else f"Cluster {int(best_match['Cluster'])}"
        adoption_level = str(best_match["Tingkat_Adopsi"]) if "Tingkat_Adopsi" in best_match.index else "-"
        matched_cluster = int(best_match["Cluster"])

        st.markdown("### Hasil simulasi target user")
        a, b, c = st.columns(3)
        with a:
            kpi_box("Kategori target", segment_label, "Kategori terdekat dari data historis")
        with b:
            kpi_box("Tingkat adopsi", adoption_level, "Potensi adopsi calon target")
        with c:
            kpi_box("Cluster acuan", str(matched_cluster), "Kelompok referensi paling mirip")

        reason_text = (
            f"Profil target yang Anda simulasikan paling dekat dengan kategori <b>{segment_label}</b>. "
            f"Ini berarti pola persepsi target pasar tersebut paling mirip dengan responden pada kelompok itu. "
            f"Kekuatan utamanya ada pada <b>{DISPLAY_NAMES[strengths[0]]}</b> dan <b>{DISPLAY_NAMES[strengths[1]]}</b>. "
            f"Hambatan utamanya ada pada <b>{DISPLAY_NAMES[weaknesses[0]]}</b> dan <b>{DISPLAY_NAMES[weaknesses[1]]}</b>."
        )
        insight_box(reason_text)

        cluster_bi = float(best_match["Behavioral_Intention"]) if "Behavioral_Intention" in best_match.index else np.nan
        if pd.notna(cluster_bi):
            if cluster_bi >= 3.4:
                insight_box(
                    "Makna bisnis. Profil target ini dekat dengan kelompok yang siap diaktivasi. Perusahaan dapat fokus pada konversi, aktivasi fitur, dan peningkatan frekuensi penggunaan.",
                    "takeaway",
                )
            elif cluster_bi >= 2.8:
                insight_box(
                    "Makna bisnis. Profil target ini berada pada area menengah. Perusahaan masih perlu memperkuat manfaat, rasa aman, dan kemudahan penggunaan agar target bergeser ke kategori adopsi tinggi.",
                    "risk",
                )
            else:
                insight_box(
                    "Makna bisnis. Profil target ini masih dekat dengan kelompok adopsi rendah. Prioritas strategi sebaiknya dimulai dari trust building, edukasi manfaat, dan onboarding yang lebih sederhana.",
                    "risk",
                )

        rec_map = {
            "Accessibility": "Perluas titik akses, sederhanakan entry point, dan perjelas jalur menuju fitur utama.",
            "Customer_Support": "Perkuat layanan bantuan, live chat, dan respons komplain agar hambatan penggunaan cepat turun.",
            "Security": "Tonjolkan bukti keamanan, proteksi akun, notifikasi transaksi, dan komunikasi anti-fraud.",
            "Perceived_Usefulness": "Fokuskan komunikasi pada manfaat nyata, efisiensi waktu, dan relevansi use case harian.",
            "Perceived_Ease_of_Use": "Sederhanakan alur onboarding, kurangi langkah, dan perjelas navigasi inti aplikasi.",
            "Subjective_Norm": "Gunakan social proof, testimoni, referral, dan kampanye berbasis komunitas.",
            "Perceived_Behavioral_Control": "Tambahkan tutorial, panduan langkah demi langkah, dan bantuan di momen pertama penggunaan.",
        }

        st.markdown("#### Rekomendasi aksi untuk perusahaan")
        rc1, rc2 = st.columns(2)
        with rc1:
            insight_box(
                f"Prioritas intervensi 1. <b>{DISPLAY_NAMES[weaknesses[0]]}</b>. {rec_map[weaknesses[0]]}",
                "risk",
            )
        with rc2:
            insight_box(
                f"Prioritas intervensi 2. <b>{DISPLAY_NAMES[weaknesses[1]]}</b>. {rec_map[weaknesses[1]]}",
                "risk",
            )

        st.markdown("#### Penjelasan kedekatan dengan kategori target")
        explain_df = pd.DataFrame({
            "Variabel": [DISPLAY_NAMES[c] for c in usable_cols],
            "Profil target": [user_vector[c] for c in usable_cols],
            "Rata-rata kategori terdekat": [round(float(best_match[c]), 2) for c in usable_cols],
            "Selisih": [round(abs(float(user_vector[c]) - float(best_match[c])), 2) for c in usable_cols],
        })
        st.dataframe(explain_df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


def recommendations(df: pd.DataFrame):
    section_title("7. Inti Temuan dan Implikasi")
    available = [c for c in PERCEPTION_COLUMNS if c in df.columns]
    means = df[available].mean().sort_values(ascending=False)

    strongest = means.index[0] if len(means) else "-"
    weakest = means.index[-1] if len(means) else "-"
    dominant_seg = df["Pseudo_Label"].astype(str).value_counts().idxmax() if "Pseudo_Label" in df.columns else "-"
    bi_mean = df["Behavioral_Intention"].mean() if "Behavioral_Intention" in df.columns else np.nan

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="card"><b>Kekuatan utama.</b><br><br>Aspek yang paling kuat adalah <b>{strongest}</b>. Strategi komunikasi bisa menonjolkan aspek ini sebagai alasan utama menggunakan fintech.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><b>Area prioritas.</b><br><br>Aspek yang paling lemah adalah <b>{weakest}</b>. Jika perusahaan ingin menaikkan adopsi, intervensi paling logis dimulai dari area ini.</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><b>Fokus pasar.</b><br><br>Segmen dominan adalah <b>{dominant_seg}</b>. Segmen ini dapat dijadikan target utama karena paling banyak muncul dalam data.</div>', unsafe_allow_html=True)

    if pd.notna(bi_mean):
        if bi_mean >= 3.4:
            insight_box("Implikasi strategis. Pasar sudah memiliki kesiapan adopsi yang baik. Perusahaan dapat fokus pada konversi yang lebih agresif, retensi, dan peningkatan frekuensi penggunaan.", "takeaway")
        elif bi_mean >= 2.8:
            insight_box("Implikasi strategis. Pasar belum sepenuhnya kuat. Perusahaan perlu memperjelas manfaat, menyederhanakan pengalaman penggunaan, dan mengurangi keraguan pengguna.", "risk")
        else:
            insight_box("Implikasi strategis. Pasar masih menahan diri untuk mengadopsi fintech. Fokus utama sebaiknya pada edukasi, trust building, dan pengurangan hambatan awal.", "risk")

def tables(bundle: Dict[str, Optional[pd.DataFrame]]):
    section_title("8. Tabel Data Pendukung")
    tabs = st.tabs(["Data utama", "Model", "Overfitting", "Ranking fitur", "Fitur terpilih"])
    keys = list(EXPECTED_FILES.keys())
    for tab, key in zip(tabs, keys):
        with tab:
            df = bundle.get(key)
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info(f"{key} belum ada.")

def main():
    add_css()
    root = find_root()
    bundle = load_all()

    st.markdown('<div class="title">Dashboard Niat Adopsi Fintech Gen Z</div>', unsafe_allow_html=True)
   
    if bundle["data_dashboard.csv"] is None:
        st.error("data_dashboard.csv belum ditemukan. Jalankan dulu script Python sampai file output terbentuk.")
        st.stop()

    df = build_df(bundle["data_dashboard.csv"])
    df = sidebar_filters(df)

    if df.empty:
        st.warning("Data kosong setelah filter.")
        st.stop()

    overview(df, bundle["model_results_dashboard.csv"])
    perception(df)
    drivers(df, bundle["Rangking_Fitur_SMOTE_RFE_SVM_BI.csv"])
    segmentation(df)
    model_section(bundle["model_results_dashboard.csv"], bundle["Hasil_Final_Overfitting_Semua_Algoritma.csv"])
    prediction_form(df)
    recommendations(df)
    tables(bundle)

if __name__ == "__main__":
    main()
