import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="BGN Research | Analisis Sentimen MBG",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# CUSTOM CSS — LIGHT BLUE + WHITE, FRIENDLY
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global */
.stApp { background: #f0f5fa; }
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; color: #1e293b !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Force dark text everywhere */
p, span, li, label, div, .stMarkdown, .stText, [data-testid="stMarkdownContainer"] { color: #1e293b !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #334155 !important; }
[data-testid="stSidebar"] .stRadio label span { color: #1e293b !important; }
.stAlert p, .stAlert span { color: inherit !important; }
blockquote, blockquote p { color: #475569 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebarContent"] { padding-top: 1rem !important; }

/* Merapikan Navigasi (Menghilangkan bundaran radio & membuat style tombol) */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: transparent;
    padding: 10px 16px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    border: 1px solid transparent;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    border-color: #3b82f6 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) span,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* Headings */
h1 { color: #1e293b !important; font-weight: 800 !important; font-size: 1.6rem !important; }
h2 { color: #1e293b !important; font-weight: 700 !important; font-size: 1.25rem !important; }
h3 { color: #334155 !important; font-weight: 700 !important; font-size: 1.05rem !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff; padding: 18px 20px; border-radius: 14px;
    border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
[data-testid="stMetricValue"] { color: #1e293b !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* Images */
[data-testid="stImage"] img {
    border-radius: 12px; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}

/* Text area */
.stTextArea textarea {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important; padding: 14px !important;
    font-size: 15px !important; color: #1e293b !important;
}
.stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,.15) !important; }

/* Buttons */
.stButton button {
    background: #3b82f6 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    padding: 8px 24px !important; font-weight: 600 !important;
    font-size: 14px !important; transition: all .2s !important;
}
.stButton button:hover { background: #2563eb !important; transform: translateY(-1px) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0;
    padding: 8px 18px; font-size: .85rem; font-weight: 500;
}
.stTabs [aria-selected="true"] { background: #3b82f6 !important; color: white !important; border-color: #3b82f6 !important; }

/* Alerts */
.stAlert { border-radius: 12px !important; }

/* Expander */
.streamlit-expanderHeader { font-weight: 600 !important; font-size: .92rem !important; }

/* Divider */
hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# PATHS
# ───────────────────────────────────────────────
VIZ = "Visualisasi"
DS_DIR = "Dataset"

def viz(name):
    p = os.path.join(VIZ, name)
    return p if os.path.exists(p) else None

# ───────────────────────────────────────────────
# MODEL SVM (CACHED)
# ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        path = os.path.join(DS_DIR, "Dataset_BGN_Labeled_10000_by_LLM.csv")
        df = pd.read_csv(path)
        df['teks_bersih'] = df['teks_bersih'].fillna('')
        vec = TfidfVectorizer(max_features=10000, min_df=3, max_df=0.9)
        X = vec.fit_transform(df['teks_bersih'])
        mdl = LinearSVC(random_state=42, class_weight='balanced')
        mdl.fit(X, df['Label_Sentimen'])
        return vec, mdl
    except Exception as e:
        return None, None

# ───────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## BGN Research")
    st.caption("Analisis Sentimen & Topik MBG")
    st.divider()
    menu = st.radio(
        "Navigasi",
        ["Overview", "Sentimen (SVM)", "Topik (LDA)", "TextRank", "Galeri", "Demo"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Three-Stage Text Mining")
    st.caption("SVM → LDA → TextRank")

# ───────────────────────────────────────────────
# PAGE: OVERVIEW
# ───────────────────────────────────────────────
if menu == "Overview":
    st.markdown("# Overview")
    st.markdown("Ringkasan hasil analisis **47.934 komentar** Instagram @badangizinasional.ri tentang program Makan Bergizi Gratis (MBG).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", "47.934")
    c2.metric("Kontra", "39.994", "83.4%")
    c3.metric("Pro", "3.342", "7.0%")
    c4.metric("Netral", "4.598", "9.6%")

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Sentimen")
        p = viz("viz_1_distribusi_sentimen.png")
        if p: st.image(p, use_container_width=True)
    with col2:
        st.markdown("### Distribusi Topik Keluhan")
        p = viz("viz_lda_1_distribusi_topik.png")
        if p: st.image(p, use_container_width=True)

    st.markdown("### Tren Sentimen Harian")
    p = viz("viz_3_tren_harian.png")
    if p: st.image(p, use_container_width=True)

# ───────────────────────────────────────────────
# PAGE: SENTIMEN
# ───────────────────────────────────────────────
elif menu == "Sentimen (SVM)":
    st.markdown("# Analisis Sentimen SVM")
    st.markdown("Model **LinearSVC** dengan TF-IDF (`max_features=10000`) dan `class_weight='balanced'` untuk mengatasi ketimpangan data.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi", "88.65%")
    c2.metric("F1 Kontra", "0.94")
    c3.metric("F1 Pro", "0.62")
    c4.metric("F1 Netral", "0.42")

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix")
        p = viz("viz_svm_1_confusion_matrix.png")
        if p: st.image(p, use_container_width=True)
    with col2:
        st.markdown("### Distribusi Akhir Sentimen")
        p = viz("viz_svm_2_distribusi_akhir.png")
        if p: st.image(p, use_container_width=True)

    st.info("**Arsitektur:** Kolom `teks_bersih` (preservasi negasi) → TF-IDF 10K dimensi → LinearSVC(balanced). Akurasi Kontra tinggi (0.94) karena dominasi data.")

# ───────────────────────────────────────────────
# PAGE: TOPIK LDA
# ───────────────────────────────────────────────
elif menu == "Topik (LDA)":
    st.markdown("# Pemodelan Topik LDA")
    st.markdown("Unsupervised learning pada **39.994 komentar Kontra** menggunakan CountVectorizer + Custom Stopwords.")

    c1, c2 = st.columns(2)
    c1.metric("Perplexity", "834.51")
    c2.metric("Jumlah Topik", "5 Klaster")

    st.markdown("")

    topik_data = [
        ("Topik 1", "Ekspektasi Eksekusi di Sekolah", "makan · gizi · gratis · anak · sekolah · menu · dapat · kasih"),
        ("Topik 2", "Defisit Upah Pekerja Dapur (SPPG)", "kerja · gaji · sppg · dapur · tolong · badan · bapak · nasional"),
        ("Topik 3", "Kepanikan Keracunan & Korupsi", "gizi · makan · gratis · program · korupsi · racun · indonesia"),
        ("Topik 4", "Inefisiensi Pengadaan Tersier", "uang · motor · rakyat · anggar · beli · negara · pajak · kaki"),
        ("Topik 5", "Substitusi Pangan Ultra-Proses", "roti · susu · kacang · anak · beda · menu · kecil · telur"),
    ]

    for icon, title, keywords in topik_data:
        with st.expander(f"{icon}: {title}"):
            st.markdown(f"**Kata kunci:** _{keywords}_")

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Klaster")
        p = viz("viz_lda_1_distribusi_topik.png")
        if p: st.image(p, use_container_width=True)
    with col2:
        st.markdown("### Bobot Kata Kunci")
        p = viz("viz_lda_2_word_weights.png")
        if p: st.image(p, use_container_width=True)

    st.info("**Input:** Kolom `teks_preprocessed` (stemmed). Vektorisasi: CountVectorizer + 60 custom stopwords. Parameter: `n_components=5`.")

# ───────────────────────────────────────────────
# PAGE: TEXTRANK
# ───────────────────────────────────────────────
elif menu == "TextRank":
    st.markdown("# TextRank Summarization")
    st.markdown("Algoritma PageRank pada graf kemiripan kosinus yang mengekstrak komentar paling sentral secara **verbatim**.")

    quotes = [
        ("Topik 1: Ekspektasi Sekolah", "#6366f1", [
            "MAKAN GRATIS TIDAK BERGIZI"
        ]),
        ("Topik 2: Jeritan Pekerja Dapur", "#ef4444", [
            "Gaji AK AG yg diambil dari operasional yg seolah kami disamakan dengan relawan... AK bukan robot yg ngak butuh istirahat, kami juga butuh istirahat."
        ]),
        ("Topik 3: Keracunan & Korupsi", "#f59e0b", [
            "MBG = Makan Beracun Gratis",
            "Bubarkan saja BGN dengan program MBG nya"
        ]),
        ("Topik 4: Pengadaan Tersier", "#a855f7", [
            "Apa urgensi nya sampai beli motor listrik seharga 42 juta??? Bukan nya negara lagi efisiensi??",
            "Kepala BGN kya ga ada malu nya ya mark up anggaran pake duit rakyat.."
        ]),
        ("Topik 5: Substitusi Pangan", "#10b981", [
            "Kenyataan realnya menu rapelan 3hr dapet nya cuma kurma 3biji, susu 125ml, kacang seuprit, pisang, roti",
            "Jauh bgt ama real di lapangan. Sekolah anak ku cuma dpt susu 125ml, roti abon 1rb, kurma, jeruk, pisang"
        ]),
    ]

    for title, color, texts in quotes:
        with st.container():
            st.markdown(f"#### {title}")
            for t in texts:
                st.markdown(f"> *\"{t}\"*")
            st.markdown("")

    st.markdown("### Graf Sentralitas PageRank")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Topik 1", "Topik 2", "Topik 3", "Topik 4", "Topik 5"])
    for i, tab in enumerate([tab1, tab2, tab3, tab4, tab5], 1):
        with tab:
            p = viz(f"viz_textrank_topik_{i}.png")
            if p: st.image(p, use_container_width=True)

    st.info("**Dual-Column Processing:** Cosine similarity pada `teks_preprocessed`, output verbatim dari `teks_komentar`. Algoritma: `nx.pagerank()` pada graf TF-IDF.")

# ───────────────────────────────────────────────
# PAGE: GALERI
# ───────────────────────────────────────────────
elif menu == "Galeri":
    st.markdown("# Galeri Visualisasi")
    st.markdown("15 artefak grafik yang dihasilkan selama pipeline pemrosesan data.")

    tab_prep, tab_svm, tab_lda, tab_tr = st.tabs(["Preprocessing", "SVM", "LDA", "TextRank"])

    with tab_prep:
        c1, c2, c3 = st.columns(3)
        with c1:
            p = viz("viz_prep_1_reduksi_data.png")
            if p: st.image(p, caption="Reduksi Data", use_container_width=True)
        with c2:
            p = viz("viz_prep_2_kata_kotor.png")
            if p: st.image(p, caption="Sebelum Stopword", use_container_width=True)
        with c3:
            p = viz("viz_prep_3_kata_bersih.png")
            if p: st.image(p, caption="Setelah Stemming", use_container_width=True)

    with tab_svm:
        c1, c2 = st.columns(2)
        with c1:
            p = viz("viz_svm_1_confusion_matrix.png")
            if p: st.image(p, caption="Confusion Matrix", use_container_width=True)
        with c2:
            p = viz("viz_svm_2_distribusi_akhir.png")
            if p: st.image(p, caption="Distribusi Akhir", use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            p = viz("viz_1_distribusi_sentimen.png")
            if p: st.image(p, caption="Pie Chart", use_container_width=True)
        with c4:
            p = viz("viz_3_tren_harian.png")
            if p: st.image(p, caption="Tren Temporal", use_container_width=True)

    with tab_lda:
        c1, c2 = st.columns(2)
        with c1:
            p = viz("viz_lda_1_distribusi_topik.png")
            if p: st.image(p, caption="Distribusi Klaster", use_container_width=True)
        with c2:
            p = viz("viz_lda_2_word_weights.png")
            if p: st.image(p, caption="Bobot Kata Kunci", use_container_width=True)

    with tab_tr:
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate([c1, c2, c3], 1):
            with col:
                p = viz(f"viz_textrank_topik_{i}.png")
                if p: st.image(p, caption=f"Graf Topik {i}", use_container_width=True)
        c4, c5 = st.columns(2)
        with c4:
            p = viz("viz_textrank_topik_4.png")
            if p: st.image(p, caption="Graf Topik 4", use_container_width=True)
        with c5:
            p = viz("viz_textrank_topik_5.png")
            if p: st.image(p, caption="Graf Topik 5", use_container_width=True)

# ───────────────────────────────────────────────
# PAGE: DEMO AI
# ───────────────────────────────────────────────
elif menu == "Demo":
    st.markdown("# Demo AI")
    st.markdown("Ketik komentar apapun tentang program MBG, lalu model SVM akan memprediksi sentimennya secara **real-time**.")

    vec, mdl = load_model()

    if vec is None:
        st.error("Model gagal dimuat. Pastikan file `Dataset_BGN_Labeled_10000_by_LLM.csv` ada di folder `Dataset/`.")
    else:
        st.success("Model SVM siap digunakan.")

        teks = st.text_area(
            "Masukkan komentar:",
            placeholder="Contoh: Program makan bergizi ini sangat membantu anak-anak sekolah dasar di daerah terpencil...",
            height=120
        )

        if st.button("Analisis Sentimen"):
            if teks.strip():
                pred = mdl.predict(vec.transform([teks]))[0]

                st.markdown("---")
                st.markdown("### Hasil Prediksi:")

                if pred == "Kontra":
                    st.error("**Sentimen Kontra (Negatif)**\n\nModel memprediksi komentar ini mengandung kritik, protes, atau keluhan terhadap program MBG.")
                elif pred == "Pro":
                    st.success("**Sentimen Pro (Positif)**\n\nModel memprediksi komentar ini mengandung dukungan atau apresiasi terhadap program MBG.")
                else:
                    st.info("**Sentimen Netral**\n\nModel memprediksi komentar ini bersifat informatif atau tidak memihak.")
            else:
                st.warning("Ketik komentar terlebih dahulu.")

        st.markdown("")
        st.info("**Model:** LinearSVC + TF-IDF (10.000 dimensi), dilatih pada 10.000 data berlabel. Kolom input: `teks_bersih`.")
