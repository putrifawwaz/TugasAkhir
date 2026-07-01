import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import datetime
import re


from streamlit_option_menu import option_menu

# ============================================================
# 1. KONFIGURASI HALAMAN & DATABASE LOKAL
# ============================================================
st.set_page_config(
    page_title="Sistem Deteksi Risiko Gangguan Kepribadian OMNI-IV",
    layout="wide",
)

# PATH ABSOLUT KHUSUS DIREKTORI TUGAS AKHIR PUTRI (OTOMATIS OVERWRITE)
DB_FILE = r"C:\Users\PUTRI\Documents\SEMESTER 8\TUGAS AKHIR\[WORKING]_PROJECT_TA - Copy\database_riwayat.csv"

# Fungsi untuk memuat data dari folder TA lokal
def load_database_from_csv():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_csv(DB_FILE).to_dict(orient='records')
        except:
            return []
    return []

# Fungsi untuk menyimpan/menimpa data secara presisi ke folder TA lokal
def save_database_to_csv(data_list):
    df = pd.DataFrame(data_list)
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    df.to_csv(DB_FILE, index=False)

# Inisialisasi Session State Database dari file eksternal tersebut
if 'history_db' not in st.session_state:
    st.session_state['history_db'] = load_database_from_csv()

if 'just_submitted' not in st.session_state:
    st.session_state['just_submitted'] = False

if 'active_diagnosis' not in st.session_state:
    st.session_state['active_diagnosis'] = None

# ============================================================
# 2. LOAD ASET MODEL & PREPROCESSING (ASSETS_8020)
# ============================================================
@st.cache_resource
def load_assets_8020():
    
    folder_assets = 'assets_8020'
        
    model_data = joblib.load(os.path.join(folder_assets, 'model_final_omni_8020.pkl'))
    scaler = joblib.load(os.path.join(folder_assets, 'scaler_minmax_8020.pkl'))
    features = joblib.load(os.path.join(folder_assets, 'features_selected_8020.pkl'))
    target_cols = joblib.load(os.path.join(folder_assets, 'target_columns_8020.pkl'))
    
    # Ambil model & thresholds
    if isinstance(model_data, dict):
        model = model_data['model']
        dict_thresholds = model_data.get('thresholds', {}) # Pakai dict dari file model
    else:
        model = model_data
        dict_thresholds = {col: 0.40 for col in target_cols} # Fallback jika tidak ada dict
        
    # Pembersihan Label (Menghilangkan prefix agar rapi di UI)
    labels = [col.replace('Indikasi_', '').replace('t_', '').replace('_', ' ').strip().title() for col in target_cols]
    
    return model, scaler, features, labels, target_cols, dict_thresholds

model, scaler, features, labels, target_cols, dict_thresholds = load_assets_8020()
expected_features = features

shap.plots._waterfall.color_each_ff = "#ef2266"  
shap.plots._waterfall.blue_rgb = "#456d91"       

# ============================================================
# 3. SIDEBAR: NAVIGASI 
# ============================================================
with st.sidebar:
    # Header 
    st.markdown(
        """
        <div style="padding: 10px 0px; font-family: sans-serif; color: black; text-align: left;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span style="color: black; font-weight: bold; font-size: 18px; line-height: 1.2; word-spacing: 0.5px;">
                    Prediktor Risiko Gangguan Kepribadian OMNI-IV
                </span>
            </div>
            <div style="color: black; font-family: sans-serif; font-size: 11.5px; line-height: 1.4; text-align: justify; padding-right: 5px; word-spacing: 0.5px;">
                Decision support tool berbasis multi-output machine learning & XAI (LightGBM-TreeSHAP) 
                untuk skrining dan pemetaan faktor risiko gangguan kepribadian mahasiswa.
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.divider()
    
    # Label Menu
    st.markdown("<p style='color: black; font-weight: 500; margin-bottom: 5px; font-size: 14px; font-family: sans-serif; word-spacing: 0.5px;'>Menu:</p>", unsafe_allow_html=True)
    
    # Pengaturan indeks default
    default_index = 0
    if st.session_state.get('just_submitted', False):
        default_index = 2
        st.session_state['just_submitted'] = False 

    # Menu Navigasi
    menu = option_menu(
        menu_title=None, 
        options=["ℹ️ Informasi Sistem", "🎯 Prediksi Individu", "📤 Prediksi Massal (Batch)", "📋 Riwayat Pemeriksaan"],
        default_index=default_index,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "black", "font-size": "16px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin": "5px 0px", 
                "color": "black",
                "--hover-color": "#E5E7EB",
                "font-family": "sans-serif",
                "word-spacing": "0.7px"     
            },
            "nav-link-selected": {
                "background-color": "#CFE2FE",  
                "color": "black",             
                "font-weight": "bold",
                "font-family": "sans-serif"
            },
        }
    )
    
    st.divider()
    
    # Footer
    st.markdown(
        """
        <div style="font-family: sans-serif; color: black; font-size: 12px; line-height: 1.2; word-spacing: 0.5px;">
            <div style="margin-bottom: 5px;">Tugas Akhir - Putri Nabilah Fawwaz</div>
            <div style="font-style: italic; font-size: 12px;">*Dioptimalkan Juni 2026*</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# 4. MAIN CONTENT
# ============================================================

# --- MENU 1: INFORMASI SISTEM ---
if menu == "ℹ️ Informasi Sistem":
    st.header("ℹ️ Mengenai Sistem Prediksi Gangguan Kepribadian OMNI-IV")
    
    # Catatan penting
    st.warning("""
    ***Catatan Penting**: Sistem ini merupakan **decision support tool (alat bantu pengambilan keputusan)**, bukan instrumen diagnosis utama. 
    Hasil skrining ini merupakan triase awal berdasarkan data tes OMNI pada masa orientasi. 
    **Seluruh indikasi WAJIB ditindaklanjuti oleh psikolog, konselor, atau pihak profesional yang berkompeten** sebelum intervensi final ditetapkan.*
    """)

    st.write("Sistem ini dirancang berbasis LightGBM multi-output wrapper machine learning untuk mengidentifikasi risiko gangguan kepribadian mahasiswa yang mampu menangkap pola komorbiditas. Sistem dilengkapi dengan fitur interpretabilitas model (XAI) untuk memetakan faktor risiko.")
    
    st.markdown("---")
    
    # Kolom Konfigurasi, Performa dan Distribusi Populasi
    col_main1, col_main2 = st.columns([1.2, 1])
    
    with col_main1:
        st.subheader("Konfigurasi Model")
        # Konfigurasi Model
        st.markdown("""
        * **Algoritma**: LightGBM via MultiOutputClassifier-wrapper
        * **Rasio Data**: 80% Training dan 20% Testing
        * **Threshold**: Best Threshold per label
        * **Metode XAI**: TreeSHAP Values (Gain-based Feature Importance)
        """)
        
        # Ringkasan Performa
        st.subheader("Metrik Performa")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "86.37%")
        c2.metric("Recall", "79.52%")
        c3, c4 = st.columns(2)
        c3.metric("F1-Score", "87.42%")
        c4.metric("Hamming Loss", "13.63%")
    
    with col_main2:
        st.subheader("Distribusi Populasi Uji Coba")
        labels_dist = ['Non-Klinis (Normal)', 'Indikasi Risiko Tunggal', 'Indikasi Komorbiditas']
        values_dist = [1825, 344, 318] 
        
        
        colors = ['#19548E', "#68789D", "#97D2F7"] 
        
        # Explode: 0 untuk Aman, 0.1 untuk dua kategori risiko agar terpisah
        explode = (0, 0.1, 0.2) 
        
        fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
        
        # Fungsi untuk menampilkan persentase dan angka populasi
        def func(pct, allvals): 
            absolute = int(round(pct/100.*sum(allvals)))
            return f"{pct:.1f}%\n({absolute})"

        wedges, texts, autotexts = ax_pie.pie(
            values_dist, 
            labels=None, 
            autopct=lambda pct: func(pct, values_dist), 
            colors=colors, 
            explode=explode,
            shadow=True, # Memberikan efek 3D shadow
            startangle=90,
            textprops={'color':"w", 'weight':"bold", 'fontsize': 9}
        )
        
        for patch in ax_pie.patches:
            if type(patch).__name__ == 'Shadow':
                patch.set_alpha(0.15)         
                patch.set_facecolor("#807b7b") 

        ax_pie.axis('equal')
        
        # Legenda atau nama label
        ax_pie.legend(
            wedges, labels_dist,
            title="Kategori",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        st.pyplot(fig_pie)

        st.caption(
            "*Grafik disajikan berdasarkan dataset uji coba sebanyak 2.149 baris observasi instrumen OMNI*"
        )

    st.markdown("---")
    
    # Grafik Batang Horizontal: 10 Gangguan dan Fitur Global Importance
    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        st.subheader("Prevalensi Gangguan Terdeteksi")
        data_tren = pd.DataFrame({
            'Gangguan': ['Schizoid', 'Borderline', 'Avoidant', 'Schizotypal', 'Paranoid', 
                        'Antisocial', 'Obsessive-compulsive', 'Histrionic', 'Dependent', 'Narcissistic'],
            'Jumlah_Kasus': [206, 189, 172, 161, 152, 115, 107, 96, 88, 82]
        })
        
        reds = sns.color_palette("Reds_r", 12)
        fig_bar, ax_bar = plt.subplots(figsize=(4.5, 3.8))
        bars = ax_bar.bar(data_tren['Gangguan'], data_tren['Jumlah_Kasus'], color=reds[2:])
        
        ax_bar.set_title("Distribusi Frekuensi Hasil Prediksi Gangguan Kepribadian", fontsize=7, fontweight='bold', pad=10)
        max_val = data_tren['Jumlah_Kasus'].max()
        ax_bar.set_ylim(0, max_val * 1.05) 

        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=6, fontweight='bold')
        
        ax_bar.set_ylabel("Jumlah Kasus")
        plt.xticks(rotation=30, ha='right', fontsize=6)
        
        for spine in ['top', 'right', 'left']:
            ax_bar.spines[spine].set_visible(False)
        ax_bar.axes.get_yaxis().set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig_bar)

    with col_graph2:
        st.subheader("Top Fitur Global (Model Importance)")
        feat_path = r"C:\Users\PUTRI\Documents\SEMESTER 8\TUGAS AKHIR\[WORKING]_PROJECT_TA - Copy\assets_8020\global_feature_importance.png"
        
        if os.path.exists(feat_path):
            st.image(feat_path, use_container_width=True)
        else:
            st.warning("Gambar 'global_feature_importance.png' tidak ditemukan.")


    # Interpretasi Grafik
    col_text1, col_text2 = st.columns(2)

    with col_text1:
        st.info("""
        **Interpretasi**: Gangguan *Schizoid* dan *Borderline* mendominasi temuan dalam populasi uji coba. 
        Hal ini menunjukkan perlunya atensi khusus pada aspek regulasi emosi dan perilaku sosial mahasiswa.
        """)

    with col_text2:
        if os.path.exists(feat_path):
            st.info("**Interpretasi**: Fitur *Extraversion* dan *Neuroticism* menjadi prediktor paling signifikan, mengindikasikan bahwa dimensi kepribadian dasar memiliki korelasi kuat terhadap risiko gangguan yang terdeteksi.")


# --- MENU 2: PREDIKSI INDIVIDU ---
elif menu == "🎯 Prediksi Individu":
    st.header("Prediksi Risiko Gangguan Kepribadian  Mahasiswa")
    

    if expected_features is not None:
        st.subheader("Identitas Subjek")
        c_id1, c_id2, c_id3 = st.columns(3)
        with c_id1: 
            nama_subjek = st.text_input("Nama Responden:", value="")
        with c_id2: 
            nim_subjek = st.text_input("Nomor Induk Mahasiswa (NIM):", value="")
        with c_id3: 
            pemeriksa = st.text_input("Konselor:", "Team Konseling Telkom University")

        st.subheader("Input Nilai Indikator Skor Fitur OMNI (0-100)")
        
        # Merender slider secara langsung (default ke 30)
        cols = st.columns(4)
        user_input_values = {}
        
        for i, feat in enumerate(expected_features):
            input_key = f"num_{feat}"
            with cols[i % 4]:
                user_input_values[feat] = st.number_input(
                    f" {feat}", 
                    min_value=0, 
                    max_value=100, 
                    value=None,  # Menggunakan None agar kolom wajib diketik manual oleh user
                    key=input_key,
                    step=1,
                    placeholder="0-100"
                )
                
        st.divider()
        
        if st.button("Jalankan Diagnosis", type="primary"):
            # Validasi identitas diri
            if not nama_subjek.strip() or not nim_subjek.strip():
                st.warning("⚠️ **Gagal Menjalankan Diagnosis:** Kolom 'Nama Responden' dan 'Nomor Induk Mahasiswa (NIM)' wajib diisi!")
                st.session_state['active_diagnosis'] = None
            
            # Validasi input indikator fitur tidak boleh ada yang kosong (None)
            elif any(v is None for v in user_input_values.values()):
                st.warning("⚠️ **Gagal Menjalankan Diagnosis:** Semua nilai indikator skor fitur OMNI wajib diisi angka (0-100) dan tidak boleh kosong!")
                st.session_state['active_diagnosis'] = None
                
            else:
                # Sinkronisasi nama fitur input dengan nama fitur scaler (mengganti spasi dengan underscore)
                normalized_user_inputs = {k.replace(" ", "_"): v for k, v in user_input_values.items()}
                scaler_features = list(scaler.feature_names_in_)
                
                full_input_values = {}
                for feat in scaler_features:
                    feat_clean = feat.replace(" ", "_")
                    feat_no_t = feat_clean.replace('t_', '', 1)
                    
                    if feat_clean in normalized_user_inputs:
                        full_input_values[feat] = normalized_user_inputs[feat_clean]
                    elif feat_no_t in normalized_user_inputs:
                        full_input_values[feat] = normalized_user_inputs[feat_no_t]
                    elif feat_clean.replace('_', ' ') in user_input_values:
                        full_input_values[feat] = user_input_values[feat_clean.replace('_', ' ')]
                    else:
                        full_input_values[feat] = 0

                # Buat DataFrame dengan urutan yang sudah DIKUNCI sesuai scaler
                input_df_full = pd.DataFrame([full_input_values])[scaler_features]

                # Normalisasi ulang ke skala 0-4 (sesuai skala yang digunakan saat training model)
                input_df_scaler = (input_df_full / 100.0) * 4.0 

                #  Scaling dengan MinMaxScaler yang sudah dilatih
                scaled_input = scaler.transform(input_df_scaler)
                scaled_df = pd.DataFrame(scaled_input, columns=scaler_features)

                # Sinkronasi urutan model
                if hasattr(model.estimators_[0], "feature_name_"):
                    model_features = list(model.estimators_[0].feature_name_)
                else:
                    model_features = scaler_features

                model_features_clean = [f.replace(" ", "_") for f in model_features]
                scaled_df.columns = [c.replace(" ", "_") for c in scaled_df.columns]
                
                input_model_ready = scaled_df[model_features_clean].values
                cleaned_model_features = [f.replace('t_', '', 1).replace('_', ' ') for f in model_features_clean]
                
                # PEREDIKSI MULTI-LABEL PROBABILITAS
                probabilities = model.predict_proba(input_model_ready)
                prob_values = []

                for idx, estimator in enumerate(model.estimators_):
                    proba = probabilities[idx][0]
                    classes_list = list(estimator.classes_)
                    
                    if 1 in classes_list:
                        class_index = classes_list.index(1)
                        prob_klinis = proba[class_index]
                    else:
                        prob_klinis = proba[1] if len(proba) > 1 else 0.0
                        
                    prob_values.append(prob_klinis)
                
                # Penentuan status dan severity
                detected_klinis_list = []
                results_all_list = []
                for idx, col_name in enumerate(target_cols):
                    label_clean = labels[idx]
                    prob = prob_values[idx]
                    thr_target = dict_thresholds.get(col_name, 0.40)
                    status_terdeteksi = prob >= thr_target

                    if prob >= thr_target + 0.20:
                        severity = "Risiko Tinggi"
                    elif prob >= thr_target:
                        severity = "Perlu Monitoring"
                    else:
                        severity = "Normal"
                    
                    res_item = {
                        'Gangguan': label_clean,
                        'Probabilitas': prob,
                        'Threshold': thr_target,
                        'Status': status_terdeteksi,
                        'Severity': severity
                    }
                    results_all_list.append(res_item)
                    if status_terdeteksi:
                        detected_klinis_list.append(res_item)
                
                # Simpan hasil akhir ke session state untuk visualisasi di bawahnya
                st.session_state['active_diagnosis'] = {
                    'results_all_list': results_all_list,
                    'detected_klinis_list': detected_klinis_list,
                    'input_model_ready': input_model_ready,
                    'cleaned_model_features': cleaned_model_features,
                    'normalized_user_inputs': normalized_user_inputs
                }

        if st.session_state['active_diagnosis'] is not None:
            diag = st.session_state['active_diagnosis']
            results_df = pd.DataFrame(diag['results_all_list'])
            
            # Clean nama gangguan (menghilangkan kata 'Label ')
            results_df['Gangguan'] = results_df['Gangguan'].str.replace('Label ', '', case=False)
            
            detected_klinis = pd.DataFrame(diag['detected_klinis_list']).sort_values('Probabilitas', ascending=False) if diag['detected_klinis_list'] else pd.DataFrame()
            if not detected_klinis.empty:
                detected_klinis['Gangguan'] = detected_klinis['Gangguan'].str.replace('Label ', '', case=False)

            # Baris 1: Visuaslisai bar chart gangguan dan radar chart
            st.markdown("### Profil Diagnostik Mahasiswa")
            graph_col1, graph_col2 = st.columns([1.2, 0.8]) # Mengatur rasio agar radar chart lebih kecil
            
            # --- BAR CHART DENGAN THRESHOLD  ---
            with graph_col1:
                fig_prob, ax_prob = plt.subplots(figsize=(6, 4.5))
                df_sorted_prob = results_df.sort_values('Probabilitas')
                plot_colors = ["#C43636" if r['Status'] else "#59AA7C" for _, r in df_sorted_prob.iterrows()]
                
                sns.barplot(x=df_sorted_prob['Probabilitas'] * 100, y='Gangguan', data=df_sorted_prob, palette=plot_colors, ax=ax_prob)
                
                # Garis penanda threshold dinamis
                for idx, (pandas_idx, row_data) in enumerate(df_sorted_prob.iterrows()):
                    thr_val = row_data['Threshold'] * 100
                    ax_prob.plot([thr_val, thr_val], [idx - 0.25, idx + 0.25], color='#F59E0B', ls='--', lw=1.5, alpha=0.9)
                
                ax_prob.plot([], [], color='#F59E0B', ls='--', lw=1.5, label='Ambang Batas Klinis (Kuning)')
                ax_prob.set_xlim(0, 100)
                ax_prob.set_xlabel('Probabilitas Risiko (%)', fontsize=9, fontweight='bold')
                ax_prob.set_ylabel('Label Gangguan', fontsize=9, fontweight='bold')
                ax_prob.tick_params(axis='both', which='major', labelsize=8)
                ax_prob.legend(loc='lower right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_prob)

            # --- RADAR CHART INDIVDIUAL ---
            with graph_col2:
                categories = results_df['Gangguan'].tolist()
                N = len(categories)
                
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]
                
                values = (results_df['Probabilitas'] * 100).tolist()
                values += values[:1]
                
                fig_radar = plt.figure(figsize=(4, 4))
                ax_radar = plt.subplot(111, polar=True)
                
                # Memunculkan nama gangguan di luar radar
                plt.xticks(angles[:-1], categories, color='#333333', size=8, fontweight='semibold') 
                
                # Mengatur label persentase lingkaran dalam agar tetap rapi
                ax_radar.set_rlabel_position(0)
                plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=7)
                plt.ylim(0, 100)
                
                ax_radar.plot(angles, values, color='#19548E', linewidth=1.5, linestyle='solid')
                ax_radar.fill(angles, values, color='#19548E', alpha=0.25)
                
                ax_radar.set_title("Peta Pola Spektrum Komorbiditas", size=9, weight='bold', pad=30)
                plt.tight_layout()
                st.pyplot(fig_radar)

            # Baris 2: Tabel dan card threshold
            st.write("---")
            info_col1, info_col2 = st.columns([1.1, 0.9])
            
            # --- STATUS AKHIR & PENYAJIAN TABLE ---
            with info_col1:
                st.markdown("#### Rekomendasi Klinis & Output Sistem")
                
                if not detected_klinis.empty:
                    high_risk = detected_klinis[detected_klinis['Severity'] == 'Risiko Tinggi']
                    if len(high_risk) > 0:
                        st.error("⚠️ **STATUS AKHIR: RISIKO KLINIS TINGGI**")
                    else:
                        st.warning("🟡 **STATUS AKHIR: MONITORING DIPERLUKAN**")
                    
                    # Pembuatan Dataframe untuk hasil deteksi klinis yang rapi
                    advanced_table_df = detected_klinis[['Gangguan', 'Probabilitas', 'Threshold', 'Severity']].copy()
                    
                    advanced_table_df['Tingkat Urgensi'] = advanced_table_df['Severity'].apply(
                        lambda x: f"🚨 {x}" if x == "Risiko Tinggi" else f"⚠️ {x}"
                    )
                    
                    # Format presentasi angka menjadi persen agar serasi bagi user konselor
                    advanced_table_df['Probabilitas'] = advanced_table_df['Probabilitas'].apply(lambda x: f"{x*100:.1f}%")
                    advanced_table_df['Threshold'] = advanced_table_df['Threshold'].apply(lambda x: f"{x*100:.0f}%")
                    
                    # Drop kolom 'Severity' lama karena diganti kolom dengan baru
                    advanced_table_df = advanced_table_df.drop(columns=['Severity'])
                    
                    # Merender tabel interaktif advanced bawaan Streamlit
                    st.dataframe(
                        advanced_table_df, 
                        column_config={
                            "Gangguan": "Indikasi Klinis",
                            "Probabilitas": "Probabilitas Risiko",
                            "Threshold": "Ambang Batas",
                            "Tingkat Urgensi": "Tingkat Urgensi"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.success("✅ **STATUS AKHIR: NON-KLINIS (NORMAL / AMAN)**")
                    st.write("Seluruh data uji indikasi klinis berada dalam rentang aman operasional model.")
                
            st.warning("""
            ***Catatan Penting**: Sistem ini merupakan **decision support tool (alat bantu pengambilan keputusan)**. 
            **Hasil indikasi WAJIB ditindaklanjuti oleh psikolog, konselor, atau pihak profesional yang berkompeten**.*
            """)
            
            # Implementasi card background untuk threshold dinamis per label
            with info_col2:
                st.markdown("#### Parameter Ambang Batas (Threshold)")
                
                # CSS Custom untuk mencetak efek elevasi 3D Shadow pada card putih
                st.markdown("""
                    <style>
                    .advanced-3d-card {
                        background-color: #FFFFFF;
                        padding: 12px 16px;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
                        border-left: 5px solid #19548E;
                        margin-bottom: 10px;
                    }
                    .advanced-3d-card-critical {
                        background-color: #FFFFFF;
                        padding: 12px 16px;
                        border-radius: 8px;
                        box-shadow: 0 4px 10px rgba(196, 54, 54, 0.15);
                        border-left: 5px solid #C43636;
                        margin-bottom: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                # Membagi visualisasi card ke dalam sub-kolom
                card_cols = st.columns(2)
                for idx, row in results_df.iterrows():
                    sub_col = card_cols[idx % 2]
                    
                    with sub_col:
                        # Menentukan style pembungkus html berdasarkan status urgensi klinis subjek
                        card_class = "advanced-3d-card-critical" if row['Status'] else "advanced-3d-card"
                        status_bullet = "🔴" if row['Status'] else "🟢"
                        
                        # Injeksi HTML Card Komponen
                        st.markdown(f"""
                            <div class="{card_class}">
                                <div style="font-size: 11px; font-weight: bold; color: #4B5563; margin-bottom: 2px;">
                                    {status_bullet} {row['Gangguan']}
                                </div>
                                <div style="font-size: 18px; font-weight: 800; color: #111827; line-height: 1;">
                                    {row['Threshold']*100:.0f}%
                                </div>
                                <div style="font-size: 10px; color: #6B7280; margin-top: 4px;">
                                    Output: <span style="font-weight: 600; color: {'#C43636' if row['Status'] else '#59AA7C'};">{row['Probabilitas']*100:.1f}%</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)



            st.divider()
            st.markdown("### Interpretasi Diagnosis (Personalized Explainable AI)")

            if not detected_klinis.empty:
                for _, det_row in detected_klinis.iterrows():
                    gangguan_name = det_row['Gangguan']
                    
                    st.subheader(f"Analisis Komponen Kontribusi: {gangguan_name}")
                    
                    # Sinkronasi indeks label
                    idx = None
                    for i, lbl in enumerate(labels):
                        if gangguan_name.lower() in lbl.lower():
                            idx = i
                            break
                    
                    if idx is None:
                        st.warning(f"Sistem tidak dapat menemukan pemetaan estimator untuk {gangguan_name}")
                        continue
                        
                    # Memanggil estimator dan ekstraksi SHAP values
                    sub_estimator = model.estimators_[idx]
                    explainer = shap.TreeExplainer(sub_estimator)
                    shap_values_raw = explainer.shap_values(diag['input_model_ready'])
                    
                    actual_shap = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
                    if len(actual_shap.shape) == 3: 
                        actual_shap = actual_shap[:, :, 1]
                    
                    current_shap_row = actual_shap[0]
                    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    
                    # Proses distribusi kontribusi data untuk presentase
                    df_impact = pd.DataFrame({
                        'Indikator Perilaku / Butir Soal': diag['cleaned_model_features'],
                        'Nilai Kontribusi (SHAP)': current_shap_row
                    })
                    
                    # Hitung persentase pengaruh absolut untuk interpretasi non-teknis
                    total_abs_impact = df_impact['Nilai Kontribusi (SHAP)'].abs().sum()
                    if total_abs_impact == 0: 
                        total_abs_impact = 1
                        
                    df_impact['Dampak terhadap Keputusan AI (%)'] = (df_impact['Nilai Kontribusi (SHAP)'] / total_abs_impact) * 100
                    
                    # Memisahkan pendorong dan pelindung top 4 teratas
                    pendorong_df = df_impact[df_impact['Nilai Kontribusi (SHAP)'] > 0].sort_values('Nilai Kontribusi (SHAP)', ascending=False).head(4)
                    pelindung_df = df_impact[df_impact['Nilai Kontribusi (SHAP)'] < 0].sort_values('Nilai Kontribusi (SHAP)', ascending=True).head(4)

                    # Layouting Bersandingan: Kiri Grafik, Kanan Advanced Table Panel
                    col_graph, col_table = st.columns([1.3, 1.2])

                    # --- WATERFALL PLOT ---
                    with col_graph:
                        fig_wf, ax_wf = plt.subplots(figsize=(7, 4.5))
                        plt.figure(fig_wf.number) 

                        exp_obj = shap.Explanation(
                            values=current_shap_row, 
                            base_values=base_value, 
                            data=diag['input_model_ready'][0], 
                            feature_names=diag['cleaned_model_features']
                        )

                        shap.plots.waterfall(exp_obj, max_display=8, show=False)
                        
                        plt.title(f"Arah Gradien Prediksi {gangguan_name}", fontsize=9, fontweight='bold', pad=10)
                        plt.tick_params(axis='both', which='major', labelsize=8)
                        
                        # tight_layout dipanggil aman di objek figure-nya langsung
                        fig_wf.tight_layout()
                        st.pyplot(fig_wf)
                        
                        # Tutup figur secara eksplisit untuk membersihkan memori RAM
                        plt.close(fig_wf)

                    # --- ADVANCED STYLED TABLE (RISIKO VS PROTEKTIF) ---
                    with col_table:
                        st.markdown(f"##### Pemetaan Komponen Psikometris")
                        
                        # Gabungkan data untuk visualisasi tabel 
                        composite_summary = pd.concat([pendorong_df, pelindung_df])
                        composite_summary['Tipe Komponen'] = composite_summary['Nilai Kontribusi (SHAP)'].apply(
                            lambda x: "🚨 Faktor Risiko (Pendorong)" if x > 0 else "🛡️ Faktor Protektif (Pelindung)"
                        )
                        
                        # Memperbaiki urutan kolom untuk tampilan tabel
                        display_summary_df = composite_summary[[
                            'Indikator Perilaku / Butir Soal', 
                            'Tipe Komponen', 
                            'Dampak terhadap Keputusan AI (%)'
                        ]].copy()
                        
                        # Render Advance Table dengan Pewarnaan Dinamis per Sektor
                        st.dataframe(
                            display_summary_df.style.background_gradient(
                                subset=['Dampak terhadap Keputusan AI (%)'], 
                                cmap='vlag', 
                                vmin=-50, 
                                vmax=50
                            ),
                            column_config={
                                "Dampak terhadap Keputusan AI (%)": st.column_config.NumberColumn(
                                    "Dampak Risiko (%)",
                                    format="%.1f%%"
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                       
            else:
                st.success("✅ **Hasil Evaluasi Bersih**: Tidak ditemukan indikasi risiko klinis komorbid yang melampaui batas ambang sistem. Analisis mendalam SHAP dinonaktifkan.")
            
                    
            # --- LEMBAR REKOMENDASI KLINIS ---
            st.divider()
            st.subheader("Rekomendasi dan Catatan Konselor")

            with st.form("form_psikolog"):
                catatan_intervensi = st.text_area(
                    "Tuliskan analisis kualitatif, intervensi medis, atau rencana konseling lanjutan:",
                    placeholder="Contoh: Terdeteksi komorbiditas risiko tinggi. Rencana rujukan bimbingan kognitif..."
                )
                
                jumlah_gangguan = len(diag['detected_klinis_list'])
                if jumlah_gangguan == 0:
                    tingkat_risiko = "Normal (Non-Klinis)"
                elif jumlah_gangguan <= 2:
                    tingkat_risiko = "Waspada Ringan (1-2 Gangguan)"
                else:
                    tingkat_risiko = "Risiko Tinggi Klinis (>2 Gangguan Komorbid)"
                
                nama_gangguan_terdeteksi = ", ".join([g['Gangguan'] for g in diag['detected_klinis_list']]) if diag['detected_klinis_list'] else "Normal"
                
                submitted = st.form_submit_button("Simpan Hasil Pemeriksaan", type="primary")
                if submitted:
                    list_pendorong_mhs = []
                    list_pelindung_mhs = []
                    
                    if not detected_klinis.empty:
                        for _, det_row in detected_klinis.iterrows():
                            gangguan_name = det_row['Gangguan']
                            idx_est = next((i for i, lbl in enumerate(labels) if gangguan_name.lower() in lbl.lower()), None)
                            
                            if idx_est is not None:
                                try:
                                    sub_estimator = model.estimators_[idx_est]
                                    explainer = shap.TreeExplainer(sub_estimator)
                                    single_row_input = diag['input_model_ready'][0:1]
                                    shap_values_raw = explainer.shap_values(single_row_input)
                                    
                                    if isinstance(shap_values_raw, list):
                                        actual_shap = shap_values_raw if len(shap_values_raw) > 1 else shap_values_raw
                                    elif hasattr(shap_values_raw, "values"):
                                        actual_shap = shap_values_raw.values
                                    else:
                                        actual_shap = shap_values_raw
                                        
                                    if len(actual_shap.shape) == 3: 
                                        actual_shap = actual_shap[:, :, 1]
                                        
                                    flat_shap_row = np.array(actual_shap).flatten()
                                    
                                    df_impact_tmp = pd.DataFrame({
                                        'Fitur': diag['cleaned_model_features'],
                                        'SHAP': flat_shap_row
                                    })
                                    
                                    top_p = df_impact_tmp[df_impact_tmp['SHAP'] > 0].sort_values('SHAP', ascending=False)['Fitur'].tolist()
                                    top_l = df_impact_tmp[df_impact_tmp['SHAP'] < 0].sort_values('SHAP', ascending=True)['Fitur'].tolist()
                                    
                                    for f in top_p:
                                        if f not in list_pendorong_mhs: list_pendorong_mhs.append(f)
                                    for f in top_l:
                                        if f not in list_pelindung_mhs: list_pelindung_mhs.append(f)
                                except:
                                    pass

                    if not list_pendorong_mhs or not list_pelindung_mhs:
                        try:
                            for f, v in diag['normalized_user_inputs'].items():
                                f_clean = str(f).replace('t_', '', 1).replace('_', ' ')
                                if v >= 40:
                                    if f_clean not in list_pendorong_mhs: list_pendorong_mhs.append(f_clean)
                                else:
                                    if f_clean not in list_pelindung_mhs: list_pelindung_mhs.append(f_clean)
                        except:
                            list_pendorong_mhs = ["Indikator Skor Tinggi"]
                            list_pelindung_mhs = ["Indikator Skor Rendah"]

                    # 🛠️ REVISI STATUS KATEGORI BARU UNTUK INDIVIDU
                    if not detected_klinis.empty:
                        string_diagnosa = ", ".join(detected_klinis['Gangguan'].tolist())
                        # Jika terdeteksi lebih dari 1 gangguan kepribadian, dikelompokkan sebagai Komorbiditas
                        if len(detected_klinis) > 1:
                            string_status = "Indikasi Risiko Tinggi (Komorbiditas)"
                        else:
                            string_status = "Indikasi Risiko Tunggal"
                    else:
                        string_diagnosa = "Non-Klinis (Normal)"
                        string_status = "Non-Klinis (Normal)"

                    string_catatan = catatan_intervensi.strip() if catatan_intervensi.strip() else "Tidak ada catatan konselor."

                    log_pemeriksaan_individu = {
                        "Tanggal": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Nama Responden": nama_subjek,
                        "NIM": nim_subjek,
                        "Faktor Tingkat Risiko": string_status,
                        "Hasil Diagnosa Klinis": string_diagnosa,
                        "Faktor Risiko": ", ".join(list_pendorong_mhs) if list_pendorong_mhs else "Tidak ada",
                        "Faktor Protektif": ", ".join(list_pelindung_mhs) if list_pelindung_mhs else "Tidak ada",
                        "Catatan Klinis": string_catatan,
                        "Konselor": pemeriksa
                    }
                    
                    st.session_state['history_db'].append(log_pemeriksaan_individu)
                    save_database_to_csv(st.session_state['history_db'])
                    st.success("✅ Seluruh faktor risiko, faktor protektif, dan catatan konselor berhasil direkam ke Database Log!")
                    st.rerun()


# --- MENU 3: BATCH PROCESSING --- #
elif menu == "📤 Prediksi Massal (Batch)":
    st.header("Prediksi Massal Risiko Gangguan Kepribadian Mahasiswa")
    st.write("Unggah file Excel berisi data indikator OMNI mahasiswa untuk melakukan diagnosis sekaligus.")

    # --- GENERATOR TEMPLATE EXCEL --- 
    kolom_identitas = ["Nama Responden", "NIM", "Konselor"]
    kolom_template = kolom_identitas + expected_features
    
    #  Contoh data dummy 1 baris sebagai panduan pengisian
    data_panduan = {col: (30 if col in expected_features else "") for col in kolom_template}
    data_panduan["Nama Responden"] = "Nama"
    data_panduan["NIM"] = "120xxxxxxx"
    data_panduan["Konselor"] = "Team Konseling Telkom University"
    
    # Konversi menjadi DataFrame pandas
    df_template = pd.DataFrame([data_panduan])
    
    # Bungkus ke dalam objek BytesIO agar bisa diunduh langsung dari RAM tanpa save file lokal
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Template OMNI')
    
    download_bytes = buffer.getvalue()
    
    # 5. Tampilkan tombol download template resmi
    st.download_button(
        label="📥 Unduh Template Excel (.xlsx)",
        data=download_bytes,
        file_name="Template_Skrining_Massal_OMNI.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Gunakan template ini agar nama kolom dan urutan fitur 100% selaras dengan model."
    )
    st.write("")
    
    # --- PROCEESSING FILE --- 
    uploaded_file = st.file_uploader("Pilih File Excel Hasil Pengisian (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_excel(uploaded_file)
            st.success(f"Berhasil memuat data: {len(df_upload)} baris ditemukan.")
            
            with st.expander("Pratinjau Data Unggahan"):
                st.dataframe(df_upload, use_container_width=True)

            required_cols = ["Nama Responden", "NIM"]
            missing_cols = [col for col in required_cols if col not in df_upload.columns]

            if missing_cols:
                st.error(f"❌ **Gagal Memproses:** File Excel harus memiliki kolom identitas berikut: {missing_cols}")
            elif expected_features is None:
                st.error("❌ **Gagal Memproses:** Model atau fitur ekspektasi (`expected_features`) tidak terdefinisi.")
            else:
                st.divider()
                
                if st.button("Jalankan Diagnosis Massal", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    batch_results = []
                    total_rows = len(df_upload)

                    for index, row in df_upload.iterrows():
                        progress_bar.progress((index + 1) / total_rows)
                        status_text.text(f"Memproses {index + 1} dari {total_rows} mahasiswa...")

                        nama_subjek = str(row.get("Nama Responden", f"Mhs_{index+1}"))
                        nim_subjek = str(row.get("NIM", f"NIM_{index+1}"))
                        konselor_input = str(row.get("Konselor", "Team Konseling Telkom University"))

                        # PENCETAKAN DICTIONARY FRESH UNTUK MAHASISWA AKTIF (AMBIL DATA EXCEL)
                        current_mhs_features = {}

                        excel_cols_clean = {}
                        for col in row.index:
                            c_clean = str(col).strip().lower().replace(" ", "").replace("_", "")
                            if c_clean.startswith("t") and len(c_clean) > 1:
                                c_clean = c_clean[1:]
                            excel_cols_clean[c_clean] = col
                        
                        for feat in expected_features:
                            # Buat mapping nama fitur ekspektasi versi super bersih (lowercase, hapus spasi/underscore, HAPUS awalan 't')
                            feat_lookup = str(feat).strip().lower().replace(" ", "").replace("_", "")
                            if feat_lookup.startswith("t") and len(feat_lookup) > 1:
                                feat_lookup = feat_lookup[1:] # Kupas huruf 't' di awal fitur program jika ada
                            
                            # Jalankan pencocokan data kuantitatif
                            if feat_lookup in excel_cols_clean:
                                actual_excel_col = excel_cols_clean[feat_lookup]
                                val = row.get(actual_excel_col, 30)
                            else:
                                val = 30 # Baseline cadangan jika kolom benar-benar tidak ditemukan
                                
                            current_mhs_features[feat] = 30 if pd.isna(val) else int(val)

                        normalized_user_inputs = {k.replace(" ", "_"): v for k, v in current_mhs_features.items()}
                        scaler_features = list(scaler.feature_names_in_)
                        
                        full_input_values = {}
                        for feat in scaler_features:
                            feat_clean = feat.replace(" ", "_")
                            feat_no_t = feat_clean.replace('t_', '', 1)
                            
                            if feat_clean in normalized_user_inputs:
                                full_input_values[feat] = normalized_user_inputs[feat_clean]
                            elif feat_no_t in normalized_user_inputs:
                                full_input_values[feat] = normalized_user_inputs[feat_no_t]
                            elif feat_clean.replace('_', ' ') in current_mhs_features:
                                full_input_values[feat] = current_mhs_features[feat_clean.replace('_', ' ')]
                            else:
                                full_input_values[feat] = 0

                        # DataFrame terisolasi murni
                        input_df_full = pd.DataFrame([full_input_values])[scaler_features].copy()

                        # Sesuai dengan standar operasional MinMaxScaler bawaan model 
                        input_df_scaler = input_df_full / 100.0

                        # Lakukan transformasi data secara aman
                        scaled_input = scaler.transform(input_df_scaler)
                        scaled_df = pd.DataFrame(scaled_input, columns=scaler_features)

                        if hasattr(model.estimators_, "feature_name_"):
                            model_features = list(model.estimators_.feature_name_)
                        else:
                            model_features = scaler_features

                        model_features_clean = [f.replace(" ", "_") for f in model_features]
                        scaled_df.columns = [c.replace(" ", "_") for c in scaled_df.columns]
                        
                        input_model_ready = scaled_df[model_features_clean].values
                        cleaned_model_features = [f.replace('t_', '', 1).replace('_', ' ') for f in model_features_clean]

                        # PENILAIAN PROBABILITAS MULTI-LABEL 
                        probabilities = model.predict_proba(input_model_ready)
                        prob_values = []
                        
                        try:
                            # Menggunakan predict_proba standar
                            probabilities = model.predict_proba(input_model_ready)
                            
                            # Cek apakah output berupa array NumPy tunggal (XGBoost Native Multi-Output)
                            if isinstance(probabilities, np.ndarray) and len(probabilities.shape) == 3:
                                # Struktur array: (sampel, kelas, label) -> ambil sampel ke-0
                                for idx in range(len(model.estimators_)):
                                    # Mengambil probabilitas kelas 1 untuk label ke-idx
                                    prob_values.append(float(probabilities[0, 1, idx]))
                            else:
                                # Jika berupa list of arrays (Sklearn MultiOutputClassifier standar)
                                for idx, estimator in enumerate(model.estimators_):
                                    proba_output = probabilities[idx]
                                    classes_list = list(estimator.classes_)
                                    proba_flat = np.array(proba_output).flatten()
                                    
                                    if len(proba_flat) > 1:
                                        if 1 in classes_list:
                                            class_index = classes_list.index(1)
                                            prob_values.append(float(proba_flat[class_index]))
                                        else:
                                            prob_values.append(float(proba_flat[-1]))
                                    else:
                                        prob_values.append(float(proba_flat[0]))
                        except Exception as prob_err:
                            # FALLBACK AMAN: Jika predict_proba gagal total karena struktur dimensi,
                            # model.predict() untuk mendapatkan nilai biner (0 atau 1) langsung
                            try:
                                raw_predictions = model.predict(input_model_ready).flatten()
                                for val in raw_predictions:
                                    prob_values.append(1.0 if val == 1 else 0.0)
                            except:
                                # Jalankan emergency fallback dari nilai skor input mentah excel jika model menolak input
                                for feat in target_cols:
                                    prob_values.append(0.0)

                        # Threshold adjustment
                        if 'dict_thresholds' in globals() or 'dict_thresholds' in locals():
                            active_thresholds = dict_thresholds
                        else:
                            try:
                                active_thresholds = model_data.get('thresholds', {col: 0.40 for col in target_cols})
                            except:
                                active_thresholds = {col: 0.40 for col in target_cols}

                        # Evaluasi Batas Threshold Klinis Mahasiswa Aktif
                        detected_batch_klinis = []
                        highest_prob_idx = 0
                        max_prob_val = -1

                        for idx, col_name in enumerate(target_cols):
                            # Jika prob_values kosong akibat kegagalan ekstrim, isi dengan 0.0 agar tidak out of range
                            prob = prob_values[idx] if idx < len(prob_values) else 0.0
                            thr_target = active_thresholds.get(col_name, 0.40) 
                            
                            if prob >= thr_target:
                                label_bersih = labels[idx].replace('Label ', '')
                                # Cetak persentase probabilitas riil di samping gangguan untuk pembuktian data dinamis
                                detected_batch_klinis.append(f"{label_bersih} ({prob*100:.1f}%)")
                                
                            if prob >= thr_target:
                                label_bersih = labels[idx].replace('Label ', '')
                                detected_batch_klinis.append(f"{label_bersih} ({prob*100:.1f}%)")
                            if prob > max_prob_val:
                                max_prob_val = prob
                                highest_prob_idx = idx

                        # SHAP Interpretability
                        list_pendorong_mhs = []
                        list_pelindung_mhs = []
                        
                        try:
                            estimators_to_check = []
                            if detected_batch_klinis:
                                # Jika ada gangguan terdeteksi, iterasi semua klaster gangguan untuk ambil semua faktornya
                                for g_encoded in detected_batch_klinis:
                                    g_name = g_encoded.split(" (")[0]
                                    idx_e = next((i for i, lbl in enumerate(labels) if g_name.lower() in lbl.lower()), None)
                                    if idx_e is not None:
                                        estimators_to_check.append(idx_e)
                            else:
                                estimators_to_check.append(highest_prob_idx)

                            for idx_est in estimators_to_check:
                                sub_estimator = model.estimators_[idx_est]
                                explainer = shap.TreeExplainer(sub_estimator)
                                shap_values_raw = explainer.shap_values(input_model_ready.copy())
                                
                                if isinstance(shap_values_raw, list):
                                    actual_shap = shap_values_raw[-1] if len(shap_values_raw) > 1 else shap_values_raw
                                elif hasattr(shap_values_raw, "values"):
                                    actual_shap = shap_values_raw.values
                                else:
                                    actual_shap = shap_values_raw
                                    
                                if len(actual_shap.shape) == 3: 
                                    actual_shap = actual_shap[:, :, 1]
                                    
                                flat_shap_row = np.array(actual_shap).flatten()
                                
                                df_impact_b = pd.DataFrame({
                                    'Fitur': cleaned_model_features,
                                    'SHAP': flat_shap_row
                                })
                                
                                # Mengambil SELURUH fitur pendorong (>0) dan pelindung (<0) tanpa batasan .head()
                                top_p = df_impact_b[df_impact_b['SHAP'] > 0].sort_values('SHAP', ascending=False)['Fitur'].tolist()
                                top_l = df_impact_b[df_impact_b['SHAP'] < 0].sort_values('SHAP', ascending=True)['Fitur'].tolist()
                                
                                for f in top_p:
                                    if f not in list_pendorong_mhs: list_pendorong_mhs.append(f)
                                for f in top_l:
                                    if f not in list_pelindung_mhs: list_pelindung_mhs.append(f)
                        except:
                            sorted_features_by_val = sorted(current_mhs_features.items(), key=lambda item: item[1], reverse=True)
                            list_pendorong_mhs = [str(f[0]).replace('t_', '', 1).replace('_', ' ') for f in sorted_features_by_val if f[1] >= 45]
                            list_pelindung_mhs = [str(f[0]).replace('t_', '', 1).replace('_', ' ') for f in sorted_features_by_val if f[1] < 45]

                        # Seluruh data lengkap digabungkan menjadi string terpisah koma
                        risiko_fitur = ", ".join(list_pendorong_mhs) if list_pendorong_mhs else "Tidak ada"
                        protektif_fitur = ", ".join(list_pelindung_mhs) if list_pelindung_mhs else "Tidak ada"

                        if detected_batch_klinis:
                            string_diagnosa = ", ".join(detected_batch_klinis)
                            if len(detected_batch_klinis) > 1:
                                string_status = "Indikasi Risiko Tinggi (Komorbiditas)"
                            else:
                                string_status = "Indikasi Risiko Tunggal"
                        else:
                            string_diagnosa = "Non-Klinis (Normal)"
                            string_status = "Non-Klinis (Normal)"

                        # --- STRUKTUR ENTRI REKAM MEDIS ---
                        log_entry = {
                            "Tanggal": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Nama Responden": nama_subjek,
                            "NIM": nim_subjek,
                            "Faktor Tingkat Risiko": string_status,
                            "Hasil Diagnosa Klinis": string_diagnosa,
                            "Faktor Risiko": risiko_fitur,
                            "Faktor Protektif": protektif_fitur,
                            "Catatan Klinis": "Hasil pemrosesan massal.",
                            "Konselor": konselor_input
                        }
                        batch_results.append(log_entry)

                    # Kunci penyimpanan gabungan setelah loop 15 mahasiswa selesai sempurna
                    st.session_state['history_db'].extend(batch_results)
                    save_database_to_csv(st.session_state['history_db'])

                    status_text.text("✅ Diagnosis Massal Selesai & Database Diperbarui!")
                    st.session_state['current_batch_results'] = batch_results
                    st.rerun()

            # --- OUTPUT & METRIK SUMMARY POPULASI BATCH REAL DATA ---
            if 'current_batch_results' in st.session_state and st.session_state['current_batch_results']:
                st.divider()
                st.subheader("📋 Tabel Log Hasil Pemeriksaan Terkini (Batch)")
                
                df_batch_res = pd.DataFrame(st.session_state['current_batch_results'])
                
                def hitung_komorbid_batch(teks):
                    if not teks or pd.isna(teks) or "Non-Klinis" in str(teks) or "Normal" in str(teks):
                        return 0
                    return len([x.strip() for x in str(teks).split(",") if x.strip()])

                def style_batch(row):
                    styles = [''] * len(row)
                    
                    # Dapatkan indeks posisi koordinat kolom secara dinamis
                    idx_tingkat_risiko = row.index.get_loc("Faktor Tingkat Risiko")
                    idx_faktor_risiko = row.index.get_loc("Faktor Risiko")
                    idx_faktor_protektif = row.index.get_loc("Faktor Protektif")
                    
                    status_str = str(row['Faktor Tingkat Risiko']).lower()
                    
                    if "komorbiditas" in status_str:
                        color_style = 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
                        styles[idx_tingkat_risiko] = color_style
                        styles[idx_faktor_risiko] = color_style
                        styles[idx_faktor_protektif] = color_style
                    elif "tunggal" in status_str:
                        color_style = 'background-color: #FEF3C7; color: #92400E;'
                        styles[idx_tingkat_risiko] = color_style
                        styles[idx_faktor_risiko] = color_style
                        styles[idx_faktor_protektif] = color_style
                        
                    return styles

                st.dataframe(df_batch_res.style.apply(style_batch, axis=1), use_container_width=True)

                st.divider()
                st.subheader("📊 Summary & Prevalensi Populasi (Khusus Batch Ini)")
                
                df_batch_res['Jumlah_Gangguan'] = df_batch_res['Hasil Diagnosa Klinis'].apply(hitung_komorbid_batch)
                df_batch_res['Kategori_Makro'] = df_batch_res['Jumlah_Gangguan'].apply(
                    lambda n: 'Non-Klinis (Normal)' if n == 0 else ('Indikasi Risiko Tunggal' if n == 1 else 'Indikasi Risiko Tinggi (Komorbiditas)')
                )
                
                m_normal = int(sum(df_batch_res['Kategori_Makro'] == 'Non-Klinis (Normal)'))
                m_tunggal = int(sum(df_batch_res['Kategori_Makro'] == 'Indikasi Risiko Tunggal'))
                m_komorbid = int(sum(df_batch_res['Kategori_Makro'] == 'Indikasi Risiko Tinggi (Komorbiditas)'))
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Diperiksa", f"{len(df_batch_res)} Mhs")
                c2.metric("Non-Klinis (Normal)", f"{m_normal} Mhs")
                c3.metric("Risiko Tunggal (Kuning)", f"{m_tunggal} Mhs")
                c4.metric("Komorbiditas (Merah)", f"{m_komorbid} Mhs")

                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("##### Proporsi Kondisi Mental")
                    fig_p, ax_p = plt.subplots(figsize=(5, 3.5))
                    ax_p.pie(
                        [m_normal, m_tunggal, m_komorbid],
                        labels=['Normal', 'Tunggal', 'Komorbid'],
                        autopct='%1.1f%%',
                        colors=['#59AA7C', '#F59E0B', '#C43636'],
                        startangle=90
                    )
                    ax_p.axis('equal')
                    plt.tight_layout()
                    st.pyplot(fig_p)

                with g2:
                    st.markdown("##### Densitas Indikasi Gangguan")
                    fig_k, ax_k = plt.subplots(figsize=(5, 3.5))
                    x_axis = list(range(4))
                    y_axis = [int(sum(df_batch_res['Jumlah_Gangguan'] == idx)) for idx in x_axis]
                    
                    ax_k.bar(x_axis, y_axis, color='#93A6D2', edgecolor='grey', linewidth=0.5)
                    ax_k.set_xlabel("Jumlah Gangguan Terdeteksi")
                    ax_k.set_ylabel("Jumlah Mahasiswa")
                    ax_k.set_xticks(x_axis)
                    ax_k.set_xticklabels(['0', '1', '2', '3+'])
                    
                    for i, val in enumerate(y_axis):
                        if val > 0:
                            ax_k.text(i, val + 0.1, str(val), ha='center', va='bottom', fontsize=8, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_k)

                # --- BARIS GRAFIK 2: GLOBAL TOP FITUR PENDORONG VS PROTEKTIF ---
                st.write("")
                g3, g4 = st.columns(2)
                
                with g3:
                    st.markdown("##### Top Faktor Risiko Utama Global (Pendorong)")
                    all_risks = []
                    for item in df_batch_res['Faktor Risiko'].dropna().tolist():
                        all_risks.extend([x.strip() for x in str(item).split(",") if x.strip() and x.strip() != "Tidak ada"])
                        
                    if all_risks:
                        df_top_risks = pd.Series(all_risks).value_counts().head(5).reset_index()
                        df_top_risks.columns = ['Fitur OMNI', 'Frekuensi Kasus']
                        df_top_risks = df_top_risks.sort_values('Frekuensi Kasus', ascending=True)
                        
                        fig_r, ax_r = plt.subplots(figsize=(5, 3.5))
                        sns.barplot(x='Frekuensi Kasus', y='Fitur OMNI', data=df_top_risks, color='#C43636', ax=ax_r)
                        
                        for bar in ax_r.patches:
                            width = bar.get_width()
                            if width > 0:
                                ax_r.annotate(f' {int(width)} mhs', xy=(width, bar.get_y() + bar.get_height() / 2), ha='left', va='center', fontsize=8, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_r)
                    else:
                        st.info("Seluruh data bersih.")

                with g4:
                    st.markdown("##### Top Faktor Protektif Utama Global (Pelindung)")
                    all_protects = []
                    for item in df_batch_res['Faktor Protektif'].dropna().tolist():
                        all_protects.extend([x.strip() for x in str(item).split(",") if x.strip() and x.strip() != "Tidak ada"])
                        
                    if all_protects:
                        df_top_protects = pd.Series(all_protects).value_counts().head(5).reset_index()
                        df_top_protects.columns = ['Fitur OMNI', 'Frekuensi Kasus']
                        df_top_protects = df_top_protects.sort_values('Frekuensi Kasus', ascending=True)
                        
                        fig_l, ax_l = plt.subplots(figsize=(5, 3.5))
                        sns.barplot(x='Frekuensi Kasus', y='Fitur OMNI', data=df_top_protects, color='#59AA7C', ax=ax_l)
                        
                        for bar in ax_l.patches:
                            width = bar.get_width()
                            if width > 0:
                                ax_l.annotate(f' {int(width)} mhs', xy=(width, bar.get_y() + bar.get_height() / 2), ha='left', va='center', fontsize=8, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig_l)
                    else:
                        st.info("Tidak ada faktor protektif.")
                        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")

                        
# --- MENU 4: RIWAYAT PEMERIKSAAN (CRUD PERSISTEN) ---
elif menu == "📋 Riwayat Pemeriksaan":
    st.header("Database Log & Analisis Pemeriksaan Mahasiswa")
    
    # Tab untuk memisahkan database riwayat dan visualiasi untuk analisis global
    tab_data, tab_analytics = st.tabs(["🗂️ Database Log Pemeriksaan", "📊 Analisis Global"])
    
    # --- DATABASE LOG PEMERIKSAAN ---
    with tab_data:
        st.write("Daftar catatan rekam medis. Kolom tingkat risiko otomatis berwarna:")
        
        if st.session_state['history_db']:
            df_history = pd.DataFrame(st.session_state['history_db'])

            kolom_rapi = [
                "Tanggal", "Nama Responden", "NIM", 
                "Faktor Tingkat Risiko", "Hasil Diagnosa Klinis", 
                "Faktor Risiko", "Faktor Protektif", "Catatan Klinis", "Konselor"
            ]

            # Mengamankan nilai default jika ada sel data yang kosong
            for col in kolom_rapi:
                if col not in df_history.columns:
                    df_history[col] = "Tidak ada data"

            df_history = df_history[kolom_rapi]
            
            # --- TAMPILKAN MODE EDIT ATAU MODE VIEW BERWARNA ---
            mode_edit = st.checkbox("✍️ Aktifkan Mode Edit Tabel")

            if mode_edit:
                st.info("Mode edit data")
                # Menggunakan data polosan agar streamlit data_editor tidak crash/bug
                edited_df = st.data_editor(df_history, use_container_width=True, num_rows="fixed")
                
                if not pd.DataFrame(edited_df).equals(df_history):
                    st.session_state['history_db'] = pd.DataFrame(edited_df).to_dict(orient='records')
                    save_database_to_csv(st.session_state['history_db']) 
                    st.success("Perubahan database log berhasil diperbarui secara permanen!")
                    st.rerun()
            else:
                def style_row_by_risk(row):
                    styles = [''] * len(row)
                    idx_kolom_risiko = row.index.get_loc("Faktor Tingkat Risiko")
                    status_risiko = str(row['Faktor Tingkat Risiko']).strip().lower()
                    
                    # JIKA STATUS ADALAH KOMORBIDITAS -> WARNA MERAH LEMBUT
                    if "komorbiditas" in status_risiko:
                        styles[idx_kolom_risiko] = 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
                    
                    # JIKA STATUS ADALAH RISIKO TUNGGAL -> WARNA KUNING LEMBUT
                    elif "tunggal" in status_risiko:
                        styles[idx_kolom_risiko] = 'background-color: #FEF3C7; color: #92400E; font-weight: bold;'
                        
                    return styles

                # Menerapkan style warna ke dalam dataframe sebelum dirender
                styled_df = df_history.style.apply(style_row_by_risk, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                
            st.divider()
            st.subheader("Hapus Rekam Medis")
            nama_hapus_list = df_history["Nama Responden"].tolist()
            opsi_hapus = st.selectbox("Pilih nama responden mahasiswa yang ingin dihapus:", ["-- Pilih Mahasiswa --"] + nama_hapus_list)
            
            if st.button("Hapus Data Mahasiswa Terpilih", type="secondary"):
                if opsi_hapus != "-- Pilih Mahasiswa --":
                    updated_list = [row for row in st.session_state['history_db'] if row["Nama Responden"] != opsi_hapus]
                    st.session_state['history_db'] = updated_list
                    save_database_to_csv(updated_list) 
                    st.success(f"Sukses menghapus rekam medis atas nama {opsi_hapus} dari folder TA.")
                    st.rerun()
                else:
                    st.warning("Silakan tentukan nama responden mahasiswa terlebih dahulu sebelum menekan tombol hapus.")
                    
        else:
            st.info("Belum ada rekam data riwayat pemeriksaan mahasiswa yang dimasukkan dalam sesi ini.")


    # --- ANALISIS & PREVALENSI GLOBAL (GRID 2x2 DENGAN INTERPRETASI DINAMIS) ---
    with tab_analytics:
        if st.session_state['history_db']:
            df_analitik = pd.DataFrame(st.session_state['history_db'])
            total_log = len(df_analitik)
            
            st.markdown(f"### Statistik Prevalensi Populasi (Total: {total_log} Pemeriksaan)")
            
            # Fungsi pembantu untuk memisahkan teks multi-label komorbid secara dinamis
            def hitung_komorbid(teks):
                if not teks or pd.isna(teks) or "Non-Klinis" in str(teks) or "Normal" in str(teks):
                    return 0
                bersih = str(teks).replace("Dominan ", "")
                return len([x.strip() for x in bersih.split(",") if x.strip()])
            
            df_analitik['Jumlah_Gangguan'] = df_analitik['Hasil Diagnosa Klinis'].apply(hitung_komorbid)
            
            # Pengelompokan makro untuk grafik lingkaran
            def kategori_makro(n):
                if n == 0: return 'Non-Klinis (Normal)'
                elif n == 1: return 'Indikasi Risiko Tunggal'
                else: return 'Indikasi Komorbiditas'
                
            df_analitik['Kategori_Makro'] = df_analitik['Jumlah_Gangguan'].apply(kategori_makro)
            
            # Pembuatan struktur grid tata letak 2 Baris x 2 Kolom
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)
            
            # --- PIE CHART 3D DISTRIBUSI POPULASI 
            with row1_col1:
                st.markdown("##### Proporsi Kondisi Mental Global")
                dist_makro = df_analitik['Kategori_Makro'].value_counts()
                
                labels_pie = ['Non-Klinis (Normal)', 'Indikasi Risiko Tunggal', 'Indikasi Komorbiditas']
                values_pie = [dist_makro.get(l, 0) for l in labels_pie]
                
                fig_p, ax_p = plt.subplots(figsize=(5, 4))
                colors_p = ['#19548E', '#93A6D2', '#B6DCF4']
                explode_p = (0, 0.1, 0.1)
                
                def fmt_p(pct, allvals):
                    absolute = int(round(pct/100.*sum(allvals)))
                    return f"{pct:.1f}%\n({absolute} mhs)" if absolute > 0 else ""
                
                wedges, texts, autotexts = ax_p.pie(
                    values_pie, labels=None, autopct=lambda pct: fmt_p(pct, values_pie),
                    colors=colors_p, explode=explode_p, shadow=True, startangle=90,
                    textprops={'color':"w", 'weight':"bold", 'fontsize': 8}
                )
                
                # Modifikasi bayangan agar terlihat soft (estetika 3D lanjut)
                for patch in ax_p.patches:
                    if type(patch).__name__ == 'Shadow':
                        patch.set_alpha(0.15)          
                        patch.set_facecolor("#807b7b")

                ax_p.axis('equal')
                ax_p.legend(wedges, labels_pie, title="Kategori", loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=7, title_fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_p)
                
                # Insight Dinamis Grid 1
                mhs_komorbid = dist_makro.get('Indikasi Komorbiditas', 0)
                pct_komorbid = (mhs_komorbid / total_log) * 100 if total_log > 0 else 0
                st.caption(f"*Insight*: Sebanyak **{pct_komorbid:.1f}%** ({mhs_komorbid} mahasiswa) berada dalam status komorbiditas klinis yang membutuhkan atensi prioritas utama konselor.")

            # --- BAR CHART DISTRIBUSI JUMLAH INDIKASI (DENSITAS KOMORBID) ---
            with row1_col2:
                st.markdown("##### Densitas Komorbiditas per Individu")
                
                counts_komorbid = df_analitik['Jumlah_Gangguan'].value_counts()
                x_axis_k = list(range(11))
                y_axis_k = [counts_komorbid.get(i, 0) for i in x_axis_k]
                
                fig_k, ax_k = plt.subplots(figsize=(5, 4))
                colors_k = sns.color_palette("magma", n_colors=11)
                
                bars_k = ax_k.bar(x_axis_k, y_axis_k, color=colors_k, width=0.75, edgecolor='grey', linewidth=0.3)
                
                for bar in bars_k:
                    height = bar.get_height()
                    if height > 0:
                        ax_k.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7, fontweight='bold')
                
                ax_k.set_xlabel("Jumlah Indikasi Gangguan", fontsize=7, fontweight='bold')
                ax_k.set_ylabel("Jumlah Individu", fontsize=7, fontweight='bold')
                ax_k.set_xticks(x_axis_k)
                ax_k.tick_params(axis='both', labelsize=7)
                ax_k.set_ylim(0, max(y_axis_k) * 1.15 if max(y_axis_k) > 0 else 10)
                
                for spine in ['top', 'right']:
                    ax_k.spines[spine].set_visible(False)
                ax_k.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_k)
                
                # Insight Dinamis Grid 2
                max_komorbid_id = df_analitik['Jumlah_Gangguan'].max()
                st.caption(f"*Insight*: Spektrum ketidakseimbangan data terdeteksi berat; nilai ekstrem tertinggi mencatat individu dengan **{max_komorbid_id} kluster gangguan simultan**.")

            # --- BAR CHART FREKUENSI GANGGUAN TERDETEKSI  ---
            with row2_col1:
                st.markdown("##### Prevalensi Gangguan Kepribadian Terdeteksi")
                
                # Ekstraksi frekuensi teks label dari database log
                all_detected_traits = []
                for idx, r in df_analitik.iterrows():
                    teks = str(r['Hasil Diagnosa Klinis']).replace("Dominan ", "")
                    if teks and "Non-Klinis" not in teks and "Normal" not in teks:
                        # Menggunakan regex untuk menghapus teks di dalam kurung seperti "(77.5%)"
                        raw_list = [x.strip() for x in teks.split(",") if x.strip()]
                        for item in raw_list:
                            clean_trait = re.sub(r'\s*\([^)]*\)', '', item) # Menghapus "(...)"
                            all_detected_traits.append(clean_trait)
                
                if all_detected_traits:
                    from collections import Counter
                    counts_traits = Counter(all_detected_traits)
                    df_traits = pd.DataFrame(counts_traits.items(), columns=['Gangguan', 'Frekuensi']).sort_values('Frekuensi', ascending=False)
                    
                    fig_t, ax_t = plt.subplots(figsize=(5, 4))
                    colors_t = sns.color_palette("Reds_r", n_colors=len(df_traits))
                    
                    bars_t = ax_t.bar(df_traits['Gangguan'], df_traits['Frekuensi'], color=colors_t, width=0.6)
                    
                    for bar in bars_t:
                        height = bar.get_height()
                        ax_t.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7, fontweight='bold')
                        
                    plt.xticks(rotation=35, ha='right', fontsize=7)
                    ax_t.tick_params(axis='y', labelsize=7)
                    ax_t.set_ylim(0, df_traits['Frekuensi'].max() * 1.15)
                    
                    # Bersihkan frame agar tampilan minimalis dan modern
                    for spine in ['top', 'right', 'left']:
                        ax_t.spines[spine].set_visible(False)
                    ax_t.axes.get_yaxis().set_visible(False) 
                    plt.tight_layout()
                    
                    # Merender Grafik Prevalensi Gangguan ke Interface Streamlit
                    st.pyplot(fig_t)
                    
                    # Insight otomatis berdasarkan data gangguan tertinggi
                    top_trait = df_traits.iloc[0]['Gangguan']
                    st.caption(f"*Insight*: Sindrom Kepribadian **{top_trait}** mendominasi tren log pemeriksaan populasi. Diperlukan konsentrasi program intervensi primer pada domain tersebut.")
                else:
                    st.info("Belum ada temuan indikasi klinis di dalam database log pemeriksaan.")
                        
            # --- BAR CHART TOP GLOBAL FAKTOR RISIKO DARI DATABASE  ---
            with row2_col2:
                st.markdown("##### Faktor Risiko Global Dominan")
                
                global_pendorong_list = []
                target_col = "Faktor Risiko" # Sesuaikan dengan daftar kolom Anda
    
                if target_col in df_analitik.columns:
                    for idx, r in df_analitik.iterrows():
                        teks = str(r[target_col])
                        # Pastikan bukan data kosong
                        if teks and teks.lower() not in ["nan", "none", "tidak ada", "tidak ada data"]:
                            # Split koma dan bersihkan spasi
                            raw_list = [x.strip() for x in teks.split(",") if x.strip()]
                            for item in raw_list:
                                # Menghapus teks di dalam kurung jika ada (misal: "Anxiety (0.5)")
                                clean_item = re.sub(r'\s*\([^)]*\)', '', item)
                                global_pendorong_list.append(clean_item)

                # 2. Proses Plotting
                if global_pendorong_list:
                    from collections import Counter
                    counts_risks = Counter(global_pendorong_list)
                    df_risks = pd.DataFrame(counts_risks.items(), columns=['Faktor_Risiko', 'Frekuensi']).sort_values('Frekuensi', ascending=False).head(5)
                    
                    fig_r, ax_r = plt.subplots(figsize=(5, 4))
                    bars_r = ax_r.barh(df_risks['Faktor_Risiko'], df_risks['Frekuensi'], color='#C43636', height=0.5)
                    
                    for bar in bars_r:
                        width = bar.get_width()
                        ax_r.annotate(f'{int(width)}', xy=(width, bar.get_y() + bar.get_height() / 2),
                                    xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=7, fontweight='bold')
                    
                    ax_r.tick_params(axis='both', labelsize=7)
                    ax_r.invert_yaxis() 
                    ax_r.set_xlim(0, df_risks['Frekuensi'].max() * 1.3)
                    
                    for spine in ['top', 'right', 'left', 'bottom']:
                        ax_r.spines[spine].set_visible(False)
                    ax_r.axes.get_xaxis().set_visible(False)
                    plt.tight_layout()
                    
                    st.pyplot(fig_r)
                    
                    top_risk = df_risks.iloc[0]['Faktor_Risiko']
                    st.caption(f"*Insight*: Indikator perilaku **{top_risk}** divalidasi secara matematis oleh SHAP sebagai pemicu (*trigger*) risiko paling dominan pada populasi mahasiswa.")
                else:
                    st.info("Faktor risiko akan terpetakan di sini secara otomatis apabila terdapat rekaman data mahasiswa berisiko klinis di dalam database log.")
                    
            # -----------------------------------------------------------------
            # REKOMENDASI PANEL SEBARIS PENUH (FULL WIDTH DI PALING BAWAH TAB)
            # -----------------------------------------------------------------
            st.write("")
            st.info("""
            *Seluruh visualisasi di atas bergerak secara dinamis (real-time) mengikuti manipulasi, penambahan, atau penghapusan data rekam medis pada **Tab 1**. Visualisasi ini diharapkan dapat memberikan insights untuk merancang strategi intervensi dan program preventif kesehatan mental tahunan.*
            """)
        else:
            st.info("Belum ada data pemeriksaan mahasiswa di database. Selesaikan pengisian tes OMNI-IV terlebih dahulu untuk melihat dashboard analitik global ini.")

# ============================================================
# 5. FOOTER
# ============================================================
st.divider()
st.caption("Sistem Deteksi Risiko Gangguan Kepribadian OMNI-IV | Universitas Telkom - 2026")