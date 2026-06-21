import streamlit as st
import os
import uuid
import numpy as np
import librosa
import scipy.signal as signal
from scipy.ndimage import maximum_filter
import pandas as pd
import plotly.graph_objects as go

TARGET_SR = 16000
NEIGHBORHOOD_SIZE = 7
DYNAMIC_OFFSET_DB = 15
MAX_FREQ_DISPLAY = 4000
DATABASE_FOLDER = "EE200 Project Song Database"

st.set_page_config(page_title="EE200 Audio ID Engine", page_icon="🎧", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stRadio, .stButton {
    font-family: 'Poppins', sans-serif;
}

.main-header {
    background: linear-gradient(120deg, #7b2ff7, #f107a3, #00d2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    padding-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #aeb4c4;
    font-size: 1.2rem;
    font-weight: 400;
    margin-bottom: 1.8rem;
}
.section-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f5f5f7;
    border-left: 5px solid #f107a3;
    padding-left: 14px;
    margin-top: 1.6rem;
    margin-bottom: 0.8rem;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(123,47,247,0.18), rgba(0,210,255,0.12));
    border: 1px solid rgba(123,47,247,0.35);
    border-radius: 14px;
    padding: 16px 20px;
}
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.05rem !important;
    color: #c9c9d4 !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17172e, #0c0c1a);
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stRadio div {
    font-size: 1.08rem !important;
    color: #e6e6f0 !important;
}
.section-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #f107a3, transparent);
    margin: 1.8rem 0;
}
</style>
""", unsafe_allow_html=True)


def style_fig(fig, title):
    """Shared typography pass so every Plotly chart matches the rest of the UI."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#f5f5f7")),
        font=dict(size=13, color="#d7d7e0"),
        legend=dict(font=dict(size=12)),
        margin=dict(l=45, r=20, t=60, b=45),
    )
    return fig

def extract_peaks(Sxx_db):
    """Returns (peaks as list of (t_idx, f_idx), time_idx array, freq_idx array)."""
    local_max = maximum_filter(Sxx_db, size=NEIGHBORHOOD_SIZE) == Sxx_db
    peaks_mask = local_max & (Sxx_db > (np.mean(Sxx_db) + DYNAMIC_OFFSET_DB))
    freq_idx, time_idx = np.where(peaks_mask)
    peaks = list(zip(time_idx, freq_idx))
    return peaks, time_idx, freq_idx


def hash_peaks(peaks):
    """Anchor-pair hashing: (f1, f2, dt) -> t1, for landmark matching."""
    hashes = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        for j in range(i + 1, min(i + 15, n)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if 1 <= dt <= 30:
                hashes.append(((f1, f2, dt), t1))
    return hashes


@st.cache_resource(show_spinner="Building fingerprint database from song library...")
def build_database():
    if not os.path.exists(DATABASE_FOLDER):
        return {}, {}

    audio_files = sorted(f for f in os.listdir(DATABASE_FOLDER) if f.endswith(".mp3"))
    song_database = {}
    song_mapping = {}

    for filename in audio_files:
        path = os.path.join(DATABASE_FOLDER, filename)
        try:
            y, fs = librosa.load(path, sr=TARGET_SR)
            _, _, Sxx = signal.spectrogram(y, fs, nperseg=2048)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            peaks, _, _ = extract_peaks(Sxx_db)
            local_hashes = hash_peaks(peaks)

            song_idx = len(song_mapping)
            for hash_key, t1 in local_hashes:
                song_database.setdefault(hash_key, []).append((song_idx, t1))
            song_mapping[song_idx] = os.path.splitext(filename)[0]
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    return song_database, song_mapping


try:
    song_database, song_mapping = build_database()
except Exception as e:
    st.error(f"🚫 Failed to build the song database: {e}")
    st.stop()


def pipeline_and_match(query_path):
    y, fs = librosa.load(query_path, sr=TARGET_SR)
    freqs, times, Sxx = signal.spectrogram(y, fs, nperseg=2048)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    peaks, time_idx, freq_idx = extract_peaks(Sxx_db)

    song_offsets = {idx: [] for idx in song_mapping}
    peak_song_offset = {} 

    n = len(peaks)
    for i in range(n):
        t1_q, f1 = peaks[i]
        for j in range(i + 1, min(i + 15, n)):
            t2_q, f2 = peaks[j]
            dt = t2_q - t1_q
            if 1 <= dt <= 30:
                key = (f1, f2, dt)
                if key in song_database:
                    for song_idx, t1_s in song_database[key]:
                        offset = t1_s - t1_q
                        song_offsets[song_idx].append(offset)
                        peak_song_offset.setdefault(i, []).append((song_idx, offset))

    all_song_votes = {}
    best_idx, best_votes, best_offsets, best_bin = -1, 0, [], None

    for idx, offsets in song_offsets.items():
        if not offsets:
            continue
        counts, edges = np.histogram(offsets, bins=np.arange(min(offsets) - 1, max(offsets) + 2, 1))
        peak_votes = int(np.max(counts))
        all_song_votes[song_mapping[idx]] = peak_votes
        if peak_votes > best_votes:
            best_votes = peak_votes
            best_idx = idx
            best_offsets = offsets
            best_bin = edges[np.argmax(counts)]

    predicted_song = song_mapping.get(best_idx, "Unknown Track")

    matched_t, matched_f, unmatched_t, unmatched_f = [], [], [], []
    if best_idx != -1:
        for i, (t, f) in enumerate(peaks):
            contributions = peak_song_offset.get(i, [])
            is_matched = any(s == best_idx and abs(o - best_bin) < 1 for s, o in contributions)
            (matched_t if is_matched else unmatched_t).append(times[t])
            (matched_f if is_matched else unmatched_f).append(freqs[f])
    else:
        unmatched_t = list(times[time_idx])
        unmatched_f = list(freqs[freq_idx])

    return {
        "prediction": predicted_song,
        "votes": best_votes,
        "offsets": best_offsets,
        "all_votes": all_song_votes,
        "times": times,
        "freqs": freqs,
        "Sxx_db": Sxx_db,
        "time_idx": time_idx,
        "freq_idx": freq_idx,
        "matched": (matched_t, matched_f),
        "unmatched": (unmatched_t, unmatched_f),
    }

def plot_vote_comparison(all_votes, predicted):
    if not all_votes:
        return None
    df = pd.DataFrame(list(all_votes.items()), columns=["Song", "Votes"]).sort_values("Votes", ascending=False)
    colors = ["#f107a3" if s == predicted else "#3b3f6b" for s in df["Song"]]
    fig = go.Figure(go.Bar(x=df["Song"], y=df["Votes"], marker_color=colors,
                            text=df["Votes"], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=420, xaxis_tickangle=-30, showlegend=False)
    return style_fig(fig, "🏆 Vote Comparison Across Candidate Songs")


def plot_spectrogram(times, freqs, Sxx_db):
    mask = freqs <= MAX_FREQ_DISPLAY
    step = max(1, len(times) // 400)  
    z = Sxx_db[mask][:, ::step].astype(np.float32)  
    fig = go.Figure(go.Heatmap(z=z, x=times[::step], y=freqs[mask],
                                colorscale="Plasma", colorbar=dict(title="dB")))
    fig.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    return style_fig(fig, "🌈 Spectrogram")


def plot_constellation(times, freqs, time_idx, freq_idx):
    fig = go.Figure(go.Scatter(x=times[time_idx], y=freqs[freq_idx], mode="markers",
                                marker=dict(color="#00f5d4", size=5)))
    fig.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                       yaxis_range=[0, MAX_FREQ_DISPLAY])
    return style_fig(fig, "✨ Constellation Map")


def plot_histogram(offsets):
    fig = go.Figure(go.Histogram(x=offsets, marker_color="#7b2ff7"))
    fig.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Offset (bins)", yaxis_title="Vote Count")
    return style_fig(fig, "📊 Time-Offset Alignment Histogram")


def plot_fingerprint_matches(matched, unmatched):
    fig = go.Figure()
    if unmatched[0]:
        fig.add_trace(go.Scatter(x=unmatched[0], y=unmatched[1], mode="markers",
                                  marker=dict(color="#3b3f6b", size=5), name="Unmatched peak"))
    if matched[0]:
        fig.add_trace(go.Scatter(x=matched[0], y=matched[1], mode="markers",
                                  marker=dict(color="#f107a3", size=8, symbol="star"),
                                  name="Verified fingerprint"))
    fig.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                       yaxis_range=[0, MAX_FREQ_DISPLAY])
    return style_fig(fig, "🔑 Verified Fingerprint Matches")

st.markdown('<div class="main-header">🎧 EE200 Audio Identification Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Shazam-style landmark fingerprinting, built from scratch</div>', unsafe_allow_html=True)

if not song_database:
    st.error("🚫 Song database folder is missing or empty — check the repo for 'EE200 Project Song Database'.")
else:
    with st.sidebar:
        st.markdown("### 🎛️ Mode")
        mode = st.radio("Choose mode:", ["🎯 Single-Clip Mode", "📦 Batch Mode"], label_visibility="collapsed")
        st.markdown("---")
        st.caption(f"📚 Library size: **{len(song_mapping)}** songs indexed")

    if mode == "🎯 Single-Clip Mode":
        st.markdown('<div class="section-title">🎯 Single-Clip Diagnostic Identification</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a query clip", type=["mp3", "wav"])

        if uploaded_file is not None:
            ext = os.path.splitext(uploaded_file.name)[1] or ".wav"
            temp_path = f"temp_{uuid.uuid4().hex}{ext}"  
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Analyzing audio fingerprint..."):
                    result = pipeline_and_match(temp_path)

                col1, col2 = st.columns(2)
                col1.metric("🎵 Predicted Match", result["prediction"])
                col2.metric("✅ Confidence", f"{result['votes']} votes")

                if result["votes"] >= 20:
                    st.success("High-confidence match.")
                elif result["votes"] > 0:
                    st.warning("Low-confidence match — treat with caution.")
                else:
                    st.error("No match found.")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                vote_fig = plot_vote_comparison(result["all_votes"], result["prediction"])
                if vote_fig:
                    st.plotly_chart(vote_fig, use_container_width=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.plotly_chart(plot_spectrogram(result["times"], result["freqs"], result["Sxx_db"]),
                                 use_container_width=True)
                c2.plotly_chart(plot_constellation(result["times"], result["freqs"],
                                                    result["time_idx"], result["freq_idx"]),
                                 use_container_width=True)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                c3, c4 = st.columns(2)
                if result["offsets"]:
                    c3.plotly_chart(plot_histogram(result["offsets"]), use_container_width=True)
                else:
                    c3.info("No alignment data to plot.")
                c4.plotly_chart(plot_fingerprint_matches(result["matched"], result["unmatched"]),
                                 use_container_width=True)

            except Exception as e:
                st.error(f"Couldn't process this file: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    else:  # Batch mode
        st.markdown('<div class="section-title">📦 Automated Batch Identification</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload multiple clips", type=["mp3", "wav"], accept_multiple_files=True)

        if uploaded_files:
            results = []
            progress = st.progress(0)
            for i, up_file in enumerate(uploaded_files):
                ext = os.path.splitext(up_file.name)[1] or ".wav"
                temp_path = f"temp_{uuid.uuid4().hex}{ext}"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(up_file.getbuffer())
                    result = pipeline_and_match(temp_path)
                    results.append({"filename": up_file.name, "prediction": result["prediction"],
                                     "votes": result["votes"]})
                except Exception:
                    results.append({"filename": up_file.name, "prediction": "Error", "votes": 0})
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                progress.progress((i + 1) / len(uploaded_files))

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            csv_df = df_results[["filename", "prediction"]]
            st.download_button("📥 Download results.csv", csv_df.to_csv(index=False).encode("utf-8"),
                                file_name="results.csv", mime="text/csv")
