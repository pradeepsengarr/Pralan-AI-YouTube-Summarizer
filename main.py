import os
import re
import json
import tempfile
import time
import yt_dlp
import whisper
import requests
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
TOGETHER_URL = "https://api.together.xyz/v1/completions"

st.set_page_config(page_title="Pralan AI — YouTube Summarizer", page_icon="🎥", layout="wide")

st.markdown("""
<h1 style='text-align:center'>🎥 Pralan AI — YouTube Summarizer</h1>
<p style='text-align:center; font-size:16px; color:gray'>
Transform any YouTube video into an AI-generated summary with key points and insights.<br>
Built with  by <b>Prada + Alan</b> using Together API + Whisper.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ======================= HELPERS =======================
def extract_video_id(url: str) -> str:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Invalid YouTube URL. Example: https://www.youtube.com/watch?v=Fd0p17MLlSM")

def try_fetch_youtube_transcript(video_id: str):
    """Try to fetch transcript via youtube-transcript-api"""
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcripts.find_transcript(['en']).fetch()
        except NoTranscriptFound:
            transcript = list(transcripts)[0].fetch()
        return transcript
    except (NoTranscriptFound, TranscriptsDisabled):
        raise RuntimeError("No transcript or captions available for this video.")
    except VideoUnavailable:
        raise RuntimeError("Video unavailable or private.")
    except Exception as e:
        raise RuntimeError(f"Transcript fetch failed: {e}")

def download_audio_with_ytdlp(url: str) -> str:
    """Download audio using yt-dlp API"""
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "%(title)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")
        audio_file = os.path.join(tmp_dir, f"{title}.m4a")
    if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
        raise RuntimeError("Downloaded audio file invalid or empty.")
    return audio_file

def whisper_transcribe_file(audio_path: str, model_name="small") -> str:
    st.info("🎧 Transcribing with Whisper (this might take a minute)...")
    model = whisper.load_model(model_name)
    time.sleep(1)
    result = model.transcribe(audio_path)
    text = result.get("text", "").strip()
    if not text:
        raise RuntimeError("Whisper transcription returned empty text.")
    return text

def transcript_to_text(transcript):
    return " ".join([seg.get("text", "").strip() for seg in transcript])

def chunk_text(text, max_len=3000):
    words = text.split()
    chunks, chunk, count = [], [], 0
    for w in words:
        chunk.append(w)
        count += len(w) + 1
        if count >= max_len:
            chunks.append(" ".join(chunk))
            chunk, count = [], 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def call_together_model(prompt, max_tokens=512, temperature=0.3):
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TOGETHER_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    resp = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Together API error {resp.status_code}: {resp.text}")
    data = resp.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0].get("text", "").strip()
    return data.get("output", "").strip()

# ======================= SIDEBAR =======================
with st.sidebar:
    st.header("⚙️ Settings")
    whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small"], index=2)
    chunk_size = st.slider("Chunk Size (chars)", 1000, 8000, 3000, step=500)
    max_chunks = st.slider("Max Chunks", 1, 30, 10)
    temperature = st.slider("Generation Temperature", 0.0, 1.0, 0.3, 0.05)
    st.markdown("---")
    st.caption("💡 If transcript not available, Whisper automatically generates one.")

# ======================= MAIN LOGIC =======================
video_url = st.text_input("🔗 Paste a YouTube URL:", placeholder="https://www.youtube.com/watch?v=Fd0p17MLlSM")

if st.button("✨ Generate AI Summary"):
    if not video_url:
        st.error("Please enter a YouTube video URL.")
        st.stop()

    try:
        video_id = extract_video_id(video_url)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown("### Step 1️⃣ — Checking for transcript...")
    transcript = None
    use_whisper = False

    try:
        transcript = try_fetch_youtube_transcript(video_id)
        st.success("✅ Transcript found on YouTube.")
    except Exception as ex:
        msg = str(ex)
        if "no element found" in msg.lower():
            st.warning("⚠️ The video you provided does not have captions or transcript available.")
        else:
            st.warning(f"⚠️ Could not fetch transcript: {msg}")
        st.info("🎧 Switching to Whisper audio transcription automatically.")
        use_whisper = True

    if use_whisper:
        try:
            audio_path = download_audio_with_ytdlp(video_url)
            st.success("✅ Audio downloaded successfully.")
            text = whisper_transcribe_file(audio_path, whisper_model)
            transcript = [{"text": text, "start": 0.0, "duration": 0.0}]
            st.success("✅ Whisper transcription complete.")
        except Exception as e:
            st.error(f"Whisper transcription failed: {e}")
            st.stop()
        finally:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)

    text = transcript_to_text(transcript)
    if not text:
        st.error("Transcript is empty — nothing to summarize.")
        st.stop()

    chunks = chunk_text(text, max_len=chunk_size)
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    st.info(f"🧩 Split transcript into {len(chunks)} chunks.")

    st.markdown("### Step 2️⃣ — Generating Summaries for Each Chunk...")
    chunk_summaries = []
    progress = st.progress(0)
    for i, ch in enumerate(chunks):
        prompt = f"""
Summarize this transcript chunk clearly and concisely:
{ch}

Provide:
1️⃣ A 2–3 sentence summary.
2️⃣ 5 key points.
3️⃣ 3 major topics.
"""
        try:
            summary = call_together_model(prompt, temperature=temperature)
            chunk_summaries.append(summary)
        except Exception as e:
            chunk_summaries.append(f"[Error summarizing chunk {i+1}: {e}]")
        progress.progress((i + 1) / len(chunks))
    progress.empty()
    st.success("✅ Chunk summarization complete.")

    st.markdown("### Step 3️⃣ — Creating Final Summary & Key Topics...")
    final_prompt = "Combine these chunk summaries into a clear, insightful overview with 10 bullet points:\n\n" + "\n\n".join(chunk_summaries)
    try:
        final_summary = call_together_model(final_prompt, max_tokens=700)
    except Exception as e:
        final_summary = f"[Error generating final summary: {e}]"

    st.markdown("## 🎯 Final AI Summary")
    st.text_area("Summary", final_summary, height=250)

    with st.expander("📄 View All Chunk Summaries"):
        for idx, s in enumerate(chunk_summaries):
            st.markdown(f"**Chunk {idx + 1}:**")
            st.write(s)
            st.markdown("---")

    result = {
        "video_url": video_url,
        "summary": final_summary,
        "chunks": chunk_summaries
    }

    st.download_button("⬇️ Download Summary (JSON)", json.dumps(result, indent=2),
                       file_name=f"{video_id}_summary.json", mime="application/json")

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>🚀 Powered by Together API + Whisper | Built by Prada ✨</p>", unsafe_allow_html=True)
