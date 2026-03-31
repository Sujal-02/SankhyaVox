import os
import sys
import numpy as np
import soundfile as sf
import librosa
import joblib
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from src.config import VOCAB, HMM_STATES, MODEL_DIR, SAMPLE_RATE, IDX_TO_TOKEN
from src.hmm import SankhyaHMMBase
from src.decoder import ConstrainedViterbiDecoder
from src.grammar import all_valid_sequences, number_to_tokens, _TOKEN_TO_VALUE
from scripts.extract_features import preprocess_audio, extract_mfcc

app = Flask(__name__)

print("Loading Models for API...")
# 1. HMM Engine
hmm_system = SankhyaHMMBase(VOCAB, HMM_STATES, str(MODEL_DIR))
try:
    hmm_system.load_models()
    decoder = ConstrainedViterbiDecoder(hmm_system, all_valid_sequences)
    print("🏛️ Classical HMM models loaded successfully!")
except Exception as e:
    print(f"Warning: HMM Models not found. {e}")
    decoder = None

# 2. Goated SVM Pipeline
try:
    svm_classifier = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    print("⚡ Goated SVM Pipeline loaded successfully!")
except Exception as e:
    print(f"Warning: SVM Model not found. {e}")
    svm_classifier = None

# 3. DNN Pipeline
try:
    dnn_classifier = joblib.load(os.path.join(MODEL_DIR, "dnn_model.pkl"))
    print("🧠 DNN Pipeline loaded successfully!")
except Exception as e:
    print(f"Warning: DNN Model not found. {e}")
    dnn_classifier = None

def chunk_pooling(X):
    """Pools MFCCs into 5 temporal chunks for the static models."""
    n_frames = X.shape[0]
    num_chunks = 5
    if n_frames < num_chunks:
        X = np.pad(X, ((0, num_chunks - n_frames), (0, 0)), mode='edge')
        n_frames = num_chunks
    embeddings = []
    chunk_size = n_frames / float(num_chunks)
    for i in range(num_chunks):
        s = int(i * chunk_size)
        e = int((i + 1) * chunk_size)
        if e == s: e = s + 1
        embeddings.append(np.mean(X[s:e], axis=0))
    return np.concatenate(embeddings)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_path = "temp_audio_api.wav"
    audio_file.save(temp_path)

    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        true_wav_path = "temp_decoded.wav"
        
        subprocess.run([
            ffmpeg_exe, "-y", "-i", temp_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", true_wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        audio, sr = sf.read(true_wav_path)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE: audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        audio = preprocess_audio(audio)
        X = extract_mfcc(audio)
        
        results = {}

        # HMM Prediction
        if decoder:
            h_num, h_score = decoder.decode(X)
            if h_num is not None and h_num > 99:
                h_num = -1
                h_toks = ["Out of range"]
            else:
                h_toks = number_to_tokens(h_num) if h_num is not None else []
            h_score = float(h_score) if h_score != float("-inf") else -9999.0
            results['hmm'] = {'number': h_num if h_num is not None else -1, 'tokens': h_toks, 'score': h_score}
        
        if len(X) > 0:
            emb = chunk_pooling(X)
            
            # SVM Prediction
            if svm_classifier:
                s_idx = svm_classifier.predict([emb])[0]
                s_token = IDX_TO_TOKEN[s_idx]
                s_num = _TOKEN_TO_VALUE.get(s_token, -1)
                s_score = max(svm_classifier.predict_proba([emb])[0]) * 100
                results['svm'] = {'number': s_num, 'tokens': [s_token], 'score': s_score}
                
            # DNN Prediction
            if dnn_classifier:
                d_idx = dnn_classifier.predict([emb])[0]
                d_token = IDX_TO_TOKEN[d_idx]
                d_num = _TOKEN_TO_VALUE.get(d_token, -1)
                d_score = max(dnn_classifier.predict_proba([emb])[0]) * 100
                results['dnn'] = {'number': d_num, 'tokens': [d_token], 'score': d_score}

        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(true_wav_path):
            os.remove(true_wav_path)
            
        return jsonify(results)
        
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        if 'true_wav_path' in locals() and os.path.exists(true_wav_path): os.remove(true_wav_path)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
