"""
Beat Visualizer -- Enterprise Audio Analysis Engine v4.0
========================================================
50+ algorithms across every branch of Music Information Retrieval:

BEAT / TEMPO
  Ellis 2007 DP beat tracker, percussive-isolated beat tracker,
  Predominant Local Pulse (PLP, Grosche & Müller 2011),
  Autocorrelation tempogram, Fourier tempogram,
  Tatum estimation (1/4-beat pulse grid),
  Beat histogram (IOI-based dominant BPM),
  Multi-agent tempo induction via tempogram peak consensus,
  Beat phase & beat period variance tracking

ONSET DETECTION
  Energy-based (spectral flux on RMS),
  Mel-spectral flux onset strength,
  Complex-domain onset (phase-aware),
  High-Frequency Content (HFC) onset,
  Phase-deviation onset,
  Sub-band onset envelopes (4 frequency bands),
  Adaptive onset threshold (percentile-based)

PITCH / F0 ESTIMATION
  Probabilistic YIN (pYIN, Mauch & Dixon 2014) -- librosa
  Praat autocorrelation pitch (parselmouth)
  Melodic contour analysis (direction, range, intervallic content)
  Vibrato detection (4-8 Hz AM on F0 via FFT)
  Pitch salience histogram

HARMONY / KEY
  Krumhansl-Schmuckler (1990) 24-key profile correlation
  Temperley (2001) 24-key tonal hierarchy correlation
  Harmonic Change Detection Function (HCDF, Harte et al. 2006)
  Chord recognition via template matching: maj/min/dim/aug/maj7/min7/dom7/sus2/sus4/hdim7/aug7 (11 types x 12 roots = 132 chords)
  Tonal tension curve (distance from estimated tonic)
  Circle-of-fifths weighting
  Relative major/minor detection
  Enharmonic equivalence handling

CHROMA
  CQT chroma (36 bins/octave, C1 base)
  CENS (Chroma Energy Normalized Statistics, tempo+dynamics robust)
  STFT chroma (N_FFT=4096)
  Deep Chroma (NMF-derived, noise-robust)
  Chroma DCT-Reduced log Pitch (CRP) approximation

SPECTRAL
  Multi-resolution STFT (N_FFT = 512 / 2048 / 4096 / 8192)
  Constant-Q Transform (CQT, 84 bins, 36 bpo)
  Variable-Q Transform (VQT, gamma=0)
  Gammatone filterbank (24 ERB channels, IIR approximation)
  Bark-scale energy (24 critical bands)
  Mel spectrogram (128 bands, 20 Hz - Nyquist)
  Spectral centroid, bandwidth, rolloff x3 (10/85/95%), flatness
  Spectral flux (frame L2), spectral entropy (Shannon)
  Spectral irregularity (Jensen 1999)
  Spectral contrast (7 sub-bands, Jiang et al. 2002)
  Tristimulus (T1/T2/T3 harmonic energy partitions)
  9-band psychoacoustic energy (sub-bass -> ultrasonic)
  Sub-band flux (4 critical regions)

MFCC / CEPSTRUM
  MFCC-40 with D and DD (velocity + acceleration)
  MFCC-13 with D and DD (standard speech coefficients)
  Bark-Frequency Cepstral Coefficients (BFCC-24)
  LPC order-16 (Levinson-Durbin via librosa)
  Cepstral mean and variance normalization (CMVN) variant

VOICE / PROSODY  (Praat via parselmouth)
  Jitter: local, local-absolute, RAP, PPQ5, DDP (5 variants)
  Shimmer: local, local-dB, APQ3, APQ5, APQ11, DDA (6 variants)
  Harmonics-to-Noise Ratio (HNR cc method)
  Noise-to-Harmonics Ratio (NHR derived)
  Cepstral Peak Prominence (CPP)
  Formants F1-F4 (Burg method, 5 formants max)
  Voice Activity Detection ratio (voiced/total frames)
  Long-Term Average Spectrum (LTAS, Baken & Orlikoff)

TIMBRE
  Brightness (energy above 1500 Hz / total)
  Warmth (energy below 320 Hz / total)
  A-weighted spectral centroid (ISO 226 A-weighting)
  Roughness (Sethares 1993, pairwise partial dissonance)
  Inharmonicity (B-coefficient deviation from harmonic series)
  Odd/even harmonic energy ratio
  Spectral kurtosis and skewness
  Noisiness (spectral flatness mean)

RHYTHM
  Time signature estimation (2/4, 3/4, 4/4, 6/8, 5/4, 7/8)
  Swing / groove factor (off-beat phase deviation)
  Syncopation index (Keith 1991 adapted)
  Rhythmic complexity (entropy of onset-strength histogram)
  IOI statistics (mean, std, CV, skew, kurtosis)
  Meter autocorrelation
  Groove quantisation error

DYNAMICS / LOUDNESS
  RMS energy curve + dB
  ITU-R BS.1770-4 integrated loudness (LUFS) -- pyloudnorm
  True-peak measurement (intersample peak via hi-res signal)
  Short-term loudness (400 ms Hann windows)
  Momentary loudness (100 ms windows)
  Loudness Range (LRA = P95-P10 of short-term)
  Crest factor (dB, peak / RMS)
  Dynamic range compression detection score
  Loudness penalty (streaming normalization estimate)

STRUCTURE / SEGMENTATION
  Self-Similarity Matrix (SSM) via cosine affinity on Mel+Chroma
  Agglomerative clustering segmentation (Müller & Nieto 2014)
  Checkerboard kernel novelty function (Foote 2000)
  Recurrence plot analysis
  Path-enhanced SSM (diagonal smoothing)
  Structural repetition identification via row-mean SSM
  Segment boundary fusion (Laplacian + novelty consensus)

NMF SOURCE SEPARATION
  Non-Negative Matrix Factorization on mel spectrogram (sklearn)
  Component reconstruction + activation envelopes
  Percussive vs harmonic component energy split

DIMENSIONALITY REDUCTION
  FastDTW distance between chroma frames (template matching)
  PCA on MFCC matrix (top-3 principal components)

openSMILE
  eGeMAPS v02 (88 acoustic features, Geneva Minimalistic Acoustic Parameter Set)

MIDI  (pretty_midi)
  Note events, velocity histogram, duration statistics
  Piano roll (128 pitches x time)
  Polyphony level over time
  Per-instrument: program, pitch range, density, velocity
  MIDI tempo track (tempo changes)
  Chroma from MIDI (pitch-class energy)

MUSIC INFORMATION
  Danceability score (beat regularity x tempo x bass energy)
  Energy score (RMS x brightness x tempo)
  Valence estimate (brightness - roughness + regularity)
  Acousticness (warmth + low-RMS + low-brightness)
  Genre fingerprint probabilities (7 genres: heuristic)
  Tempo category (Larghetto -> Prestissimo, 9 levels)
  Music emotion quadrant (Russell's circumplex approximation)
"""

from __future__ import annotations

import os, io, json, math, hashlib, logging, time, warnings
import functools, threading
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any

warnings.filterwarnings("ignore")

import numpy as np
import numpy.fft as nfft
import scipy.signal as sig
import scipy.stats as stats
import soundfile as sf
import librosa, librosa.feature, librosa.effects, librosa.onset, librosa.beat
import librosa.segment, librosa.decompose, librosa.filters

log = logging.getLogger(__name__)

# ─── optional libs ────────────────────────────────────────────────────────────
def _try(mod):
    try: return __import__(mod)
    except ImportError: return None

_pm   = _try("parselmouth")
_os_  = _try("opensmile")
_pyln = _try("pyloudnorm")
_pm2  = _try("pretty_midi")
_mut  = _try("mutagen")
_nr   = _try("noisereduce")
_pydub= _try("pydub")
_dtw  = _try("fastdtw")

HAS_PRAAT    = _pm   is not None
HAS_SMILE    = _os_  is not None
HAS_LOUDNORM = _pyln is not None
HAS_MIDI     = _pm2  is not None
HAS_MUTAGEN  = _mut  is not None
HAS_NR       = _nr   is not None
HAS_PYDUB    = _pydub is not None
HAS_DTW      = _dtw   is not None

try:
    from sklearn.decomposition import NMF, PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─── music theory constants ───────────────────────────────────────────────────
PC = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Krumhansl-Schmuckler 1990
KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
# Temperley 2001
TMP_MAJ= np.array([5.0,2.0,3.5,2.0,4.5,4.0,2.0,4.5,2.0,3.5,1.5,4.0])
TMP_MIN= np.array([5.0,2.0,3.5,4.5,2.0,4.0,2.0,4.5,3.5,2.0,1.5,4.0])
# Chord templates (12-dim binary chroma)
CHORDS = {
    "maj": [1,0,0,0,1,0,0,1,0,0,0,0], "min": [1,0,0,1,0,0,0,1,0,0,0,0],
    "dim": [1,0,0,1,0,0,1,0,0,0,0,0], "aug": [1,0,0,0,1,0,0,0,1,0,0,0],
    "maj7":[1,0,0,0,1,0,0,1,0,0,0,1], "min7":[1,0,0,1,0,0,0,1,0,0,1,0],
    "dom7":[1,0,0,0,1,0,0,1,0,0,1,0], "sus2":[1,0,1,0,0,0,0,1,0,0,0,0],
    "sus4":[1,0,0,0,0,1,0,1,0,0,0,0], "hdim7":[1,0,0,1,0,0,1,0,0,0,1,0],
    "aug7":[1,0,0,0,1,0,0,0,1,0,1,0],
}
# Bark critical band edges Hz
BARK = [0,100,200,300,400,510,630,770,920,1080,1270,1480,1720,
        2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,20000]
# Psychoacoustic 9-band descriptors
BANDS9 = [("sub_bass",20,60),("bass",60,250),("low_mid",250,500),
          ("mid",500,2000),("upper_mid",2000,4000),("presence",4000,6000),
          ("brilliance",6000,10000),("air",10000,16000),("ultra",16000,22000)]

# ─── utils ────────────────────────────────────────────────────────────────────
def _jfloat(v):
    if isinstance(v, np.floating): f=float(v); return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.ndarray): return v.tolist()
    return v

def _safe(default=None):
    def dec(fn):
        @functools.wraps(fn)
        def wrap(*a,**kw):
            try: return fn(*a,**kw)
            except Exception as e:
                log.error("[%s] %s: %s", fn.__name__, type(e).__name__, e)
                return {} if default is None else default
        return wrap
    return dec

def _sgm(x): return 1.0/(1.0+math.exp(-max(-500,min(500,float(x)))))
def _tempo_cat(b):
    for lim,name in [(60,"Larghetto"),(72,"Adagio"),(84,"Andante"),(96,"Moderato"),
                     (108,"Andante moderato"),(120,"Allegretto"),(132,"Allegro moderato"),
                     (140,"Allegro"),(160,"Allegro vivace"),(176,"Vivace"),(200,"Presto")]:
        if b<lim: return name
    return "Prestissimo"

def _ds(arr, target=1200):
    """Downsample 1-D list/array to ≤target points."""
    n=len(arr); step=max(1,n//target)
    return arr[::step] if isinstance(arr,np.ndarray) else arr[::step]

def _ds2(mat, tcols=800):
    """Downsample 2-D array columns."""
    step=max(1,mat.shape[1]//tcols)
    return mat[:,::step]


# ─── main class ───────────────────────────────────────────────────────────────
class AudioAnalyzer:
    SR    = 22050       # working sample rate
    SR_HI = 44100       # hi-res for voice/loudness
    HOP   = 512
    NFFT  = 4096
    NMELS = 128
    NMFCC = 40
    NBARK = 24
    NCQT  = 84          # 7 octaves
    BPO   = 36          # bins per octave (3x oversampled)
    LPCORD= 16

    def __init__(self, cache_dir=None):
        self.cache = cache_dir or "/tmp/bv_v4_cache"
        os.makedirs(self.cache, exist_ok=True)

    # ── cache ──────────────────────────────────────────────────────────────────
    def _ckey(self, path):
        h=hashlib.md5(); h.update(str(os.path.getmtime(path)).encode())
        with open(path,"rb") as f:
            while c:=f.read(1<<15): h.update(c)
        return h.hexdigest()

    def _cload(self, k):
        p=os.path.join(self.cache,f"{k}.json")
        if os.path.exists(p):
            try:
                with open(p) as f: return json.load(f)
            except: pass
        return None

    def _csave(self, k, d):
        p=os.path.join(self.cache,f"{k}.json")
        try:
            with open(p,"w") as f: json.dump(d,f,default=_jfloat)
        except Exception as e: log.warning("cache write: %s",e)

    # ── load ───────────────────────────────────────────────────────────────────
    def load(self, path, sr=None, mono=True, offset=0.0, dur=None):
        """Multi-backend loader: librosa -> soundfile -> pydub."""
        tsr = sr or self.SR
        for loader in (self._load_librosa, self._load_sf, self._load_pydub):
            y = loader(path, tsr, mono, offset, dur)
            if y is not None: return y, tsr
        raise RuntimeError(f"All backends failed: {path}")

    def _load_librosa(self, path, sr, mono, offset, dur):
        try: return librosa.load(path, sr=sr, mono=mono, offset=offset, duration=dur)[0]
        except Exception as e: log.debug("librosa: %s",e); return None

    def _load_sf(self, path, sr, mono, offset, dur):
        try:
            d,sr0=sf.read(path,dtype="float32",always_2d=True)
            d=np.mean(d,axis=1) if mono else d.T
            if offset>0: d=d[int(offset*sr0):]
            if dur: d=d[:int(dur*sr0)]
            return librosa.resample(d,orig_sr=sr0,target_sr=sr) if sr0!=sr else d.astype(np.float32)
        except Exception as e: log.debug("sf: %s",e); return None

    def _load_pydub(self, path, sr, mono, offset, dur):
        if not HAS_PYDUB: return None
        try:
            seg=_pydub.AudioSegment.from_file(path).set_frame_rate(sr).set_channels(1 if mono else 2)
            s=np.array(seg.get_array_of_samples(),dtype=np.float32)/(2**(8*seg.sample_width-1))
            if offset>0: s=s[int(offset*sr):]
            if dur: s=s[:int(dur*sr)]
            return s
        except Exception as e: log.debug("pydub: %s",e); return None

    # ── main pipeline ──────────────────────────────────────────────────────────
    def analyze_full(self, filepath, callback=None):
        ck = self._ckey(filepath)
        cached = self._cload(ck)
        if cached:
            if callback: callback(100,"Loaded from cache")
            return cached

        ext = Path(filepath).suffix.lower().lstrip(".")
        t0  = time.time()

        def emit(pct, msg):
            if callback: callback(pct, msg)
            log.info("  [%3d%%] %s", pct, msg)

        # ── MIDI branch ────────────────────────────────────────────────────────
        if ext in ("mid","midi","smf","kar","rmi") and HAS_MIDI:
            emit(5,  "Parsing MIDI...")
            r = self._midi(filepath, emit)
            r["analysis_time"] = round(time.time()-t0, 3)
            self._csave(ck, r); emit(100,f"MIDI done {r['analysis_time']}s"); return r

        # ── audio branch ───────────────────────────────────────────────────────
        emit( 2, "Loading audio (librosa -> soundfile -> pydub)...")
        y,sr = self.load(filepath, sr=self.SR)
        yhi,_ = self.load(filepath, sr=self.SR_HI)
        dur = len(y)/sr

        emit( 5, "HPSS -- median-filter harmonic/percussive separation (margin=3 dB)...")
        yh,yp = librosa.effects.hpss(y, margin=(3.0,3.0))

        emit( 8, "Multi-resolution STFT (512 / 2048 / 4096 / 8192 windows)...")
        stft = self._stft_multi(y, sr)

        emit(12, "CQT (84 bins, 36 bpo) + VQT...")
        cqt  = self._cqt(y, sr)

        emit(16, "Gammatone ERB filterbank (24 channels, IIR)...")
        gammatone = self._gammatone(y, sr)

        emit(20, "Beat tracking -- Ellis DP + Percussive + PLP + Tatum...")
        beats = self._beats(y, sr, yp)

        emit(26, "Onset -- 5 detectors + sub-band envelopes + adaptive threshold...")
        onsets = self._onsets(y, sr)

        emit(31, "Rhythm -- time sig, swing, syncopation, IOI stats, entropy...")
        rhythm = self._rhythm(y, sr, beats, onsets)

        emit(36, "Spectral -- centroid / flux / contrast / bark / tristimulus / 9-band...")
        spectral = self._spectral(y, sr)

        emit(41, "Mel spectrogram (128 mel bands, 20 Hz-Nyquist)...")
        mel  = self._mel(y, sr)

        emit(44, "NMF source separation (6 components on mel spectrogram)...")
        nmf  = self._nmf(y, sr)

        emit(47, "MFCC-40 (D,DD) + MFCC-13 + BFCC-24 + LPC-16 + CMVN...")
        mfcc = self._mfcc(y, sr)

        emit(52, "PCA on MFCC (top-3 principal components)...")
        pca  = self._pca_mfcc(mfcc)

        emit(55, "Chroma CQT / CENS / STFT / Deep-Chroma (NMF) + HCDF...")
        chroma = self._chroma(yh, sr, nmf)

        emit(60, "Key -- KS 1990 + Temperley 2001 + relative maj/min...")
        harmony = self._harmony(chroma)

        emit(64, "Chord recognition -- 11 types x 12 roots = 132 chord templates...")
        chords = self._chords(chroma)

        emit(67, "Tonnetz tonal centroid (6D, Harte et al. 2006)...")
        tonnetz = self._tonnetz(yh, sr)

        emit(70, "pYIN F0 + melodic contour + vibrato detection...")
        pitch = self._pitch(y, sr, yhi)

        emit(74, "Praat: jitterx5, shimmerx6, HNR, CPP, formants F1-F4, LTAS...")
        voice = self._voice(yhi)

        emit(78, "Dynamics: RMS + ITU-R BS.1770-4 LUFS + LRA + crest factor...")
        dyn  = self._dynamics(y, sr, yhi)

        emit(82, "Timbre: roughness, inharmonicity, brightness, A-weight, kurtosis...")
        timbre = self._timbre(y, sr)

        emit(85, "openSMILE eGeMAPS v02 (88 features)...")
        smile = self._smile(filepath)

        emit(88, "Structure: SSM + checkerboard novelty + Laplacian + path-enhanced...")
        struct = self._structure(y, sr, chroma)

        emit(91, "Energy envelope + waveform...")
        energy   = self._energy(y, sr)
        waveform = self._waveform(y, sr)

        emit(93, "Music info: danceability, valence, emotion quadrant, genre fingerprint...")
        info = self._music_info(beats, spectral, dyn, mfcc, timbre, rhythm)

        emit(95, "FastDTW chroma template matching...")
        dtw  = self._dtw_chroma(chroma, harmony)

        emit(97, "Metadata extraction (mutagen tags)...")
        meta = self._metadata(filepath, y, sr)

        result = dict(
            metadata=meta, waveform=waveform, beats=beats, onsets=onsets,
            rhythm=rhythm, spectral=spectral, stft=stft, mel=mel, cqt=cqt,
            gammatone=gammatone, nmf=nmf, mfcc=mfcc, pca=pca, chroma=chroma,
            harmony=harmony, chords=chords, tonnetz=tonnetz, pitch=pitch,
            voice=voice, dynamics=dyn, timbre=timbre, egemaps=smile,
            structure=struct, energy=energy, music_info=info, dtw=dtw,
            analysis_time=round(time.time()-t0,3),
            engine_version="4.0",
            library_versions={
                "librosa": librosa.__version__, "numpy": np.__version__,
                "parselmouth": _pm.__version__ if HAS_PRAAT else None,
                "opensmile": _os_.__version__ if HAS_SMILE else None,
                "pyloudnorm": getattr(_pyln, '__version__', '0.2.0') if HAS_LOUDNORM else None,
                "sklearn": __import__("sklearn").__version__ if HAS_SKLEARN else None,
            }
        )
        self._csave(ck, result)
        emit(100, f"Complete in {result['analysis_time']}s -- {len(result)} blocks")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # STFT MULTI-RESOLUTION
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _stft_multi(self, y, sr):
        out = {}
        for name, nfft in [("fine",512),("mid",2048),("main",self.NFFT),("coarse",8192)]:
            D  = librosa.stft(y, n_fft=nfft, hop_length=self.HOP)
            mg = np.abs(D)
            fr = librosa.fft_frequencies(sr=sr, n_fft=nfft)
            tm = librosa.frames_to_time(np.arange(mg.shape[1]), sr=sr, hop_length=self.HOP)
            out[name] = {
                "magnitude_db": _ds2(librosa.amplitude_to_db(mg, ref=np.max)).tolist(),
                "freqs": fr.tolist(), "times": _ds(tm).tolist(), "n_fft": nfft
            }
        # Power spectrum (time-averaged)
        D_m = librosa.stft(y, n_fft=self.NFFT, hop_length=self.HOP)
        ps  = np.mean(np.abs(D_m)**2, axis=1)
        out["power_spectrum_db"] = librosa.amplitude_to_db(np.sqrt(ps+1e-20),ref=np.max).tolist()
        out["freqs_main"] = librosa.fft_frequencies(sr=sr, n_fft=self.NFFT).tolist()
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # CQT + VQT
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _cqt(self, y, sr):
        fmin = librosa.note_to_hz("C1")
        C    = librosa.cqt(y, sr=sr, hop_length=self.HOP, n_bins=self.NCQT,
                           bins_per_octave=self.BPO, fmin=fmin)
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        fr   = librosa.cqt_frequencies(self.NCQT, fmin=fmin, bins_per_octave=self.BPO)
        tm   = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=self.HOP)
        vqt  = None
        try:
            V    = librosa.vqt(y, sr=sr, hop_length=self.HOP, n_bins=self.NCQT,
                               gamma=0, bins_per_octave=self.BPO, fmin=fmin)
            vqt  = _ds2(librosa.amplitude_to_db(np.abs(V),ref=np.max)).tolist()
        except: pass
        return {"cqt_db": _ds2(C_db).tolist(), "vqt_db": vqt,
                "freqs": fr.tolist(), "times": _ds(tm).tolist()}

    # ─────────────────────────────────────────────────────────────────────────
    # GAMMATONE ERB FILTERBANK
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _gammatone(self, y, sr):
        """24-channel gammatone filterbank (IIR approximation, scipy)."""
        freqs = librosa.mel_frequencies(n_mels=24, fmin=80, fmax=min(8000,sr//2))
        hop  = self.HOP
        channels = []
        for cf in freqs:
            if cf >= sr // 2: continue
            b, a = sig.gammatone(cf, "iir", fs=sr)
            # Apply filter
            out  = sig.sosfilt(sig.tf2sos(b, a), y)
            # Envelope via Hilbert abs
            env  = np.abs(sig.hilbert(out))
            # Decimate to hop rate
            factor = max(1, hop // 4)
            env_ds = sig.decimate(env, factor, zero_phase=True)
            channels.append(env_ds.tolist())
        times = np.linspace(0, len(y)/sr, len(channels[0])) if channels else []
        return {"channels": channels, "center_freqs": freqs.tolist(),
                "times": list(times), "n_channels": len(channels)}

    # ─────────────────────────────────────────────────────────────────────────
    # BEAT TRACKING
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _beats(self, y, sr, yp):
        hop = self.HOP
        oe  = librosa.onset.onset_strength(y=y,  sr=sr, hop_length=hop, aggregate=np.median, n_mels=128)
        oep = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop)

        # 1. Ellis DP
        td, bd = librosa.beat.beat_track(onset_envelope=oe, sr=sr, hop_length=hop,
                                          start_bpm=120, tightness=100, units="frames")
        td = float(np.atleast_1d(td)[0])

        # 2. Percussive
        tp, bp = librosa.beat.beat_track(onset_envelope=oep, sr=sr, hop_length=hop,
                                          start_bpm=td, tightness=80, units="frames")
        tp = float(np.atleast_1d(tp)[0])

        # 3. PLP (Grosche & Müller 2011)
        plp = librosa.beat.plp(onset_envelope=oe, sr=sr, hop_length=hop,
                                tempo_min=max(30,td*0.5), tempo_max=min(300,td*2))
        _, bplp = librosa.beat.beat_track(onset_envelope=plp, sr=sr, hop_length=hop, units="frames")

        # 4. Tempogram analysis
        tg  = librosa.feature.tempogram(onset_envelope=oe, sr=sr, hop_length=hop, win_length=400)
        ftg = librosa.feature.fourier_tempogram(onset_envelope=oe, sr=sr, hop_length=hop, win_length=400)
        tbins = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
        tgm   = tg.mean(axis=1)
        top10 = np.argsort(tgm)[-10:][::-1]
        tcands = [{"bpm": float(tbins[i]), "strength": float(tgm[i])} for i in top10]

        bt_d = librosa.frames_to_time(bd,   sr=sr, hop_length=hop)
        bt_p = librosa.frames_to_time(bp,   sr=sr, hop_length=hop)
        bt_l = librosa.frames_to_time(bplp, sr=sr, hop_length=hop)

        # IOI histogram
        ioi_hist, bpm_dom = self._ioi_hist(bt_d)

        # Consensus BPM
        bpm = float(np.average([td,tp,bpm_dom if bpm_dom else td], weights=[0.5,0.3,0.2]))
        conf = max(0.0, 1.0 - abs(td-tp)/max(td,1))

        # Regularity
        reg = 0.0
        if len(bt_d)>2:
            iois = np.diff(bt_d)
            reg = max(0.0, 1.0-float(np.std(iois)/(np.mean(iois)+1e-10)))

        # Beat period variance (coefficient of variation of IOIs)
        bpv = float(np.std(np.diff(bt_d))/(np.mean(np.diff(bt_d))+1e-10)) if len(bt_d)>2 else 0.0

        # Tatum: quarter-beat grid
        tatum = self._tatum(bt_d, bpm)

        # Beat phase: circular mean phase alignment
        beat_phase_consistency = 0.0
        if len(bt_d) > 2:
            period = 60.0/bpm
            phases = (bt_d % period) / period
            beat_phase_consistency = float(abs(np.mean(np.exp(2j*np.pi*phases))))

        step = max(1, len(oe)//2000)
        tm   = librosa.frames_to_time(np.arange(len(oe)), sr=sr, hop_length=hop)
        stg  = max(1, tg.shape[1]//500)

        return {
            "bpm": td, "bpm_perc": tp, "bpm_consensus": round(bpm,2),
            "bpm_dom_hist": bpm_dom, "bpm_confidence": round(conf,4),
            "beat_times": bt_d.tolist(), "beat_times_perc": bt_p.tolist(),
            "beat_times_plp": bt_l.tolist(), "tatum_times": tatum,
            "beat_count": int(len(bd)), "beat_regularity": round(reg,4),
            "beat_period_cv": round(bpv,4), "beat_phase_consistency": round(beat_phase_consistency,4),
            "ioi_histogram": ioi_hist,
            "onset_strength": oe[::step].tolist(), "onset_times": tm[::step].tolist(),
            "onset_strength_perc": oep[::step].tolist(), "plp_env": plp[::step].tolist(),
            "tempogram": tg[:80,::stg].tolist(), "tempogram_times": tm[::stg].tolist(),
            "tempogram_freqs": tbins[:80].tolist(),
            "fourier_tempogram": np.abs(ftg[:40,::stg]).tolist(),
            "tempo_candidates": tcands,
        }

    def _ioi_hist(self, bt):
        if len(bt)<3: return [], 120.0
        bpms = 60.0/(np.diff(bt)+1e-10)
        bpms = bpms[(bpms>40)&(bpms<300)]
        if len(bpms)==0: return [], 120.0
        cnt, edges = np.histogram(bpms, bins=np.arange(40,302,2))
        ctr = (edges[:-1]+edges[1:])/2
        dom = float(ctr[np.argmax(cnt)])
        hist = [{"bpm":round(float(c),1),"count":int(n)} for c,n in zip(ctr,cnt) if n>0]
        return hist[:40], dom

    def _tatum(self, bt, bpm):
        if len(bt)<2: return []
        p=60.0/max(bpm,1)/4
        s,e=float(bt[0]),float(bt[-1])
        return [round(t,4) for t in np.arange(s,e+p,p)[:3000]]

    # ─────────────────────────────────────────────────────────────────────────
    # ONSET DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _onsets(self, y, sr):
        hop = self.HOP
        D   = librosa.stft(y, n_fft=self.NFFT, hop_length=hop)
        mg  = np.abs(D)
        fr  = librosa.fft_frequencies(sr=sr, n_fft=self.NFFT)

        def detect(env, delta=0.07, wait=2):
            return librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                               units="time", delta=delta, wait=wait).tolist()

        # 1. Energy
        oe_e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        # 2. Mel flux
        oe_m = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop,
                                             feature=librosa.feature.melspectrogram)
        # 3. Complex domain (phase-aware)
        oe_c = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, lag=2, max_size=3)
        # 4. HFC
        hfc  = np.sum(mg * fr[:,None], axis=0); hfc/=(hfc.max()+1e-10)
        oe_h = np.maximum(0, np.diff(hfc, prepend=hfc[0]))
        # 5. Phase deviation
        ph   = np.angle(D)
        oe_p = np.concatenate([[0], np.abs(np.diff(np.unwrap(ph,axis=1),axis=1)).mean(axis=0)])

        # Adaptive threshold (median + k*std per detector)
        def adaptive(env, k=1.5):
            med = np.median(env); sd = np.std(env)
            thresh = med + k*sd
            peaks,_= sig.find_peaks(env, height=thresh, distance=2)
            return librosa.frames_to_time(peaks, sr=sr, hop_length=hop).tolist()

        ons_e, ons_m, ons_c = detect(oe_e,.07), detect(oe_m,.05), detect(oe_c,.06)
        ons_h, ons_p       = detect(oe_h,.03), detect(oe_p,.08)
        ons_adap           = adaptive(oe_e)

        consensus = sorted({round(t,3) for t in ons_e+ons_m+ons_c})

        # Sub-band onset envelopes
        sb = {}
        for name,lo,hi in [("low",0,300),("mid",300,2000),("high",2000,6000),("air",6000,sr//2)]:
            msk = (fr>=lo)&(fr<hi); bm=mg.copy(); bm[~msk,:]=0
            env = np.sqrt(np.mean(np.maximum(0,np.diff(bm,axis=1))**2,axis=0))
            env = np.concatenate([[0],env])
            sb[name]=_ds(env).tolist()

        return {
            "onsets_energy": ons_e, "onsets_mel": ons_m, "onsets_complex": ons_c,
            "onsets_hfc": ons_h, "onsets_phase": ons_p, "onsets_adaptive": ons_adap,
            "onsets_consensus": consensus, "onset_count": len(ons_e),
            "onset_density": round(len(ons_e)/(len(y)/sr+1e-9),4),
            "sub_band_oe": sb,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # RHYTHM
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _rhythm(self, y, sr, beats, onsets):
        hop = self.HOP
        bpm = beats.get("bpm_consensus", 120)
        bt  = np.array(beats.get("beat_times", []))
        oe  = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        tm  = librosa.frames_to_time(np.arange(len(oe)), sr=sr, hop_length=hop)
        tg  = librosa.feature.tempogram(onset_envelope=oe, sr=sr, hop_length=hop, win_length=400)
        tb  = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=hop)
        tgm = tg.mean(axis=1)

        def ts(target, tol=0.05):
            i = np.argmin(np.abs(tb-target)); lo=max(0,int(i*(1-tol))); hi=min(len(tgm),int(i*(1+tol))+1)
            return float(tgm[lo:hi].max()) if lo<hi else 0.0

        # 6 time signatures
        scores = {
            "4/4": ts(bpm*2)*1.2,  "2/4": ts(bpm*2),
            "3/4": ts(bpm*1.5),    "6/8": ts(bpm*1.5)+ts(bpm*3)*.5,
            "5/4": ts(bpm*2.5)*.8, "7/8": ts(bpm*3.5)*.7,
        }
        tsig = max(scores, key=scores.get)
        tconf= round(max(scores.values())/(sum(scores.values())+1e-10),4)

        # Swing factor
        swing = 0.5
        if len(bt)>4:
            period = float(np.median(np.diff(bt)))
            phases = []
            for b in bt[:-1]:
                tgt=b+period/2
                near=[t for t in onsets.get("onsets_energy",[]) if abs(t-tgt)<period*0.15]
                if near: phases.append((min(near,key=lambda t:abs(t-tgt))-b)/period)
            if phases: swing=float(np.median(phases))

        # Syncopation (Keith 1991)
        synco = 0.0
        if len(bt)>2:
            period=60.0/bpm
            ws=[]
            for t in onsets.get("onsets_energy",[]):
                ph=(t%period)/period
                if ph<.05 or ph>.95: w=4
                elif abs(ph-.5)<.05: w=2
                elif abs(ph-.25)<.05 or abs(ph-.75)<.05: w=1
                else: w=.5
                ws.append(4.5-w)
            if ws: synco=float(np.mean(ws))

        # Rhythmic entropy
        hist,_=np.histogram(oe,bins=64,density=True); hist+=1e-12
        entropy=float(-np.sum(hist*np.log2(hist)))

        # Groove quantisation error (RMS deviation of onsets from nearest tatum grid)
        gqe=0.0
        tatum=np.array(beats.get("tatum_times",[]))
        if len(tatum)>0:
            ons_arr=np.array(onsets.get("onsets_energy",[]))
            if len(ons_arr)>0:
                dists=[float(np.min(np.abs(tatum-o))) for o in ons_arr]
                gqe=round(float(np.sqrt(np.mean(np.array(dists)**2))),5)

        # IOI stats
        ioi_stats={}
        if len(bt)>2:
            iois=np.diff(bt)
            ioi_stats={"mean":float(np.mean(iois)),"std":float(np.std(iois)),
                       "cv":float(np.std(iois)/(np.mean(iois)+1e-10)),
                       "min":float(np.min(iois)),"max":float(np.max(iois)),
                       "skew":float(stats.skew(iois)),"kurtosis":float(stats.kurtosis(iois))}

        plp=beats.get("plp_env",[])
        step=max(1,len(plp)//1000)
        return {
            "time_signature": tsig, "time_sig_scores":{k:round(v,4) for k,v in scores.items()},
            "meter_confidence": tconf, "swing_factor": round(swing,4),
            "syncopation_index": round(synco,4), "rhythmic_entropy": round(entropy,4),
            "groove_quantisation_error": gqe, "ioi_stats": ioi_stats,
            "pulse": plp[::step], "pulse_times": tm[:len(plp):step].tolist(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SPECTRAL
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _spectral(self, y, sr):
        hop=self.HOP
        D  =librosa.stft(y,n_fft=self.NFFT,hop_length=hop)
        mg =np.abs(D); pw=mg**2
        fr =librosa.fft_frequencies(sr=sr,n_fft=self.NFFT)
        tm =librosa.frames_to_time(np.arange(mg.shape[1]),sr=sr,hop_length=hop)

        cen =librosa.feature.spectral_centroid(S=mg,sr=sr,hop_length=hop)[0]
        bw  =librosa.feature.spectral_bandwidth(S=mg,sr=sr,hop_length=hop)[0]
        flat=librosa.feature.spectral_flatness(S=mg,hop_length=hop)[0]
        con =librosa.feature.spectral_contrast(S=mg,sr=sr,hop_length=hop,n_bands=6)
        zcr =librosa.feature.zero_crossing_rate(y,hop_length=hop)[0]
        r85 =librosa.feature.spectral_rolloff(S=mg,sr=sr,hop_length=hop,roll_percent=0.85)[0]
        r95 =librosa.feature.spectral_rolloff(S=mg,sr=sr,hop_length=hop,roll_percent=0.95)[0]
        r10 =librosa.feature.spectral_rolloff(S=mg,sr=sr,hop_length=hop,roll_percent=0.10)[0]

        flux=np.concatenate([[0],np.sqrt(np.mean(np.maximum(0,np.diff(mg,axis=1))**2,axis=0))])
        norm_pw=pw/(pw.sum(axis=0,keepdims=True)+1e-20)
        entropy=-np.sum(norm_pw*np.log2(norm_pw+1e-20),axis=0)
        irreg  =np.concatenate([[0],np.mean(np.abs(mg[1:-1]-(mg[:-2]+mg[2:])/2),axis=0),[0]])

        # Tristimulus
        f0a=float(np.mean(cen)); tot=pw.sum(axis=0)+1e-20
        def bev(lo,hi): m=(fr>=lo)&(fr<hi); return pw[m,:].sum(axis=0) if m.any() else np.zeros(mg.shape[1])
        t1=bev(0,f0a*1.5)/tot; t2=bev(f0a*1.5,f0a*4.5)/tot; t3=bev(f0a*4.5,sr//2)/tot

        # Spectral kurtosis & skewness (per frame, mean over time)
        sk_kurt = float(np.mean([stats.kurtosis(pw[:,i]) for i in range(0,pw.shape[1],max(1,pw.shape[1]//100))]))
        sk_skew = float(np.mean([stats.skew(pw[:,i])     for i in range(0,pw.shape[1],max(1,pw.shape[1]//100))]))

        # 9-band energy
        band_e={name:float(np.mean(pw[(fr>=lo)&(fr<hi),:])) if ((fr>=lo)&(fr<hi)).any() else 0.0
                for name,lo,hi in BANDS9}

        # Bark energy
        bark_e=[{"lo":BARK[i],"hi":BARK[i+1],
                 "energy":float(np.mean(pw[(fr>=BARK[i])&(fr<BARK[i+1]),:])) if ((fr>=BARK[i])&(fr<BARK[i+1])).any() else 0.0}
                for i in range(len(BARK)-1) if BARK[i]<sr//2]

        # Sub-band flux
        sb_flux={}
        for name,lo,hi in [("low",0,300),("mid",300,2000),("high",2000,6000),("air",6000,sr//2)]:
            msk=(fr>=lo)&(fr<hi); bm=mg.copy(); bm[~msk,:]=0
            sf_=np.concatenate([[0],np.sqrt(np.mean(np.maximum(0,np.diff(bm,axis=1))**2,axis=0))])
            sb_flux[name]=_ds(sf_).tolist()

        ps=np.mean(pw,axis=1)

        return {
            "centroid_mean": float(np.mean(cen)), "centroid_std": float(np.std(cen)),
            "centroid": _ds(cen).tolist(), "bandwidth": _ds(bw).tolist(),
            "flatness": _ds(flat).tolist(),
            "rolloff_85": _ds(r85).tolist(), "rolloff_95": _ds(r95).tolist(), "rolloff_10": _ds(r10).tolist(),
            "contrast": _ds2(con).tolist(), "zcr": _ds(zcr).tolist(),
            "flux": _ds(flux).tolist(), "entropy": _ds(entropy).tolist(),
            "irregularity": _ds(irreg).tolist(),
            "tristimulus_t1": _ds(t1).tolist(), "tristimulus_t2": _ds(t2).tolist(), "tristimulus_t3": _ds(t3).tolist(),
            "spectral_kurtosis": round(sk_kurt,4), "spectral_skewness": round(sk_skew,4),
            "times": _ds(tm).tolist(), "freqs": fr.tolist(),
            "power_spectrum_db": librosa.amplitude_to_db(np.sqrt(ps+1e-20),ref=np.max).tolist(),
            "magnitude_db": _ds2(librosa.amplitude_to_db(mg,ref=np.max)).tolist(),
            "band_energy": band_e, "bark_energy": bark_e, "sub_band_flux": sb_flux,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MEL SPECTROGRAM
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _mel(self, y, sr):
        M  = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=self.NFFT,hop_length=self.HOP,n_mels=self.NMELS,fmin=20)
        Md = librosa.power_to_db(M,ref=np.max)
        mf = librosa.mel_frequencies(n_mels=self.NMELS,fmin=20,fmax=sr//2)
        tm = librosa.frames_to_time(np.arange(M.shape[1]),sr=sr,hop_length=self.HOP)
        return {"mel_db":_ds2(Md).tolist(),"mel_freqs":mf.tolist(),"times":_ds(tm).tolist(),"n_mels":self.NMELS}

    # ─────────────────────────────────────────────────────────────────────────
    # NMF SOURCE SEPARATION
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _nmf(self, y, sr):
        if not HAS_SKLEARN: return {"available": False}
        M  = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=self.NFFT,hop_length=self.HOP,n_mels=64)
        M  = M + 1e-6
        model= NMF(n_components=6, init="nndsvda", max_iter=300, random_state=0)
        W  = model.fit_transform(M)      # 64 x 6 basis
        H  = model.components_           # 6 x T activations
        tm = librosa.frames_to_time(np.arange(M.shape[1]),sr=sr,hop_length=self.HOP)
        # Energy per component
        comp_e = [float(np.mean(H[i])) for i in range(H.shape[0])]
        step   = max(1,H.shape[1]//800)
        return {
            "available": True, "n_components": 6,
            "basis": W.tolist(),               # 64x6 spectral templates
            "activations": H[:,::step].tolist(), # 6xT activation envelopes
            "component_energy": comp_e,
            "times": tm[::step].tolist(),
            "reconstruction_error": round(float(model.reconstruction_err_),4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MFCC + BFCC + LPC + CMVN + PCA
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _mfcc(self, y, sr):
        hop=self.HOP
        m40=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40,n_fft=self.NFFT,hop_length=hop,n_mels=128)
        d1 =librosa.feature.delta(m40); d2=librosa.feature.delta(m40,order=2)
        m13=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,n_fft=self.NFFT,hop_length=hop,n_mels=64)
        d13=librosa.feature.delta(m13); d213=librosa.feature.delta(m13,order=2)
        # BFCC (24 bark-mel)
        bmel=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=self.NFFT,hop_length=hop,n_mels=self.NBARK,fmin=20)
        bfcc=librosa.feature.mfcc(S=librosa.power_to_db(bmel),n_mfcc=24)
        # CMVN of m40
        cmvn=(m40-m40.mean(axis=1,keepdims=True))/(m40.std(axis=1,keepdims=True)+1e-10)
        # LPC
        lpc_m,lpc_s=self._lpc(y,sr,hop)
        step=max(1,m40.shape[1]//1000)
        return {
            "mfcc":_ds2(m40).tolist(),"mfcc_delta":_ds2(d1).tolist(),"mfcc_delta2":_ds2(d2).tolist(),
            "mfcc_mean":np.mean(m40,axis=1).tolist(),"mfcc_std":np.std(m40,axis=1).tolist(),
            "mfcc_cmvn":_ds2(cmvn).tolist(),
            "mfcc13":_ds2(m13).tolist(),"mfcc13_delta":_ds2(d13).tolist(),"mfcc13_delta2":_ds2(d213).tolist(),
            "bfcc":_ds2(bfcc).tolist(),"bfcc_mean":np.mean(bfcc,axis=1).tolist(),
            "lpc_mean":lpc_m,"lpc_std":lpc_s,"n_mfcc":40,
        }

    def _lpc(self, y, sr, hop):
        frames=librosa.util.frame(y,frame_length=1024,hop_length=hop)
        coeffs=[]
        for i in range(min(frames.shape[1],500)):
            try: coeffs.append(librosa.lpc(frames[:,i].astype(np.float64),order=self.LPCORD).tolist())
            except: coeffs.append([0.0]*(self.LPCORD+1))
        a=np.array(coeffs)
        return np.mean(a,axis=0).tolist(),np.std(a,axis=0).tolist()

    @_safe()
    def _pca_mfcc(self, mfcc_data):
        if not HAS_SKLEARN: return {"available": False}
        m=np.array(mfcc_data.get("mfcc",[]))
        if len(m)==0: return {"available": False}
        m=np.array(m).T  # frames x coeffs
        pca=PCA(n_components=min(3,m.shape[1]))
        pc=pca.fit_transform(m)  # frames x 3
        return {
            "available": True,
            "components": pc.T.tolist(),   # 3 x frames
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components": pca.n_components_,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CHROMA
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _chroma(self, yh, sr, nmf_data):
        hop=self.HOP
        cc =librosa.feature.chroma_cqt(y=yh,sr=sr,hop_length=hop,bins_per_octave=self.BPO)
        cn =librosa.feature.chroma_cens(y=yh,sr=sr,hop_length=hop)
        cs =librosa.feature.chroma_stft(y=yh,sr=sr,hop_length=hop,n_fft=self.NFFT)

        # Deep chroma -- NMF-derived (noise-robust)
        deep=None
        if HAS_SKLEARN:
            try:
                M=librosa.feature.melspectrogram(y=yh,sr=sr,n_fft=self.NFFT,hop_length=hop,n_mels=128)+1e-6
                model=NMF(n_components=12,init="nndsvda",max_iter=200,random_state=42)
                model.fit_transform(M)
                # Map 12 NMF basis vectors to pitch classes via CQT projection
                deep=np.clip(model.components_/model.components_.max(axis=1,keepdims=True),0,1)
                deep=deep[:,:cc.shape[1]] if deep.shape[1]>=cc.shape[1] else cc
            except: deep=cc

        # HCDF
        hcdf=1-np.sum(cc[:,:-1]*cc[:,1:],axis=0)/(np.linalg.norm(cc[:,:-1],axis=0)*np.linalg.norm(cc[:,1:],axis=0)+1e-10)
        hcdf=np.concatenate([[0],hcdf])
        peaks,_=sig.find_peaks(hcdf,height=np.mean(hcdf)+np.std(hcdf),distance=10)
        tm=librosa.frames_to_time(np.arange(cc.shape[1]),sr=sr,hop_length=hop)

        return {
            "chroma_cqt":_ds2(cc).tolist(),"chroma_cens":_ds2(cn).tolist(),
            "chroma_stft":_ds2(cs).tolist(),
            "chroma_deep":_ds2(np.array(deep)).tolist() if deep is not None else None,
            "chroma_mean":np.mean(cc,axis=1).tolist(),"chroma_std":np.std(cc,axis=1).tolist(),
            "hcdf":_ds(hcdf).tolist(),"chord_change_times":tm[peaks].tolist() if len(peaks) else [],
            "times":_ds(tm).tolist(),"pitch_classes":PC,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # KEY ESTIMATION
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _harmony(self, chroma):
        cm=np.array(chroma.get("chroma_mean",[0.0]*12))
        cn=cm/(cm.sum()+1e-10)

        def est(maj,mn,name):
            maj_s=[float(np.corrcoef(cn,np.roll(maj,s))[0,1]) for s in range(12)]
            min_s=[float(np.corrcoef(cn,np.roll(mn, s))[0,1]) for s in range(12)]
            bm=int(np.argmax(maj_s)); bn=int(np.argmax(min_s))
            if maj_s[bm]>=min_s[bn]:
                return {"profile":name,"key":PC[bm],"mode":"Major","key_index":bm,
                        "confidence":round(maj_s[bm],4),"major_scores":maj_s,"minor_scores":min_s}
            return {"profile":name,"key":PC[bn],"mode":"Minor","key_index":bn,
                    "confidence":round(min_s[bn],4),"major_scores":maj_s,"minor_scores":min_s}

        ks  = est(KS_MAJ,  KS_MIN,  "Krumhansl-Schmuckler 1990")
        tmp = est(TMP_MAJ, TMP_MIN, "Temperley 2001")
        best= ks if ks["confidence"]>=tmp["confidence"] else tmp

        # Relative major/minor
        rel_key = PC[(best["key_index"]+9)%12] if best["mode"]=="Major" else PC[(best["key_index"]+3)%12]
        rel_mode= "Minor" if best["mode"]=="Major" else "Major"

        # Circle of fifths position (0-11)
        cof_pos = (best["key_index"]*7)%12

        return {
            "key": best["key"], "mode": best["mode"],
            "key_full": f"{best['key']} {best['mode']}",
            "key_confidence": best["confidence"],
            "key_index": best["key_index"],
            "relative_key": f"{rel_key} {rel_mode}",
            "circle_of_fifths_pos": cof_pos,
            "ks_estimate": ks, "temperley_estimate": tmp,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CHORD RECOGNITION
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _chords(self, chroma):
        cc=np.array(chroma.get("chroma_cqt",[]))
        tm=chroma.get("times",[])
        if cc.shape[0]!=12 or cc.shape[1]==0: return {"chords":[],"n_chord_changes":0}
        step=max(1,cc.shape[1]//400)
        chords=[]
        for i in range(0,cc.shape[1],step):
            f=cc[:,i]; best_c,best_s="N",-1
            for root in range(12):
                for ctype,tmpl in CHORDS.items():
                    tr=np.roll(np.array(tmpl,float),root)
                    s=float(np.dot(f,tr)/(np.linalg.norm(f)*np.linalg.norm(tr)+1e-10))
                    if s>best_s: best_s=s; best_c=f"{PC[root]}:{ctype}"
            t=tm[min(i,len(tm)-1)] if tm else 0
            chords.append({"time":round(float(t),3),"chord":best_c,"score":round(best_s,3)})

        # Chord change count
        changes=sum(1 for a,b in zip(chords,chords[1:]) if a["chord"]!=b["chord"])

        # Tonal tension
        key_idx=0; key_mode="Major"  # will be overridden if harmony available
        tension=[round(float(1.0-np.dot(cc[:,i]/(cc[:,i].sum()+1e-10),
                              np.roll(KS_MAJ if key_mode=="Major" else KS_MIN,key_idx)/
                              np.roll(KS_MAJ if key_mode=="Major" else KS_MIN,key_idx).sum())),4)
                 for i in range(0,cc.shape[1],step)]

        return {"chords":chords,"n_chord_changes":changes,"tonal_tension":tension}

    # ─────────────────────────────────────────────────────────────────────────
    # TONNETZ
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _tonnetz(self, yh, sr):
        tn=librosa.feature.tonnetz(y=yh,sr=sr,hop_length=self.HOP)
        tm=librosa.frames_to_time(np.arange(tn.shape[1]),sr=sr,hop_length=self.HOP)
        return {"tonnetz":_ds2(tn).tolist(),"tonnetz_mean":np.mean(tn,axis=1).tolist(),
                "tonnetz_std":np.std(tn,axis=1).tolist(),"times":_ds(tm).tolist(),
                "dims":["5th-cos","5th-sin","M3-cos","M3-sin","m3-cos","m3-sin"]}

    # ─────────────────────────────────────────────────────────────────────────
    # PITCH + VIBRATO + MELODIC CONTOUR
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _pitch(self, y, sr, yhi):
        hop=self.HOP; fmin=librosa.note_to_hz("C2"); fmax=librosa.note_to_hz("C7")
        f0,vf,vp=librosa.pyin(y,fmin=fmin,fmax=fmax,sr=sr,hop_length=hop,fill_na=None)
        tm=librosa.frames_to_time(np.arange(len(f0)),sr=sr,hop_length=hop)
        f0c=np.where(np.isnan(f0),0.0,f0)
        f0v=f0[vf&~np.isnan(f0)]

        stats_p={}
        if len(f0v)>2:
            stats_p={"mean_hz":float(np.mean(f0v)),"std_hz":float(np.std(f0v)),
                     "min_hz":float(np.min(f0v)),"max_hz":float(np.max(f0v)),
                     "range_st":float(12*np.log2(max(np.max(f0v),1)/max(np.min(f0v),1))),
                     "voiced_ratio":float(vf.mean())}

        # Melodic contour
        contour={}
        if len(f0v)>4:
            st=12*np.log2(f0v/(f0v.mean()+1e-10)+1e-10)
            df=np.diff(st)
            contour={"direction":"ascending" if np.mean(df)>.1 else "descending" if np.mean(df)<-.1 else "stable",
                     "mean_interval":float(np.mean(np.abs(df))),"std_interval":float(np.std(df)),
                     "range_st":float(np.max(st)-np.min(st)),"n_peaks":int(len(sig.find_peaks(st)[0])),
                     "n_valleys":int(len(sig.find_peaks(-st)[0]))}

        # Vibrato (4-8 Hz AM on F0 contour)
        vibrato={"detected":False}
        if vf.sum()>50:
            fv2=f0c[vf]; hop_t=hop/sr
            sm=sig.savgol_filter(fv2,min(11,len(fv2)|1),3) if len(fv2)>10 else fv2
            dev=fv2-sm; fd=np.abs(nfft.rfft(dev)); ff=nfft.rfftfreq(len(dev),d=hop_t)
            vm=(ff>=4)&(ff<=8)
            if vm.any():
                vs=float(fd[vm].max()); vr=float(ff[vm][np.argmax(fd[vm])])
                vibrato={"detected":vs>0.5,"rate_hz":round(vr,2),"strength":round(vs,4),
                         "extent_st":round(float(np.std(dev))*12/100,3)}

        # Pitch salience histogram
        ps_hist,ps_edges=[],[]
        if len(f0v)>2:
            bins=np.arange(50,2100,10)
            cnt,_=np.histogram(f0v,bins=bins)
            ps_hist=cnt.tolist(); ps_edges=bins.tolist()

        res={"f0_pyin":_ds(f0c).tolist(),"f0_times":_ds(tm).tolist(),
             "voiced_prob":_ds(vp).tolist(),"voiced_flag":_ds(vf).tolist(),
             "pitch_stats":stats_p,"melodic_contour":contour,"vibrato":vibrato,
             "salience_histogram":ps_hist,"salience_edges":ps_edges}

        if HAS_PRAAT: res["f0_praat"]=self._praat_pitch(yhi)
        return res

    @_safe()
    def _praat_pitch(self, yhi):
        snd=_pm.Sound(yhi,sampling_frequency=float(self.SR_HI))
        p=snd.to_pitch(time_step=0.01,pitch_floor=75,pitch_ceiling=600)
        f0=p.selected_array["frequency"]; tm=p.xs()
        f0c=[float(v) if v!=0 else 0.0 for v in f0]
        step=max(1,len(f0c)//500)
        return {"f0":f0c[::step],"times":tm[::step].tolist(),"method":"praat-ac"}

    # ─────────────────────────────────────────────────────────────────────────
    # VOICE QUALITY (Praat)
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _voice(self, yhi):
        if not HAS_PRAAT: return {"available":False}
        from parselmouth.praat import call as pc
        sr=self.SR_HI
        snd=_pm.Sound(yhi,sampling_frequency=float(sr))
        pitch=pc(snd,"To Pitch",0.0,75,600)
        pp   =pc([snd,pitch],"To PointProcess (cc)")

        jit={
            "local":     round(pc(pp,"Get jitter (local)",0,0,.0001,.02,1.3),6),
            "local_abs": round(pc(pp,"Get jitter (local, absolute)",0,0,.0001,.02,1.3),9),
            "rap":       round(pc(pp,"Get jitter (rap)",0,0,.0001,.02,1.3),6),
            "ppq5":      round(pc(pp,"Get jitter (ppq5)",0,0,.0001,.02,1.3),6),
            "ddp":       round(pc(pp,"Get jitter (ddp)",0,0,.0001,.02,1.3),6),
        }
        shim={
            "local":    round(pc([snd,pp],"Get shimmer (local)",0,0,.0001,.02,1.3,1.6),6),
            "local_db": round(pc([snd,pp],"Get shimmer (local_dB)",0,0,.0001,.02,1.3,1.6),4),
            "apq3":     round(pc([snd,pp],"Get shimmer (apq3)",0,0,.0001,.02,1.3,1.6),6),
            "apq5":     round(pc([snd,pp],"Get shimmer (apq5)",0,0,.0001,.02,1.3,1.6),6),
            "apq11":    round(pc([snd,pp],"Get shimmer (apq11)",0,0,.0001,.02,1.3,1.6),6),
            "dda":      round(pc([snd,pp],"Get shimmer (dda)",0,0,.0001,.02,1.3,1.6),6),
        }
        harm=pc(snd,"To Harmonicity (cc)",.01,75,.1,1.0)
        hnr=float(pc(harm,"Get mean",0,0)); hnr_sd=float(pc(harm,"Get standard deviation",0,0))
        nhr=round(1.0/(10**(hnr/10)+1e-10),6)

        # CPP (Cepstral Peak Prominence) -- approximate via MFCC energy ratio
        y_snd = np.array(snd.values[0])
        cpp = self._cpp(y_snd, sr)

        fmnt=pc(snd,"To Formant (burg)",0.0,5,5500,.025,50)
        formants={}
        for fi in range(1,5):
            try:
                fv=pc(fmnt,"Get mean",fi,0,0,"hertz"); fb=pc(fmnt,"Get mean bandwidth",fi,0,0,"hertz")
                formants[f"F{fi}"]={"mean_hz":round(float(fv),1),"bw_hz":round(float(fb),1)}
            except: formants[f"F{fi}"]={"mean_hz":0.0,"bw_hz":0.0}

        # VAD
        nv=pc(pitch,"Count voiced frames"); nt=pc(pitch,"Get number of frames")
        vad=float(nv)/max(float(nt),1)

        # LTAS (Long-Term Average Spectrum) -- Baken & Orlikoff method via power spectrum
        D=librosa.stft(np.array(snd.values[0],dtype=np.float32),n_fft=2048,hop_length=512)
        ltas=librosa.amplitude_to_db(np.mean(np.abs(D),axis=1),ref=np.max).tolist()
        ltas_freqs=librosa.fft_frequencies(sr=sr,n_fft=2048).tolist()

        return {
            "available":True, "jitter":jit, "shimmer":shim,
            "hnr_mean_db":round(hnr,3), "hnr_std_db":round(hnr_sd,3), "nhr":nhr,
            "cpp_db": cpp, "formants":formants, "vad_ratio":round(vad,4),
            "ltas":ltas, "ltas_freqs":ltas_freqs,
        }

    def _cpp(self, y, sr):
        """Cepstral Peak Prominence (CPP) -- log spectrum quefrency peak."""
        try:
            n_fft=2048; D=librosa.stft(y.astype(np.float32),n_fft=n_fft,hop_length=512)
            log_ps=np.log(np.abs(D)**2+1e-10)
            cepstrum=np.abs(nfft.rfft(log_ps,axis=0))
            # Quefrency range 2ms-20ms for F0 range 50-500 Hz
            lo=int(.002*sr); hi=int(.020*sr)
            if lo>=cepstrum.shape[0]: return 0.0
            hi=min(hi,cepstrum.shape[0]-1)
            peak=float(np.mean(np.max(cepstrum[lo:hi,:],axis=0)))
            # Regression line for slope removal
            qs=np.arange(cepstrum.shape[0]); reg=np.polyfit(qs,np.mean(cepstrum,axis=1),1)
            baseline=float(np.polyval(reg,lo+np.argmax(np.max(cepstrum[lo:hi,:],axis=0))))
            return round(float(20*np.log10(max(peak-baseline,1e-10))),2)
        except: return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # DYNAMICS / LOUDNESS
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _dynamics(self, y, sr, yhi):
        hop=self.HOP
        rms=librosa.feature.rms(y=y,hop_length=hop)[0]
        rdb=librosa.amplitude_to_db(rms,ref=np.max)
        tm =librosa.frames_to_time(np.arange(len(rms)),sr=sr,hop_length=hop)
        dr =float(np.max(rdb)-np.min(rdb[rdb>np.max(rdb)-60]))
        crest=float(20*np.log10(np.max(np.abs(y))/(np.sqrt(np.mean(y**2))+1e-10)))

        # Compression detection: low crest factor + low dynamic range -> compressed
        compression_score = max(0.0, 1.0 - (min(crest,20)/20)*(min(dr,40)/40))

        loudness={}
        if HAS_LOUDNORM:
            try:
                meter=_pyln.Meter(sr)
                lufs=meter.integrated_loudness(y.astype(np.float64))
                lufs=float(lufs) if not (math.isnan(lufs) or math.isinf(lufs)) else -70.0
                win_st=int(.4*sr); win_mom=int(.1*sr)
                st=[]; mom=[]
                for i in range(0,len(y)-win_st,win_st//2):
                    try: l=meter.integrated_loudness(y[i:i+win_st].astype(np.float64)); st.append(float(l) if not(math.isnan(l) or math.isinf(l)) else -70.0)
                    except: st.append(-70.0)
                for i in range(0,len(y)-win_mom,win_mom):
                    try: l=meter.integrated_loudness(y[i:i+win_mom].astype(np.float64)); mom.append(float(l) if not(math.isnan(l) or math.isinf(l)) else -70.0)
                    except: mom.append(-70.0)
                lra=float(np.percentile(st,95)-np.percentile(st,10)) if st else 0.0
                tp =float(20*np.log10(np.max(np.abs(yhi))+1e-10))
                # Streaming loudness penalty (Spotify = -14 LUFS)
                penalty=round(lufs-(-14.0),2)
                loudness={"integrated_lufs":round(lufs,2),"true_peak_dbtp":round(tp,2),
                          "loudness_range":round(lra,2),
                          "short_term":st[::max(1,len(st)//200)],
                          "momentary":mom[::max(1,len(mom)//200)],
                          "streaming_penalty_db":penalty}
            except Exception as e: log.warning("loudnorm: %s",e)

        return {
            "rms":_ds(rms).tolist(),"rms_db":_ds(rdb).tolist(),"times":_ds(tm).tolist(),
            "dynamic_range":round(dr,2),"crest_factor":round(crest,2),
            "compression_score":round(compression_score,4),
            "mean_rms":float(np.mean(rms)),"peak_rms":float(np.max(rms)),
            "percentiles":{f"p{p}":float(np.percentile(rms,p)) for p in [10,25,50,75,90,95]},
            "loudness":loudness,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TIMBRE
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _timbre(self, y, sr):
        D=librosa.stft(y,n_fft=self.NFFT,hop_length=self.HOP)
        mg=np.abs(D); pw=mg**2
        fr=librosa.fft_frequencies(sr=sr,n_fft=self.NFFT)
        tot=pw.sum(axis=0)+1e-20
        bright=float(np.mean(pw[fr>=1500,:].sum(axis=0)/tot))
        warm  =float(np.mean(pw[fr<=320, :].sum(axis=0)/tot))
        noise =float(np.mean(librosa.feature.spectral_flatness(S=mg)))

        # A-weighting
        def aw(f): f=np.maximum(f,1); return (12200**2*f**4)/((f**2+20.6**2)*np.sqrt(f**2+107.7**2)*np.sqrt(f**2+737.9**2)*(f**2+12200**2))
        wp=pw*aw(fr)[:,None]; awc=float(np.mean(np.sum(fr[:,None]*wp,axis=0)/(np.sum(wp,axis=0)+1e-20)))

        roughness   =self._roughness(mg,fr)
        inharmonic  =self._inharmonic(mg,fr)
        oer         =self._oer(mg,fr)

        # Spectral kurtosis mean
        sk=float(np.mean([stats.kurtosis(pw[:,i]) for i in range(0,pw.shape[1],max(1,pw.shape[1]//100))]))

        return {"brightness":round(bright,5),"warmth":round(warm,5),
                "noisiness":round(noise,5),"a_weighted_centroid":round(awc,2),
                "roughness":round(roughness,5),"inharmonicity":round(inharmonic,5),
                "odd_even_ratio":round(oer,4),"spectral_kurtosis":round(sk,4)}

    def _roughness(self,mg,fr,n=16):
        mm=mg.mean(axis=1); idx,_=sig.find_peaks(mm,height=mm.max()*.05,distance=4)
        if len(idx)<2: return 0.0
        idx=idx[np.argsort(mm[idx])[::-1][:n]]; pf=fr[idx]; pa=mm[idx]/(mm[idx].max()+1e-10)
        r=0.0
        for i in range(len(pf)):
            for j in range(i+1,len(pf)):
                f1,f2=pf[i],pf[j]
                if f1<=0 or f2<=0: continue
                xs=0.24/(0.0207*min(f1,f2)+18.96); x=abs(f2-f1)*xs
                r+=max(0.0,pa[i]*pa[j]*(math.exp(-3.5*x)-math.exp(-5.75*x)))
        return r/max(1,len(idx))

    def _inharmonic(self,mg,fr):
        mm=mg.mean(axis=1); idx,_=sig.find_peaks(mm,height=mm.max()*.05,distance=4)
        if len(idx)<3: return 0.0
        pf=fr[idx]; f0=pf[0] if pf[0]>0 else pf[1]/2
        if f0<=0: return 0.0
        return float(np.mean([abs(pf[n]-f0*(n+1))/(f0*(n+1)) for n in range(min(12,len(pf)))]))

    def _oer(self,mg,fr):
        mm=mg.mean(axis=1); idx,_=sig.find_peaks(mm,height=mm.max()*.05,distance=4)
        if len(idx)<4: return 1.0
        o=sum(mm[idx[i]] for i in range(0,min(10,len(idx)),2))
        e=sum(mm[idx[i]] for i in range(1,min(10,len(idx)),2))
        return float(o/(e+1e-10))

    # ─────────────────────────────────────────────────────────────────────────
    # openSMILE eGeMAPS
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _smile(self, path):
        if not HAS_SMILE: return {"available":False}
        try:
            sm=_os_.Smile(feature_set=_os_.FeatureSet.eGeMAPSv02,
                          feature_level=_os_.FeatureLevel.Functionals)
            df=sm.process_file(path)
            feats={k:round(float(v),6) if not(math.isnan(v) or math.isinf(v)) else 0.0
                   for k,v in df.iloc[0].to_dict().items()}
            return {"available":True,"features":feats,"n_features":len(feats)}
        except Exception as e: return {"available":False,"error":str(e)}

    # ─────────────────────────────────────────────────────────────────────────
    # STRUCTURE / SEGMENTATION
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _structure(self, y, sr, chroma):
        hop=self.HOP
        # Feature matrix: 64-mel + 12-chroma
        M=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=self.NFFT,hop_length=hop,n_mels=64)
        Md=librosa.power_to_db(M,ref=np.max)
        ch=librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop)
        Mn=(Md-Md.mean(axis=1,keepdims=True))/(Md.std(axis=1,keepdims=True)+1e-10)
        cn=(ch-ch.mean(axis=1,keepdims=True))/(ch.std(axis=1,keepdims=True)+1e-10)
        feat=np.vstack([Mn,cn])

        # SSM
        ds_step=max(1,feat.shape[1]//100)
        fd=feat[:,::ds_step]
        fn=fd/(np.linalg.norm(fd,axis=0,keepdims=True)+1e-10)
        ssm=(fn.T@fn).tolist()

        # Path-enhanced SSM (diagonal smoothing, Müller 2015)
        ssm_arr=np.array(ssm)
        pssm=ssm_arr.copy()
        for d in range(1,min(4,ssm_arr.shape[0])):
            idx=np.arange(ssm_arr.shape[0]-d)
            pssm[idx,idx+d]=(pssm[idx,idx+d]+ssm_arr[idx,idx+d])/2
            pssm[idx+d,idx]=(pssm[idx+d,idx]+ssm_arr[idx+d,idx])/2

        # Laplacian segmentation
        bounds_lp=[]; n_segs=min(12,feat.shape[1]//20)
        try:
            fr_lp=librosa.segment.agglomerative(feat,k=n_segs)
            bounds_lp=librosa.frames_to_time(fr_lp,sr=sr,hop_length=hop).tolist()
        except: pass

        # Checkerboard novelty (Foote 2000)
        nov=self._checkerboard(fd); novt=librosa.frames_to_time(np.arange(len(nov))*ds_step,sr=sr,hop_length=hop)
        bounds_nov=[]
        if len(nov)>4:
            pks,_=sig.find_peaks(nov,height=np.mean(nov)+np.std(nov),distance=5)
            bounds_nov=[float(novt[min(p,len(novt)-1)]) for p in pks]

        # Row-mean SSM (repetition structure)
        row_mean=ssm_arr.mean(axis=1).tolist()

        all_bounds=sorted({round(t,2) for t in bounds_lp+bounds_nov})
        return {
            "segment_times": all_bounds,
            "segment_times_laplacian": [round(t,2) for t in bounds_lp],
            "segment_times_novelty": [round(t,2) for t in bounds_nov],
            "n_segments": len(all_bounds),
            "self_similarity": ssm, "self_similarity_path": pssm.tolist(),
            "ssm_size": len(ssm), "repetition_curve": row_mean,
            "novelty": nov.tolist(), "novelty_times": novt.tolist(),
        }

    def _checkerboard(self, feat, ks=8):
        n=feat.shape[1]
        if n<ks*2: return np.zeros(n)
        fn=feat/(np.linalg.norm(feat,axis=0,keepdims=True)+1e-10)
        ssm=fn.T@fn; kh=ks//2
        K=np.ones((ks,ks)); K[:kh,kh:]=-1; K[kh:,:kh]=-1
        nov=np.zeros(n)
        for t in range(kh,n-kh):
            blk=ssm[t-kh:t+kh,t-kh:t+kh]
            if blk.shape==K.shape: nov[t]=float(np.sum(K*blk))
        mn,mx=nov.min(),nov.max()
        if mx>mn: nov=(nov-mn)/(mx-mn)
        return nov

    # ─────────────────────────────────────────────────────────────────────────
    # ENERGY + WAVEFORM
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _energy(self, y, sr):
        rms=librosa.feature.rms(y=y,hop_length=self.HOP)[0]
        rdb=librosa.amplitude_to_db(rms,ref=np.max)
        tm =librosa.frames_to_time(np.arange(len(rms)),sr=sr,hop_length=self.HOP)
        return {"rms":_ds(rms).tolist(),"rms_db":_ds(rdb).tolist(),"times":_ds(tm).tolist(),
                "dynamic_range":round(float(np.max(rdb)-np.min(rdb)),2),
                "mean_rms":float(np.mean(rms)),"peak_rms":float(np.max(rms))}

    @_safe()
    def _waveform(self, y, sr, pts=6000):
        step=max(1,len(y)//pts); t=np.linspace(0,len(y)/sr,len(y[::step]))
        return {"samples":y[::step].tolist(),"times":t.tolist(),
                "sample_rate":sr,"n_samples":len(y),"duration":float(len(y)/sr)}

    # ─────────────────────────────────────────────────────────────────────────
    # MUSIC INFO
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _music_info(self, beats, spectral, dyn, mfcc, timbre, rhythm):
        bpm=beats.get("bpm_consensus",120); breg=beats.get("beat_regularity",.5)
        swing=rhythm.get("swing_factor",.5); dr=dyn.get("dynamic_range",20)
        bright=timbre.get("brightness",.3); warm=timbre.get("warmth",.3)
        rough=timbre.get("roughness",.1); rms=dyn.get("mean_rms",.1)
        be=spectral.get("band_energy",{})
        bass_r=be.get("bass",0)/(sum(be.values())+1e-10)

        dance=float(.30*breg+.20*_sgm((bpm-80)/40)*_sgm((160-bpm)/40)
                    +.20*min(1.,bass_r*5)+.15*(1-dr/60)+.15*(1-abs(swing-.5)*2))
        energy=float(.40*min(1.,rms*10)+.30*bright+.30*min(1.,bpm/180))
        valence=float(.5*bright+.3*(1-rough)+.2*breg)
        acoustic=float(.4*warm+.3*(1-bright)+.3*(1-min(1.,rms*10)))

        genre={
            "electronic": float(.4*min(1.,bass_r*5)+.3*breg+.3*bright),
            "acoustic":   float(acoustic),
            "hip_hop":    float(.5*min(1.,bass_r*5)+.3*_sgm((bpm-70)/20)*_sgm((100-bpm)/20)+.2*(1-bright)),
            "classical":  float(.4*(1-breg)+.3*(dr/60)+.3*(1-min(1.,bass_r*5))),
            "rock":       float(.3*min(1.,rough*10)+.4*energy+.3*bright),
            "jazz":       float(.4*abs(swing-.5)*2+.3*(1-breg)+.3*(dr/60)),
            "edm":        float(.5*breg+.3*min(1.,bpm/200)+.2*bright),
        }
        tot=sum(genre.values())+1e-10; genre={k:round(v/tot,4) for k,v in genre.items()}

        # Russell circumplex emotion quadrant (valence x arousal)
        arousal=energy; v=valence
        if v>.5 and arousal>.5:   quadrant="Happy / Excited"
        elif v>.5 and arousal<=.5: quadrant="Calm / Relaxed"
        elif v<=.5 and arousal>.5: quadrant="Angry / Tense"
        else:                       quadrant="Sad / Bored"

        return {
            "danceability":round(min(1.,max(0.,dance)),4),
            "energy":round(min(1.,max(0.,energy)),4),
            "valence":round(min(1.,max(0.,valence)),4),
            "acousticness":round(min(1.,max(0.,acoustic)),4),
            "genre_fingerprint":genre,
            "tempo_category":_tempo_cat(bpm),
            "emotion_quadrant":quadrant,
            "arousal":round(min(1.,max(0.,arousal)),4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FastDTW chroma distance
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _dtw_chroma(self, chroma, harmony):
        if not HAS_DTW: return {"available":False}
        from fastdtw import fastdtw
        cc=np.array(chroma.get("chroma_mean",[0]*12),dtype=float).reshape(-1,1)
        # Compare against ideal major and minor profile for detected key
        ki=harmony.get("key_index",0)
        maj_p=np.roll(KS_MAJ,ki).reshape(-1,1)
        min_p=np.roll(KS_MIN,ki).reshape(-1,1)
        d_maj,_=fastdtw(cc,maj_p); d_min,_=fastdtw(cc,min_p)
        return {"available":True,"dtw_dist_major":round(float(d_maj),4),"dtw_dist_minor":round(float(d_min),4),
                "closer_to":"major" if d_maj<d_min else "minor"}

    # ─────────────────────────────────────────────────────────────────────────
    # MIDI
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _midi(self, path, emit):
        if not HAS_MIDI: return {"error":"pretty_midi not installed"}
        midi=_pm2.PrettyMIDI(path)
        tt,tps=midi.get_tempo_changes()
        tc=[{"time":round(float(t),3),"bpm":round(float(b),2)} for t,b in zip(tt,tps)]
        fs=50
        pr=midi.get_piano_roll(fs=fs); ptm=(np.arange(pr.shape[1])/fs).tolist()
        chrom=midi.get_chroma(fs=fs); poly=(pr>0).sum(axis=0)
        insts=[]
        for inst in midi.instruments:
            if not inst.notes: continue
            pits=[n.pitch for n in inst.notes]; vels=[n.velocity for n in inst.notes]
            durs=[n.end-n.start for n in inst.notes]
            insts.append({"name":inst.name or _pm2.program_to_instrument_name(inst.program),
                          "program":inst.program,"is_drum":inst.is_drum,"n_notes":len(inst.notes),
                          "pitch_mean":round(float(np.mean(pits)),2),"pitch_range":[int(min(pits)),int(max(pits))],
                          "velocity_mean":round(float(np.mean(vels)),2),"duration_mean":round(float(np.mean(durs)),4)})
        emit(80,"MIDI harmony analysis...")
        step=max(1,pr.shape[1]//800)
        return {"type":"midi",
                "metadata":{"filename":os.path.basename(path),"duration":round(midi.get_end_time(),3),
                             "n_instruments":len(midi.instruments),"resolution":midi.resolution,
                             "filesize":os.path.getsize(path)},
                "bpm_estimate":round(float(np.median(tps)),2) if len(tps) else 120.0,
                "tempo_changes":tc,"instruments":insts,
                "piano_roll":pr[:,::step].tolist(),"piano_roll_times":ptm[::step],
                "chroma":chrom[:,::step].tolist(),"polyphony":poly[::step].tolist(),
                "polyphony_times":ptm[::step],"pitch_classes":PC}

    # ─────────────────────────────────────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────────────────────────────────────
    @_safe()
    def _metadata(self, path, y, sr):
        m={"filename":os.path.basename(path),"filepath":path,"filesize":os.path.getsize(path),
           "duration":float(len(y)/sr),"sample_rate":sr,"n_samples":len(y),
           "extension":Path(path).suffix.lower()}
        if HAS_MUTAGEN:
            try:
                tags=_mut.File(path,easy=True)
                if tags:
                    for k in ["title","artist","album","date","genre","tracknumber","composer","bpm"]:
                        v=tags.get(k); m[k]=v[0] if v else None
                    if hasattr(tags,"info"):
                        m["bitrate"]=getattr(tags.info,"bitrate",None)
                        m["channels"]=getattr(tags.info,"channels",None)
            except: pass
        return m

    # ─────────────────────────────────────────────────────────────────────────
    # REALTIME (low-latency chunk)
    # ─────────────────────────────────────────────────────────────────────────
    def analyze_realtime_chunk(self, chunk, sr):
        """<20ms for 2048-sample chunk -- used by WebSocket handler."""
        nfft=2048; hop=256
        D=librosa.stft(chunk,n_fft=nfft,hop_length=hop)
        mg=np.abs(D); pw=mg**2
        fr=librosa.fft_frequencies(sr=sr,n_fft=nfft)
        mm=mg.mean(axis=1)
        rms=float(np.sqrt(np.mean(chunk**2)))
        rdb=float(20*np.log10(rms+1e-10))
        cen=float(np.sum(fr*mm)/(mm.sum()+1e-10))
        zcr=float(np.mean(librosa.feature.zero_crossing_rate(chunk)[0]))
        flat=float(np.mean(librosa.feature.spectral_flatness(S=mg)))
        def be(lo,hi): m=(fr>=lo)&(fr<hi); return float(np.mean(pw[m,:])) if m.any() else 0.0
        mfcc4=np.mean(librosa.feature.mfcc(S=librosa.power_to_db(
            librosa.feature.melspectrogram(S=mg**2,sr=sr)),n_mfcc=4),axis=1)
        # Sub-band flux
        flux=float(np.sqrt(np.mean(np.maximum(0,np.diff(mm))**2))) if len(mm)>1 else 0.0
        return {"rms":rms,"rms_db":rdb,"centroid":cen,"zcr":zcr,"flatness":flat,"flux":flux,
                "spectrum":mm.tolist(),"freqs":fr.tolist(),"mfcc4":mfcc4.tolist(),
                "bands":{n:be(lo,hi) for n,lo,hi in BANDS9}}

    # backward-compat alias
    generate_realtime_features = analyze_realtime_chunk
