# 🎵 Beat Visualizer v3 — Enterprise Audio Analysis Platform

A full-stack, enterprise-grade audio analysis and visualization platform combining a Python scientific audio engine (v4.0) with advanced 3D WebGL rendering, real-time WebAudio processing, and 11 interactive visualization panels.

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

Production:
```bash
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app
# or: python -m gunicorn gunicorn_config:app
```

---

## Analysis Engine v4.0 — 29 Output Blocks

| Block | Description |
|-------|-------------|
| `beats` | BPM (DP + percussive + PLP), beat times, tatum grid, tempogram, Fourier tempogram, IOI histogram, PLP envelope, phase consistency |
| `onsets` | 6 methods: energy, mel-flux, complex-domain, HFC, phase-deviation, adaptive; consensus, sub-band OE |
| `rhythm` | Time signature (2/4, 3/4, 4/4, 6/8), swing factor, syncopation index, groove quantisation error, rhythmic entropy, IOI stats |
| `spectral` | Centroid/bandwidth/rolloff×3/flatness/flux/entropy/irregularity/kurtosis/skewness, tristimulus, 9-band + 24-Bark energy, sub-band flux |
| `stft` | Multi-resolution STFT (fine/mid/coarse: 1024/4096/8192-pt), power spectrum |
| `mel` | 128-band Mel spectrogram (fmin=20, fmax=sr/2) |
| `cqt` | Constant-Q (84 bins, 36bpo) + Variable-Q transform |
| `gammatone` | 64-channel Gammatone filterbank (ERB scale) |
| `nmf` | Non-negative Matrix Factorization (8 components) via sklearn |
| `pca` | PCA on spectral features (10 components) via sklearn |
| `mfcc` | 40-coeff MFCC + Δ + ΔΔ + CMVN; 13-coeff MFCC; BFCC (Bark); LPC order-16 |
| `chroma` | CQT / CENS / STFT / deep-chroma; HCDF; chord change times |
| `harmony` | KS (1990) + Temperley (2001) key estimation; relative key; Circle of Fifths position |
| `chords` | Template matching: 9 chord types × 12 roots = 108 templates; tonal tension curve |
| `tonnetz` | 6-dimensional tonal centroid (Harte et al. 2006) |
| `pitch` | pYIN probabilistic F0; Praat autocorrelation; melodic contour; vibrato detection; salience histogram |
| `voice` | Praat: jitter×5, shimmer×6, HNR/NHR, CPP, formants F1–F4, VAD ratio, LTAS |
| `dynamics` | RMS envelope, ITU-R BS.1770-4 LUFS (integrated/short-term/momentary), true-peak, LRA, crest factor, compression score |
| `timbre` | Brightness, warmth, noisiness, roughness (Sethares 1993), inharmonicity, odd/even ratio, A-weighted centroid, spectral kurtosis |
| `egemaps` | openSMILE eGeMAPS v02 (88 acoustic features) |
| `structure` | Recurrence SSM, Laplacian segmentation, checkerboard novelty, repetition curve, path SSM |
| `dtw` | Dynamic Time Warping vs KS major/minor templates |
| `music_info` | Danceability, energy, valence, acousticness, arousal, emotion quadrant (Russell's circumplex), genre fingerprint (7 genres), tempo category |
| `metadata` | Filename, duration, sr, bitrate, ID3/Vorbis tags |
| `waveform` | Downsampled sample array for waveform rendering |
| `energy` | RMS energy curve, dynamic range |

---

## Supported Formats (60+)

| Category | Formats |
|----------|---------|
| **Lossless** | WAV, FLAC, AIFF, AIF, AU, W64, CAF, RF64, BWF, APE, WV, TTA, DSF, DFF |
| **Lossy** | MP3, OGG, AAC, M4A, M4B, WMA, Opus, WEBM, AC3, DTS, MKA, AMR, RA, MP2 |
| **Video/Audio** | MP4, MOV, MKV, AVI, FLV, WMV, TS, M2TS, 3GP |
| **MIDI** | MID, MIDI, SMF, KAR, RMI |
| **Tracker** | MOD, XM, IT, S3M, STM |
| **Analysis** | JSON, BVP, CSV |

---

## Frontend — 11 Panels

| Panel | Key | Visualizations |
|-------|-----|---------------|
| Upload | `1` | Drag-drop, format chips, 18-stage progress |
| Waveform | `2` | 5 modes (standard/filled/bars/mirror/RMS), beat/onset/segment overlays |
| Spectrum | `3` | STFT/Mel/CQT/Gammatone/Power/Bands/Bark heatmaps + spectral features + contrast + tristimulus |
| 3D View | `4` | 5 Three.js modes: spectrogram3d, beatSphere, waveRing, particles, tonnetz3d |
| Beats | `5` | BPM ring + metronome, tempogram, PLP envelope, tempo candidates, rhythm metrics |
| Harmony | `6` | Key badge, KS profile, pitch wheel, chromagram (CQT/CENS/STFT), chord timeline, tonal tension, HCDF, Tonnetz |
| MFCC | `7` | MFCC-40/13/Δ/ΔΔ/BFCC/CMVN heatmaps, coefficient means/std, LPC |
| Pitch | `8` | pYIN + Praat F0 contour, voiced probability, voice quality table (jitter/shimmer/HNR/formants) |
| Dynamics | `9` | ITU-R BS.1770 meters, RMS envelope, short-term loudness, genre fingerprint, timbre radar |
| Structure | `0` | SSM heatmap, novelty function, segment timeline, energy + segment overlays |
| Realtime | `-` | Mic input: spectrum/waveform/bands/circular/waterfall; 8-band meters; live stats |

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload single file |
| POST | `/api/analyze/<id>` | Start analysis |
| GET | `/api/status/<id>` | Poll progress |
| GET | `/api/result/<id>` | Full result JSON |
| GET | `/api/result/<id>/block/<block>` | Single block fetch |
| GET | `/api/export/<id>/<fmt>` | Export: json, csv, midi, svg |
| POST | `/api/batch/upload` | Multi-file upload |
| POST | `/api/batch/analyze` | Batch analysis start |
| POST | `/api/compare` | Compare two jobs side-by-side |
| GET | `/api/jobs` | List all jobs |
| DELETE | `/api/jobs/<id>` | Delete job |
| GET | `/api/supported-formats` | List all 60+ formats |
| GET | `/api/health` | Capability check |

## WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `analysis_progress` | Server→Client | `{job_id, progress, message}` |
| `analysis_complete` | Server→Client | `{job_id, result}` |
| `analysis_error` | Server→Client | `{job_id, error}` |
| `realtime_chunk` | Client→Server | `{samples, sample_rate, chunk_id}` |
| `realtime_features` | Server→Client | `{rms_db, centroid, bands, ...}` |

## Keyboard Shortcuts

```
1-9  →  switch panels (1=upload … 9=dynamics, 0=structure, -=realtime)
Space → play/pause timeline
M     → toggle BPM metronome
Esc   → close export menu
```
