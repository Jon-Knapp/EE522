# Adaptive lower stopband edge and attenuation, chosen per video from its raw spectrum.
# Replaces the fixed F_STOP_LO / F_STOP_HI / A_STOP / A_DESIGN constants and the old design_bandpass.
# Uses np, plt, has_signal, needs_input, CH_NAMES, CH_COLORS, F_PASS_LO, F_PASS_HI from the notebook.
from scipy.signal import detrend
from scipy.signal.windows import dpss

F_STOP_MIN = 0.15        # Hz: lowest lower-stopband edge allowed (caps the filter length)
F_STOP_MAX = 0.55        # Hz: highest allowed, keeps a transition of at least 0.15 Hz below F_PASS_LO
RISE_DB = 10             # drift/respiration "ends" where the raw PSD falls within 10 dB of the in-band floor
RESID_DB = 20            # after filtering, low-frequency power must be 20 dB below the in-band power
A_MIN, A_MAX = 30, 60    # dB: limits on the stopband attenuation
NW = 2                   # multitaper time-bandwidth product (2*NW - 1 = 3 tapers)


def multitaper_psd(x, fs, nfft=8192):
    # One-sided multitaper PSD of each column of x (low variance, very low leakage).
    tapers = dpss(len(x), NW, 2 * NW - 1)                       # (K, L), unit energy
    X = np.fft.rfft(tapers[:, :, None] * x[None], n=nfft, axis=1)
    P = (np.abs(X) ** 2).mean(axis=0) / fs
    P[1:-1] *= 2
    return np.fft.rfftfreq(nfft, 1 / fs), P


@needs_input
def adaptive_specs(v, verbose=True):
    # Lower stopband edge and attenuation chosen from the raw spectrum of this video.
    # The passband (0.7 to 3.5 Hz) is fixed: only the parts outside it are measured.
    fs = v["fs"]
    x = detrend(v["x"], axis=0, type="linear")
    f, P = multitaper_psd(x, fs)
    band = (f >= F_PASS_LO) & (f <= F_PASS_HI)
    search = (f > 0) & (f <= F_STOP_MAX)
    rows = []
    for c in range(3):
        if not has_signal(x[:, c]):
            continue
        floor = np.median(P[band, c])                            # robust: harmonic peaks are a small part of the band
        hot = np.flatnonzero(search & (P[:, c] > floor * 10 ** (RISE_DB / 10)))
        f_edge = f[hot].max() if hot.size else F_STOP_MIN        # last frequency still well above the floor
        low = (f > 0) & (f <= max(f_edge, F_STOP_MIN))
        ratio_db = 10 * np.log10(P[low, c].sum() / P[band, c].sum())
        rows.append((CH_NAMES[c], f_edge, ratio_db, floor))
    f_stop_lo = float(np.clip(max(r[1] for r in rows), F_STOP_MIN, F_STOP_MAX))
    a_stop = float(np.clip(max(r[2] for r in rows) + RESID_DB, A_MIN, A_MAX))
    width = F_PASS_LO - f_stop_lo
    specs = dict(f_stop_lo=f_stop_lo, f_pass_lo=F_PASS_LO, f_pass_hi=F_PASS_HI,
                 f_stop_hi=F_PASS_HI + width, a_stop=a_stop, f=f, P=P, rows=rows)
    if verbose:
        for name, fe, rdb, _ in rows:
            print(f"  {name:5s}: low-frequency energy ends at {fe:.2f} Hz, "
                  f"low-frequency/in-band power = {rdb:+.1f} dB")
        print(f"  -> lower stopband edge {f_stop_lo:.2f} Hz, transition {width:.2f} Hz, "
              f"attenuation {a_stop:.1f} dB")
    return specs


@needs_input
def design_bandpass(v, specs, verbose=True):
    fs = v["fs"]
    s = specs
    width = s["f_pass_lo"] - s["f_stop_lo"]
    fc = ((s["f_stop_lo"] + s["f_pass_lo"]) / 2, (s["f_pass_hi"] + s["f_stop_hi"]) / 2)
    a_design = s["a_stop"]
    while True:   # kaiserord is slightly optimistic for a bandpass: raise the request until the spec is met
        N, beta = kaiserord(a_design, width / (fs / 2))
        N += (N % 2 == 0)
        h = firwin(N, fc, window=("kaiser", beta), pass_zero=False, fs=fs)
        f, H = freqz(h, worN=2**15, fs=fs)
        M = (N - 1) // 2
        A = np.real(H * np.exp(1j * 2 * np.pi * f / fs * M))
        mag = 20 * np.log10(np.maximum(np.abs(A), 1e-12))
        sb = (f <= s["f_stop_lo"]) | (f >= s["f_stop_hi"])
        if -mag[sb].max() >= s["a_stop"] - 0.05:
            break
        a_design += 0.5
    if verbose:
        print(f"  N = {N} taps ({N/fs:.2f} s), M = {M} ({M/fs:.2f} s), beta = {beta:.3f}, "
              f"cutoffs {fc[0]:.3f} / {fc[1]:.3f} Hz, stopband >= {-mag[sb].max():.1f} dB")
    return dict(h=h, N=N, M=M, beta=beta, fc=fc, fs=fs, f=f, H=H, A=A, **{k: s[k] for k in
                ("f_stop_lo", "f_pass_lo", "f_pass_hi", "f_stop_hi", "a_stop")})


@needs_input
def plot_specs(specs, label):
    # Raw multitaper PSD with each channel's floor + RISE_DB and the chosen edges.
    f, P = specs["f"], specs["P"]
    fig, ax = plt.subplots(figsize=(12, 4))
    for c, (name, f_edge, _, floor) in enumerate(specs["rows"]):
        c = CH_NAMES.index(name)
        ax.semilogy(f, P[:, c], color=CH_COLORS[c], lw=0.9, label=name)
        ax.hlines(floor * 10 ** (RISE_DB / 10), 0, F_STOP_MAX, color=CH_COLORS[c], ls=":", lw=1)
    ax.axvspan(specs["f_pass_lo"], specs["f_pass_hi"], color="C2", alpha=0.08, label="Passband (fixed)")
    ax.axvline(specs["f_stop_lo"], color="k", ls="--", lw=1,
               label=f"Lower stopband edge {specs['f_stop_lo']:.2f} Hz")
    ax.set(xlim=(0, 4), xlabel="Frequency (Hz)", ylabel="PSD (intensity$^2$/Hz)",
           title=f"{label}: raw multitaper PSD (dotted: in-band floor + {RISE_DB} dB)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()
