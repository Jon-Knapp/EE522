import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram

CH_NAMES, CH_COLORS = ("Red", "Green", "Blue"), ("red", "green", "blue")


def to_uniform(t, rgb, fs):
    """Resample the frame averages onto evenly spaced times 0, 1/fs, 2/fs, ...

    The frame timestamps are not perfectly even (31.67 to 35.00 ms apart), but the FFT
    behind every PSD estimate assumes evenly spaced samples. Linear interpolation fixes that.
    """
    t_u = np.arange(0, t[-1], 1 / fs)
    x_u = np.column_stack([np.interp(t_u, t, rgb[:, c]) for c in range(3)])
    return t_u, x_u


def rgb_psd(t, rgb, fs, seg_sec=20.0, overlap=0.5, nfft=8192, t_start=0.0):
    """PSD of each color channel, using only the samples from t_start seconds onward.

    Returns a dict:
      f, welch, per : frequencies (Hz), Welch PSD (len(f), 3), whole-record periodogram (len(f), 3)
      and the settings used (fs, N, nperseg, noverlap, n_seg, nfft, t_start), for labeling plots.
    """
    keep = t >= t_start                                   # optionally skip a start-up transient
    _, x = to_uniform(t[keep] - t[keep][0], rgb[keep], fs)
    x = x - x.mean(axis=0)                                # remove the mean (DC); keep the drift
    N = len(x)
    nper = min(int(round(seg_sec * fs)), N)               # samples per Welch segment
    nover = int(round(overlap * nper))                    # samples shared by neighboring segments
    f, P_welch = welch(x, fs, window="hann", nperseg=nper, noverlap=nover,
                       nfft=nfft, detrend=False, axis=0)
    _, P_per = periodogram(x, fs, window="hann", nfft=nfft, detrend=False, axis=0)
    n_seg = 1 + (N - nper) // (nper - nover)
    return dict(f=f, welch=P_welch, per=P_per, fs=fs, N=N, nperseg=nper, noverlap=nover,
                n_seg=n_seg, nfft=nfft, t_start=t_start)


def choose_cutoffs(psd, f_pass_lo=0.7, f_pass_hi=3.5, drift_db=10.0, margin_db=20.0,
                   f_stop_lo_range=(0.15, 0.55), decide_with=(0, 1, 2), verbose=True):
    """Pick the lower stopband edge and the stopband attenuation from a video's Welch PSD.

    The passband edges are fixed inputs: they are the range of heart rates we want to be able
    to detect, so they must not depend on where this video's pulse happens to be.

    Measured from the PSD, for each channel:
      floor      = median PSD (dB) over the passband: the typical level of the signal there
      edge       = highest frequency below f_stop_lo_range[1] where the PSD is still more than
                   drift_db above the floor, i.e. where the slow drift finally sinks into the floor
      excess     = how far the strongest drift sits above the floor (dB)
    One filter is used for all channels, so the worst of the channels listed in decide_with
    (0 = red, 1 = green, 2 = blue) sets the lower stopband edge, and
    a_stop = worst excess + margin_db leaves the drift margin_db below the floor.
    """
    f, db = psd["f"], 10 * np.log10(psd["welch"])
    band = (f >= f_pass_lo) & (f <= f_pass_hi)
    lo_min, lo_max = f_stop_lo_range
    low = (f > 0) & (f <= lo_max)
    floors, edges, excesses, found, peaks = [], [], [], [], []
    for c in range(3):
        floor = np.median(db[band, c])
        hot = np.flatnonzero(low & (db[:, c] > floor + drift_db))
        floors.append(floor)
        edges.append(f[hot].max() if hot.size else lo_min)   # nothing above threshold: use the minimum
        found.append(bool(hot.size))
        i_peak = np.flatnonzero(low)[np.argmax(db[low, c])]      # strongest drift: highest point below lo_max
        peaks.append((f[i_peak], db[i_peak, c]))
        excesses.append(db[i_peak, c] - floor)

    f_stop_lo = float(np.clip(max(edges[c] for c in decide_with), lo_min, lo_max))
    width = f_pass_lo - f_stop_lo                       # transition width; reused on the upper side
    cut = dict(f_stop_lo=f_stop_lo, f_pass_lo=f_pass_lo, f_pass_hi=f_pass_hi,
               f_stop_hi=f_pass_hi + width,
               a_stop=max(excesses[c] for c in decide_with) + margin_db,
               floors=floors, edges=edges, found=found, excesses=excesses, peaks=peaks,
               drift_db=drift_db, f_search_max=lo_max, decide_with=tuple(decide_with))
    if verbose:
        for c in range(3):
            used = "" if c in decide_with else "   (not used to decide)"
            where = (f"drift sinks into the floor at {edges[c]:.2f} Hz" if found[c]
                     else f"no drift above the threshold (minimum {lo_min:.2f} Hz used)")
            print(f"  {CH_NAMES[c]:5s}: {where}, "
                  f"strongest drift {excesses[c]:.1f} dB above the floor{used}")
        print(f"  stopband 0 to {cut['f_stop_lo']:.2f} Hz | passband {f_pass_lo:.2f} to {f_pass_hi:.2f} Hz | "
              f"stopband above {cut['f_stop_hi']:.2f} Hz | attenuation {cut['a_stop']:.1f} dB")
    return cut


def psd_description(psd):
    """One line describing how the PSD was computed (for the plot subtitle)."""
    fs, nper, nover = psd["fs"], psd["nperseg"], psd["noverlap"]
    return (f"Welch: Hann window, {nper} samples/segment ({nper / fs:.1f} s), "
            f"{nover} samples overlap ({100 * nover / nper:.0f}%), {psd['n_seg']} segments, "
            f"NFFT = {psd['nfft']}, resolution ≈ {1.44 * fs / nper:.3f} Hz (-3 dB)\n"
            f"Gray: one Hann periodogram of all {psd['N']} samples ({psd['N'] / fs:.1f} s, "
            f"starting at t = {psd['t_start']:g} s), fs = {fs:.2f} Hz, mean removed")


def plot_psd(psd, title, cut=None, fmax=5.0):
    """Plot the PSD of each channel in dB. If cut (from choose_cutoffs) is given, also show the
    stopbands, the passband, each channel's floor (solid black over the passband, where the median
    is taken), the strongest drift (black dot), the excess (double arrow from the floor up to the
    dot's level), the drift threshold (dotted), and where the drift ends (dashed)."""
    f = psd["f"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for c, ax in enumerate(axes):
        ax.plot(f, 10 * np.log10(psd["per"][:, c]), color="gray", lw=0.6, alpha=0.6,
                label="Periodogram (whole record)")
        ax.plot(f, 10 * np.log10(psd["welch"][:, c]), color=CH_COLORS[c], lw=1.6,
                label="Welch PSD")
        if cut is not None:
            ax.axvspan(0, cut["f_stop_lo"], color="red", alpha=0.08,
                       label=f"Stopbands (0 to {cut['f_stop_lo']:.2f} Hz, above {cut['f_stop_hi']:.2f} Hz)")
            ax.axvspan(cut["f_stop_hi"], max(fmax, cut["f_stop_hi"]), color="red", alpha=0.08)
            ax.axvspan(cut["f_pass_lo"], cut["f_pass_hi"], color="green", alpha=0.08,
                       label=f"Passband ({cut['f_pass_lo']:.2f} to {cut['f_pass_hi']:.2f} Hz)")
            floor = cut["floors"][c]
            ax.hlines(floor, cut["f_pass_lo"], cut["f_pass_hi"], color="k", lw=1.6,
                      label=f"Floor: median Welch PSD over {cut['f_pass_lo']:g} to {cut['f_pass_hi']:g} Hz "
                            f"({floor:.1f} dB)")
            ax.axhline(floor, color="k", lw=0.7, ls="--", alpha=0.5)   # same level, extended for reference
            f_pk, db_pk = cut["peaks"][c]
            x_arrow = (cut["f_stop_lo"] + cut["f_pass_lo"]) / 2          # in the transition gap
            ax.plot(f_pk, db_pk, "o", color="k", ms=5,
                    label=f"Strongest drift ({db_pk:.1f} dB at {f_pk:.3f} Hz)")
            ax.hlines(db_pk, f_pk, x_arrow, color="k", lw=0.6, alpha=0.6)
            ax.annotate("", xy=(x_arrow, db_pk), xytext=(x_arrow, floor),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1))
            ax.text(x_arrow + 0.03, (db_pk + floor) / 2, f"excess\n{cut['excesses'][c]:.1f} dB",
                    fontsize=8, va="center", bbox=dict(fc="white", ec="none", alpha=0.85))
            thr = floor + cut["drift_db"]
            ax.hlines(thr, 0, cut["f_search_max"], color="k", ls=":", lw=1,
                      label=f"Floor + {cut['drift_db']:g} dB (drift threshold)")
            ax.axvline(cut["edges"][c], color=CH_COLORS[c], ls="--", lw=1,
                       label=(f"This channel's drift ends: {cut['edges'][c]:.2f} Hz" if cut["found"][c]
                              else f"No drift above threshold ({cut['edges'][c]:.2f} Hz minimum)"))
        ax.set_ylabel(f"{CH_NAMES[c]}\nPSD (dB re 1 intensity²/Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)   # outside, right
    axes[-1].set_xlim(0, fmax)
    axes[-1].set_xlabel("Frequency (Hz)")
    top = axes[0].secondary_xaxis("top", functions=(lambda x: 60 * x, lambda b: b / 60))
    top.set_xlabel("Beats (or breaths) per minute")
    fig.suptitle(f"{title}\n{psd_description(psd)}", fontsize=10)
    plt.tight_layout()
    plt.show()


def psd_with_cutoffs(t, rgb, fs, title, seg_sec=20.0, overlap=0.5, t_start=0.0, fmax=5.0, **cutoff_options):
    """Compute the PSD, choose the cutoffs from it, and plot both. Returns (psd, cut)."""
    psd = rgb_psd(t, rgb, fs, seg_sec=seg_sec, overlap=overlap, t_start=t_start)
    cut = choose_cutoffs(psd, **cutoff_options)
    plot_psd(psd, title, cut=cut, fmax=fmax)
    return psd, cut
