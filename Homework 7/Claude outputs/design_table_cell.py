import numpy as np
from scipy.signal import kaiserord, firwin, freqz
from IPython.display import Markdown, display


def design_kaiser(cut, fs):
    """Design the Kaiser-window FIR bandpass filter for one video from its cut (choose_cutoffs).

    kaiserord turns (attenuation, transition width) into a length N and a shape parameter beta.
    Its length formula is slightly optimistic for a bandpass, so the requested attenuation is raised
    in 0.5 dB steps until the measured stopband attenuation meets the requirement.
    """
    width = cut["f_pass_lo"] - cut["f_stop_lo"]            # same width on both sides (window method)
    fc = ((cut["f_stop_lo"] + cut["f_pass_lo"]) / 2,        # cutoffs in the middle of each transition,
          (cut["f_pass_hi"] + cut["f_stop_hi"]) / 2)        # where a windowed-sinc filter has gain 0.5 (-6 dB)
    a_design = cut["a_stop"]
    while True:
        N, beta = kaiserord(a_design, width / (fs / 2))
        N += (N % 2 == 0)                                   # odd length: a center tap, so it can be centered
        h = firwin(N, fc, window=("kaiser", beta), pass_zero=False, fs=fs)
        f, H = freqz(h, worN=2**15, fs=fs)
        mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        stop = (f <= cut["f_stop_lo"]) | (f >= cut["f_stop_hi"])
        pas = (f >= cut["f_pass_lo"]) & (f <= cut["f_pass_hi"])
        a_got = -mag[stop].max()
        if a_got >= cut["a_stop"] - 0.05:
            break
        a_design += 0.5
    return dict(h=h, N=N, M=(N - 1) // 2, beta=beta, fc=fc, fs=fs, f=f, H=H,
                a_got=a_got, ripple=(mag[pas].min(), mag[pas].max()), cut=cut)


def design_table(videos):
    """Show the Part 2 design decisions and each video's filter values as Markdown tables.

    videos: list of (name, cut, filt), with cut from choose_cutoffs and filt from design_kaiser.
    """
    def hz(x):
        return f"{x:.2f} Hz ({60 * x:.0f} bpm)"

    c0 = videos[0][1]                                                   # rule settings, same for all videos
    margin = c0["a_stop"] - max(c0["excesses"][i] for i in c0["decide_with"])

    common = """
#### Design decisions (the same for every video)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | IIR or FIR? | **FIR** | A FIR filter's impulse response has exactly $N$ taps, so the edge effects last exactly $M = (N-1)/2$ samples at each end (Part 4). Its impulse response is finite and symmetric, so it can be plotted exactly (Part 3). Its phase is linear, and zero when applied centered (Part 6). It is always stable. The recordings are processed offline, so the longer length of a FIR filter costs nothing that matters. An IIR Butterworth filter run forwards and backwards (`sosfiltfilt`) was considered: it is also zero-phase and slightly shorter, but its magnitude is squared and its edge effects only fade away gradually, so Parts 4 to 6 would rest on estimates instead of exact values. |
| 2 | FIR design method? | **Kaiser window** | The Kaiser window's shape parameter $\\beta$ sets the stopband attenuation, so each video gets exactly the attenuation it needs. `kaiserord` gives the length $N$ and $\\beta$ directly from two numbers we already have: the required attenuation and the transition width. On three test specifications, the fixed windows (Hann, Hamming, Blackman) needed about 15% to 70% more taps than Kaiser to meet the same requirements. Parks-McClellan (equiripple) needed 3% to 10% fewer taps, but it has no formula for the length (it needs a search), it failed to converge in two of three tests when the upper transition was widened, and it is harder to explain. |
| 3 | Filter type? | **Bandpass** | The PSD shows slow drift below about 0.3 Hz that is as strong as the pulse or stronger, so a lower stopband is needed (highpass side). Above the pulse harmonics the PSD is broadband noise, so an upper stopband removes the noise outside the band of interest (lowpass side). |
| 4 | How is it applied? | **Centered (non-causal), zero phase** | Each output sample $y[n]$ uses the inputs $x[n-M]$ to $x[n+M]$. The pulse is not delayed or distorted. This is possible because the whole recording is available before filtering. The causal version would delay everything by $M$ samples. |
| 5 | Length odd or even? | **Odd** | An odd length has a center tap, so the filter can be centered exactly on each sample. |
| 6 | Where are the cutoffs? | **Middle of each transition band** | A windowed-sinc filter's gain is about 0.5 (−6 dB) halfway through each transition band, so the −6 dB cutoff frequencies are the midpoints. |
"""
    head = "| Band or value | How it is chosen | " + " | ".join(n for n, _, _ in videos) + " |"
    sep = "|---|---|" + "---|" * len(videos)

    def row(label, how, fn):
        return f"| {label} | {how} | " + " | ".join(fn(c, d) for _, c, d in videos) + " |"

    rows = [
        row("Channels that set the edges", "One filter is used for all three channels, so the worst of these channels decides",
            lambda c, d: ", ".join(("Red", "Green", "Blue")[i] for i in c["decide_with"])),
        row("**Lower stopband**", f"From 0 Hz up to where the drift sinks below the floor + {c0['drift_db']:g} dB "
            f"(the floor is the median Welch PSD over the passband)",
            lambda c, d: f"0 to {hz(c['f_stop_lo'])}"),
        row("**Transition band 1**", "From the lower stopband edge to the lower passband edge",
            lambda c, d: f"{c['f_stop_lo']:.2f} to {c['f_pass_lo']:.2f} Hz (width {c['f_pass_lo'] - c['f_stop_lo']:.2f} Hz)"),
        row("Lower cutoff (−6 dB)", "Midpoint of transition band 1", lambda c, d: hz(d["fc"][0])),
        row("**Passband**", "Fixed: the range of heart rates the filter is meant to detect",
            lambda c, d: f"{hz(c['f_pass_lo'])} to {hz(c['f_pass_hi'])}"),
        row("**Transition band 2**", "Same width as transition band 1 (the Kaiser window method uses one transition width)",
            lambda c, d: f"{c['f_pass_hi']:.2f} to {c['f_stop_hi']:.2f} Hz (width {c['f_stop_hi'] - c['f_pass_hi']:.2f} Hz)"),
        row("Upper cutoff (−6 dB)", "Midpoint of transition band 2", lambda c, d: hz(d["fc"][1])),
        row("**Upper stopband**", "From the upper stopband edge to half the frame rate ($f_s/2$)",
            lambda c, d: f"{c['f_stop_hi']:.2f} to {d['fs'] / 2:.2f} Hz"),
        row("Required stopband attenuation", f"Strongest drift above the floor + {margin:g} dB margin",
            lambda c, d: f"{c['a_stop']:.1f} dB"),
        row("Achieved stopband attenuation", "Measured from the designed filter's frequency response",
            lambda c, d: f"{d['a_got']:.1f} dB"),
        row("Passband gain", "Measured: lowest to highest gain between the passband edges",
            lambda c, d: f"{d['ripple'][0]:+.3f} to {d['ripple'][1]:+.3f} dB"),
        row("Kaiser β", "From `kaiserord`: larger β gives more stopband attenuation (and needs more taps for the same transition width)",
            lambda c, d: f"{d['beta']:.3f}"),
        row("Length N", "From `kaiserord`, made odd: longer for more attenuation or a narrower transition",
            lambda c, d: f"{d['N']} taps ({d['N'] / d['fs']:.2f} s)"),
        row("Edge region at each end", "Half-length M = (N − 1)/2 samples (Part 4)",
            lambda c, d: f"{d['M']} samples ({d['M'] / d['fs']:.2f} s)"),
        row("Sampling rate $f_s$", "Frame rate of the video", lambda c, d: f"{d['fs']:.2f} Hz"),
    ]
    per_video = "#### Filter for each video\n\n" + "\n".join([head, sep] + rows)
    display(Markdown(common + "\n" + per_video))
