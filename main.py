import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from matplotlib.widgets import Slider

# --- Initial Parameters ---
Am_init = 1        # Message amplitude
Ac_init = 2        # Carrier amplitude
fm_init = 100      # Message frequency (Hz)
fc_init = 1000     # Carrier frequency (Hz)
mu_init = 0.5      # Modulation index
Fs = 10000         # Sampling frequency
t = np.arange(0, 0.01, 1/Fs)  # 10 ms duration

# --- Function to generate signals ---


def generate_signals(Am, Ac, fm, fc, mu):
    m_t = Am * np.cos(2 * np.pi * fm * t)
    c_t = Ac * np.cos(2 * np.pi * fc * t)
    s_am = Ac * (1 + mu * m_t) * c_t
    s_am_rectified = np.abs(s_am)
    b, a = butter(5, fm * 1.2 / (Fs / 2), btype='low')
    demodulated_am = lfilter(b, a, s_am_rectified)
    return m_t, s_am, demodulated_am


# --- Initial computation ---
m_t, s_am, demodulated_am = generate_signals(
    Am_init, Ac_init, fm_init, fc_init, mu_init)

# --- Plot setup ---
fig, axs = plt.subplots(3, 1, figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Plot lines
line1, = axs[0].plot(t, m_t, label="Message Signal")
axs[0].set_title("Message Signal (m_t)")
axs[0].set_ylabel("Amplitude")

line2, = axs[1].plot(t, s_am, label="AM Signal", color='orange')
axs[1].set_title("AM Signal (s_am)")
axs[1].set_ylabel("Amplitude")

line3, = axs[2].plot(
    t, demodulated_am, label="Demodulated Signal", color='green')
axs[2].set_title("Demodulated Signal")
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("Amplitude")

for ax in axs:
    ax.grid(True)

# --- Slider axes ---
axcolor = 'lightgoldenrodyellow'
ax_Am = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_Ac = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_fm = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_fc = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_mu = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)

# --- Sliders ---
s_Am = Slider(ax_Am, 'Am', 0.1, 5.0, valinit=Am_init)
s_Ac = Slider(ax_Ac, 'Ac', 0.1, 5.0, valinit=Ac_init)
s_fm = Slider(ax_fm, 'fm (Hz)', 10, 1000, valinit=fm_init)
s_fc = Slider(ax_fc, 'fc (Hz)', 500, 5000, valinit=fc_init)
s_mu = Slider(ax_mu, 'μ', 0.0, 1.0, valinit=mu_init)

# --- Update function ---


def update(val):
    Am = s_Am.val
    Ac = s_Ac.val
    fm = s_fm.val
    fc = s_fc.val
    mu = s_mu.val

    m_t, s_am, demodulated_am = generate_signals(Am, Ac, fm, fc, mu)

    line1.set_ydata(m_t)
    line2.set_ydata(s_am)
    line3.set_ydata(demodulated_am)
    fig.canvas.draw_idle()


# Connect sliders to update function
s_Am.on_changed(update)
s_Ac.on_changed(update)
s_fm.on_changed(update)
s_fc.on_changed(update)
s_mu.on_changed(update)

plt.show()
