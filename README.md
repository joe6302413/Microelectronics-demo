# Microelectronics Demo — Transistors at DC Interactive Lab

An interactive educational tool for **GUPS2001 Electronics I** at National Taiwan University. Students explore how MOSFETs and BJTs behave at DC, and how the three basic amplifier connections are biased, by moving sliders and watching the operating point move.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joe6302413/Microelectronics-demo/blob/main/Transistor_DC_demo.ipynb)

## What it covers

| Tab | Content |
|:----|:--------|
| **1 · Device I–V characteristics** | Output characteristics (family of curves, edge of saturation), transfer characteristic, operating region, the equation in use with numbers substituted, g_m and r_o |
| **2 · Amplifier bias & transfer curve** | CS / CD / CG (MOSFET) and CE / CC / CB (BJT): live schematic with node voltages, voltage transfer curve coloured by region, load line, bias equations, 0.7 V course estimate vs exponential model, small-signal gain from the formula and from the slope of the transfer curve |
| **3 · n-type vs p-type** | NMOS/PMOS or npn/pnp drawn on signed axes (first vs third quadrant) with a sign-rule table |

All tabs respond to a global **Transistor** (MOSFET / BJT), **Type** (n / p) and device-parameter selector in the sidebar.

## Models

- MOSFET: square law with channel-length modulation, I_D = ½k(V_GS − V_t)²(1 + λV_DS); triode k(V_OV·V_DS − ½V_DS²). Body effect and subthreshold conduction are not included.
- BJT: Ebers–Moll transport model with the Early effect, so I_C = I_S e^{V_BE/V_T}(1 + V_CE/V_A) in the active region and saturation appears naturally (V_CE,sat ≈ 0.1–0.2 V). V_T = 25 mV.
- p-type devices use the same equations with all voltages and currents reversed in sign.

## How to run

**Option A — Google Colab (nothing to install).** Click the *Open in Colab* badge above, run the two cells, then click the `trycloudflare.com` link.

**Option B — your own computer.** The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/joe6302413/Microelectronics-demo.git
cd Microelectronics-demo
uv run streamlit run transistor.py
```

The app opens in your browser at `http://localhost:8501`.

## Dependencies

Managed automatically by `uv` via `pyproject.toml`: Streamlit, NumPy, Plotly, Matplotlib, schemdraw.

## Course

**GUPS2001 Electronics I**
Dr. Yi-Chun Chin — Global Undergraduate Program in Semiconductors, National Taiwan University

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).  
You are free to use, modify, and distribute this software under the same license.
