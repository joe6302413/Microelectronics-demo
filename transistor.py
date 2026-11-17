
import math
import base64

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import schemdraw
import schemdraw.elements as elm

from schemdraw.elements import Element
from schemdraw.segments import Segment
from schemdraw.elements.transistors import fetw, fetl, fetgap, fetr

schemdraw.use("matplotlib")

# ─── Units used throughout: volts (V), milliamps (mA), kilo-ohms (kΩ) ─────────
VT = 0.025          # thermal voltage (V), Sedra & Smith convention
BETA_R = 0.1        # reverse beta of the BJT (only shapes the saturation region)

# Region → colour class.  "amp" = the region where the device amplifies.
CLASS_COLOR = {"amp": "#1f9d55", "ohmic": "#d97706", "off": "#8a949e"}


# ─── MOSFET: square-law model (written in n-type coordinates, all positive) ────
def mos_id(vgs, vds, P):
    """Drain current (mA). Triode carries the (1+λv_DS) factor too, so the
    curve is continuous at the edge of saturation."""
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    vov = vgs - P["Vt"]
    on = (vov > 0) & (vds > 0)
    tri = P["k"] * (vov * vds - 0.5 * vds ** 2) * (1 + P["lam"] * vds)
    sat = 0.5 * P["k"] * vov ** 2 * (1 + P["lam"] * vds)
    i = np.where(vds < vov, tri, sat)
    return np.where(on, i, 0.0)


def mos_region(vgs, vds, P):
    vov = vgs - P["Vt"]
    if vov <= 0:
        return "cutoff"
    return "triode" if vds < vov else "saturation"


def solve_mos_n(c, P):
    """n-type bias circuit: drain –R_D– V_DD, source –R_S– V_X, gate at V_G.
    No gate current, so one unknown (I_D) solved by bisection.  V_G and V_X
    may be arrays (a whole input sweep is then solved at once)."""
    VG, VX = np.broadcast_arrays(np.asarray(c["VG"], dtype=float), np.asarray(c["VX"], dtype=float))
    RD, RS, VDD = c["RD"], c["RS"], c["VDD"]

    def f(I):
        return mos_id(VG - VX - I * RS, VDD - I * RD - VX - I * RS, P)

    lo = np.zeros_like(VG)
    hi = np.maximum(f(lo), 0.0)          # I_D can never exceed its value at I = 0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        up = m - f(m) > 0
        hi = np.where(up, m, hi)
        lo = np.where(up, lo, m)
    I = 0.5 * (lo + hi)
    VD = VDD - I * RD
    VS = VX + I * RS
    VGS, VDS = VG - VS, VD - VS
    vov = VGS - P["Vt"]
    region = np.where(vov <= 0, "cutoff", np.where(VDS < vov, "triode", "saturation"))
    if VG.ndim == 0:
        return dict(I=float(I), VG=float(VG), VD=float(VD), VS=float(VS), VGS=float(VGS),
                    VDS=float(VDS), region=str(region))
    return dict(I=I, VG=VG, VD=VD, VS=VS, VGS=VGS, VDS=VDS, region=region)


# ─── BJT: Ebers–Moll transport model + Early effect (npn coordinates) ─────────
def _exp_lim(v):
    vm = 1.2
    return math.exp(vm / VT) * (1 + (v - vm) / VT) if v > vm else math.exp(v / VT)


def bjt_currents(vbe, vbc, P):
    eBE, eBC = _exp_lim(vbe), _exp_lim(vbc)
    vce = max(vbe - vbc, 0.0)
    ic = P["IS"] * (eBE - eBC) * (1 + vce / P["VA"]) - (P["IS"] / BETA_R) * (eBC - 1)
    ib = (P["IS"] / P["beta"]) * (eBE - 1) + (P["IS"] / BETA_R) * (eBC - 1)
    return ic, ib


def bjt_region(vbe, vbc):
    if vbc > 0.4:
        return "saturation"
    if vbe < 0.5:
        return "cutoff"
    return "active"


def solve_bjt_n(c, P, x0=None):
    """npn bias circuit: collector –R_C– V_CC, emitter –R_E– V_X, base –R_B– V_B.
    Damped Newton on the three node voltages (B, E, C)."""
    RB, RC, RE = max(c["RB"], 1e-3), max(c["RC"], 1e-3), max(c["RE"], 1e-3)

    def F(x):
        B, E, C = x
        ic, ib = bjt_currents(B - E, B - C, P)
        return np.array([(c["VB"] - B) / RB - ib,
                         (c["VCC"] - C) / RC - ic,
                         (E - c["VX"]) / RE - ib - ic])

    x = np.array(x0, dtype=float) if x0 is not None else \
        np.array([min(c["VB"], c["VX"] + 0.6), c["VX"], c["VCC"]])
    for _ in range(400):
        f = F(x)
        J = np.empty((3, 3))
        h = 1e-7
        for j in range(3):
            xp = x.copy()
            xp[j] += h
            J[:, j] = (F(xp) - f) / h
        try:
            d = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            break
        m = np.max(np.abs(d))
        s = 0.1 / m if m > 0.1 else 1.0
        x = x + s * d
        if m * s < 1e-10:
            break
    B, E, C = x
    ic, ib = bjt_currents(B - E, B - C, P)
    return dict(I=ic, IB=ib, IE=ic + ib, VG=B, VD=C, VS=E, VGS=B - E, VDS=C - E,
                VBC=B - C, region=bjt_region(B - E, B - C), x=x)


# ─── Amplifier circuits in real (signed) node voltages ────────────────────────
def solve_amp(fam, pol, cfg, s, P, warm=None):
    """s: VDD, Rd, Rs, RB, vI, VBIAS (Rd/Rs = drain/collector and source/emitter
    resistors).  A p-type circuit is solved as its n-type mirror image
    (every node voltage V ↦ V_DD − V)."""
    VDD = s["VDD"]
    n = pol == "n"
    ctl = s["VBIAS"] if cfg == "CG" else s["vI"]
    X = s["vI"] if cfg == "CG" else (0.0 if n else VDD)   # where R_S / R_E returns
    tr = (lambda v: v) if n else (lambda v: VDD - v)
    if fam == "MOS":
        r = solve_mos_n(dict(VDD=VDD, RD=s["Rd"], RS=s["Rs"], VG=tr(ctl), VX=tr(X)), P)
    else:
        r = solve_bjt_n(dict(VCC=VDD, RC=s["Rd"], RE=s["Rs"], RB=s["RB"], VB=tr(ctl), VX=tr(X)),
                        P, warm)
    out = dict(r)
    out["VG"], out["VD"], out["VS"] = tr(r["VG"]), tr(r["VD"]), tr(r["VS"])
    out["vO"] = out["VS"] if cfg == "CD" else out["VD"]
    out["Vint"] = VDD - tr(X)          # supply seen by the drain–source loop
    return out


def region_class(fam, region):
    if region == "cutoff":
        return "off"
    if fam == "MOS":
        return "amp" if region == "saturation" else "ohmic"
    return "amp" if region == "active" else "ohmic"


# ─── Formatting helpers ───────────────────────────────────────────────────────
def fmt_I(i_mA):
    a = abs(i_mA)
    if a >= 0.1:
        return f"{i_mA:.3f} mA"
    if a >= 1e-3:
        return f"{i_mA * 1e3:.2f} µA"
    if a >= 1e-6:
        return f"{i_mA * 1e6:.2f} nA"
    return "≈ 0"


def cur_label(sym, i_mA):
    """'$I_D$ = 0.250 mA', or '$I_D$ ≈ 0' when the current is negligible."""
    txt = fmt_I(i_mA)
    return f"{sym} {txt}" if txt.startswith("≈") else f"{sym} = {txt}"


def fmt_R(r_k):
    if not math.isfinite(r_k):
        return "∞"
    if r_k < 1:
        return f"{r_k * 1e3:.3g} Ω"
    return f"{r_k:.2f} kΩ" if r_k < 1000 else f"{r_k / 1000:.2f} MΩ"


def hsub(u):
    """'V_EC' -> 'V<sub>EC</sub>' for plot labels."""
    a, _, b = u.partition("_")
    return f"{a}<sub>{b}</sub>" if b else a


def region_box(fam, region, text):
    cls = region_class(fam, region)
    {"amp": st.success, "ohmic": st.warning, "off": st.info}[cls](text)


def svg_html(d, width=330):
    data = d.get_imagedata("svg")
    b64 = base64.b64encode(data).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="width:{width}px;max-width:100%">'


@st.cache_data(max_entries=500, show_spinner=False)
def device_svg(fam, pol, VG, VD, I):
    return svg_html(draw_device(fam, pol, VG, VD, I), 300)


@st.cache_data(max_entries=500, show_spinner=False)
def amp_svg(fam, pol, cfg, sk, rk):
    s = dict(zip(("VDD", "Rd", "Rs", "RB", "vI", "VBIAS"), sk))
    r = dict(zip(("VG", "VD", "VS", "vO", "I"), rk))
    return svg_html(draw_amp(fam, pol, cfg, s, r), 340)


# ─── Names that depend on device family and polarity ──────────────────────────
def names(fam, pol):
    p = pol == "p"
    if fam == "MOS":
        return dict(dev="PMOS" if p else "NMOS",
                    ctl="V_{SG}" if p else "V_{GS}", out="V_{SD}" if p else "V_{DS}",
                    ctl_u="V_SG" if p else "V_GS", out_u="V_SD" if p else "V_DS",
                    I="I_D", I_u="I_D", top="R_S" if p else "R_D", bot="R_D" if p else "R_S",
                    g="G", d="D", s="S", amp_region="saturation")
    return dict(dev="pnp" if p else "npn",
                ctl="V_{EB}" if p else "V_{BE}", out="V_{EC}" if p else "V_{CE}",
                ctl_u="V_EB" if p else "V_BE", out_u="V_EC" if p else "V_CE",
                I="I_C", I_u="I_C", top="R_E" if p else "R_C", bot="R_C" if p else "R_E",
                g="B", d="C", s="E", amp_region="active")


# ─── Schematics (schemdraw) ───────────────────────────────────────────────────
class SedraMos(Element):
    """Simplified MOSFET symbol as in Sedra & Smith: no body terminal, arrow on
    the source — pointing out of the device for NMOS, into it for PMOS."""
    _element_defaults = {"arrowlength": 0.25, "arrowwidth": 0.2}

    def __init__(self, *, ptype=False, **kwargs):
        super().__init__(**kwargs)
        aw, al = self.params["arrowwidth"], self.params["arrowlength"]
        bot = -2 * fetl - fetw
        self.segments.append(Segment([(0, 0), (0, -fetl), (fetw, -fetl),
                                      (fetw, -fetl - fetw), (0, -fetl - fetw), (0, bot)]))
        self.segments.append(Segment([(fetw + fetgap, -fetl), (fetw + fetgap, -fetl - fetw)]))
        self.segments.append(Segment([(fetw + fetgap, -fetl - fetw / 2),
                                      (fetw + fetgap + fetl + fetr, -fetl - fetw / 2)]))
        if ptype:   # source on top, arrow points into the channel
            self.segments.append(Segment([(0, -fetl), (0.8 * fetw, -fetl)], arrow="->",
                                         arrowwidth=aw, arrowlength=al))
            self.anchors["source"], self.anchors["drain"] = (0, 0), (0, bot)
        else:       # source at the bottom, arrow points out of the channel
            self.segments.append(Segment([(fetw, -fetl - fetw), (0.2 * fetw, -fetl - fetw)], arrow="->",
                                         arrowwidth=aw, arrowlength=al))
            self.anchors["drain"], self.anchors["source"] = (0, 0), (0, bot)
        self.anchors["gate"] = (fetw + fetgap + fetl + fetr, -fetl - fetw / 2)
        self.anchors["center"] = (0, -fetl - fetw / 2)
        self.elmparams["drop"] = (0, bot)
        self.elmparams["lblloc"] = "lft"


def _device(fam, pol, anchor_top=True):
    if fam == "MOS":
        e = SedraMos(ptype=(pol == "p")).reverse().right()
        top, bot, ctl = ("drain", "source", "gate") if pol == "n" else ("source", "drain", "gate")
    else:
        e = (elm.BjtNpn if pol == "n" else elm.BjtPnp)().right()
        top, bot, ctl = ("collector", "emitter", "base") if pol == "n" else ("emitter", "collector", "base")
    return e, top, bot, ctl


def draw_device(fam, pol, VG, VD, I):
    """Device on its own: source/emitter is the 0 V reference."""
    N = names(fam, pol)
    d = schemdraw.Drawing(show=False)
    d.config(unit=2.0, fontsize=13)
    e, top, bot, ctl = _device(fam, pol)
    e = e.anchor(top)
    d += e
    d += elm.Line().left().at(getattr(e, ctl)).length(1.0)
    d += elm.Dot(open=True).label(f"{N['g']}  {VG:+.2f} V", loc="left")
    I_txt = cur_label(f"$I_{N['I'][-1]}$", I)
    if pol == "n":
        ln = elm.Line().up().at(getattr(e, top)).length(1.6)
        d += ln
        d += elm.Dot(open=True).label(f"{N['d']}  {VD:+.2f} V", loc="right")
        # arrow beside the wire: current flows down, into the drain/collector
        d += elm.CurrentLabel(top=False, length=0.8, ofst=0.3, reverse=True).at(ln).label(I_txt, loc="bottom")
        d += elm.Line().down().at(getattr(e, bot)).length(0.4)
        d += elm.Ground().label(f"{N['s']} (0 V)", loc="right")
    else:
        d += elm.Line().up().at(getattr(e, top)).length(0.4)
        d += elm.Vdd().label(f"{N['s']} = 0 V (reference)")
        ln = elm.Line().down().at(getattr(e, bot)).length(1.6)
        d += ln
        d += elm.Dot(open=True).label(f"{N['d']}  {VD:+.2f} V", loc="right")
        # arrow beside the wire: current flows down, out of the drain/collector
        d += elm.CurrentLabel(top=True, length=0.8, ofst=0.3).at(ln).label(I_txt, loc="top")
    return d


def _res_or_wire(d, direction, value, name, at=None, label_loc="top"):
    if value <= 0:
        el = getattr(elm.Line(), direction)().length(1.5)
    else:
        el = getattr(elm.Resistor(), direction)().label(f"${name}$\n{value:g} kΩ", loc=label_loc)
    if at is not None:
        el = el.at(at)
    d += el
    return el


def draw_amp(fam, pol, cfg, s, r):
    N = names(fam, pol)
    d = schemdraw.Drawing(show=False)
    d.config(unit=2.0, fontsize=12)
    vdd_name = "V_{DD}" if fam == "MOS" else "V_{CC}"
    supply_lbl = f"${vdd_name}$ = {s['VDD']:.2f} V"
    vin_lbl = f"$v_I$ = {s['vI']:.2f} V"
    bias_lbl = f"$V_{{BIAS}}$ = {s['VBIAS']:.2f} V"
    n = pol == "n"
    Rtop, Rbot = (s["Rd"], s["Rs"]) if n else (s["Rs"], s["Rd"])

    # top rail: supply, or the input source for a p-type common-gate/base stage
    if n or cfg != "CG":
        d += elm.Vdd().label(supply_lbl)
    else:
        d += elm.Dot(open=True).label(vin_lbl, loc="top")
    rt = _res_or_wire(d, "down", Rtop, N["top"].replace("_", "_{") + "}", label_loc="bottom")
    d += elm.CurrentLabel(top=False, length=0.8, ofst=0.45).at(rt).label(
        cur_label(f"$I_{N['I'][-1]}$", abs(r["I"])), loc="top")
    top_node_V = r["VD"] if n else r["VS"]
    d += elm.Dot()
    out_top = (n and cfg != "CD") or ((not n) and cfg == "CD")
    if out_top:
        d.push()
        d += elm.Line().right().length(1.4)
        d += elm.Dot(open=True).label(f"$v_O$ = {r['vO']:.2f} V", loc="right")
        d.pop()
    else:
        d.push()
        d += elm.Line().right().length(0.01)
        d += elm.Label().label(f"{top_node_V:.2f} V", loc="right", ofst=(0.25, 0))
        d.pop()

    e, top, bot, ctl = _device(fam, pol)
    e = e.anchor(top)
    d += e

    # control terminal (gate/base)
    ctl_lbl = bias_lbl if cfg == "CG" else vin_lbl
    if fam == "MOS":
        d += elm.Line().left().at(getattr(e, ctl)).length(1.4)
        d += elm.Dot(open=True).label(ctl_lbl, loc="left")
    else:
        d.push()
        d += elm.Dot().at(getattr(e, ctl)).label(f"{r['VG']:.2f} V", loc="bottom", ofst=(-0.35, -0.1))
        d.pop()
        _res_or_wire(d, "left", s["RB"], "R_{B}", at=getattr(e, ctl))
        d += elm.Dot(open=True).label(ctl_lbl, loc="left")

    # bottom node
    d += elm.Line().down().at(getattr(e, bot)).length(0.3)
    d += elm.Dot()
    bot_node_V = r["VS"] if n else r["VD"]
    if not out_top:
        d.push()
        d += elm.Line().right().length(1.4)
        d += elm.Dot(open=True).label(f"$v_O$ = {r['vO']:.2f} V", loc="right")
        d.pop()
    else:
        d.push()
        d += elm.Line().right().length(0.01)
        d += elm.Label().label(f"{bot_node_V:.2f} V", loc="right", ofst=(0.25, 0))
        d.pop()
    _res_or_wire(d, "down", Rbot, N["bot"].replace("_", "_{") + "}", label_loc="bottom")
    if n and cfg == "CG":
        d += elm.Dot(open=True).label(vin_lbl, loc="bottom")
    else:
        d += elm.Ground()
    return d


def draw_small_signal(fam, cfg, model, v):
    """Small-signal equivalent circuit of the amplifier (DC supplies -> ac ground).
    model: 'pi' (hybrid-π) or 'T'.  v: dict of element values (kΩ, mA/V).
    The same picture holds for n-type and p-type devices."""
    mos = fam == "MOS"
    g, dn, sn = ("G", "D", "S") if mos else ("B", "C", "E")
    vc = "v_{gs}" if mos else "v_{\\pi}"
    d = schemdraw.Drawing(show=False)
    d.config(unit=2.0, fontsize=11)
    D, S = (4, 4), (4, 0)
    # ── transistor model between D, S and the control node G ──
    if model == "pi":
        G = (1.5, 4)
        d += elm.SourceControlledI().endpoints(D, S).label(f"$g_m {vc}$", loc="bottom", ofst=0.1)
        if mos:
            d += elm.Gap().endpoints(G, (1.5, 0)).label(["+", f"${vc}$", "−"], loc="bottom")
        else:
            d += (elm.Resistor().endpoints(G, (1.5, 0))
                  .label(f"$r_\\pi$\n{fmt_R(v['rpi'])}", loc="top")
                  .label(["+", f"${vc}$", "−"], loc="bottom"))
        d += elm.Line().endpoints((1.5, 0), S)
    else:
        X = (4, 2)
        G = (1.5, 2)
        d += elm.SourceControlledI().endpoints(D, X).label(
            "$g_m v_{gs}$" if mos else "$\\alpha i_e$", loc="bottom", ofst=0.1)
        d += elm.Resistor().endpoints(X, S).label(
            f"$1/g_m$\n{fmt_R(1 / v['gm'])}" if mos else f"$r_e$\n{fmt_R(v['re'])}", loc="bottom")
        d += elm.Line().endpoints(X, G)
        d += elm.Dot().at(X)
    if math.isfinite(v["ro"]):
        d += elm.Line().endpoints(D, (6, 4))
        d += elm.Line().endpoints(S, (6, 0))
        d += elm.Resistor().endpoints((6, 4), (6, 0)).label(f"$r_o$\n{fmt_R(v['ro'])}", loc="bottom")
    d += elm.Dot().at(D).label(dn, loc="top")
    d += elm.Dot().at(S).label(sn, loc="bottom", ofst=(-0.35, -0.05))
    d += elm.Dot().at(G).label(g, loc="top")

    # ── drain/collector side: R_D to ac ground, output ──
    d += elm.Line().endpoints(D, (8, 4))
    if v["Rd"] > 0:
        d += elm.Resistor().endpoints((8, 4), (8, 1.9)).label(
            f"$R_{'D' if mos else 'C'}$\n{v['Rd']:g} kΩ", loc="bottom")
        d += elm.Ground().at((8, 1.9))
    else:
        d += elm.Line().endpoints((8, 4), (8, 3.2))
        d += elm.Ground().at((8, 3.2))
    if cfg != "CD":
        d += elm.Dot().at((8, 4))
        d += elm.Line().endpoints((8, 4), (9.6, 4))
        d += elm.Dot(open=True).at((9.6, 4)).label("$v_o$", loc="right")
    else:
        d += elm.Line().endpoints(S, (9.6, 0))
        d += elm.Dot(open=True).at((9.6, 0)).label("$v_o$", loc="right")

    # ── source/emitter side: R_S to ac ground, or to the input for CG/CB ──
    bot = (4, -2.0)
    if v["Rs"] > 0:
        d += elm.Resistor().endpoints(S, bot).label(f"$R_{'S' if mos else 'E'}$\n{v['Rs']:g} kΩ", loc="bottom")
    else:
        d += elm.Line().endpoints(S, bot)
    if cfg == "CG":
        d += elm.SourceSin().endpoints((4, -4.0), bot).label("$v_i$", loc="bottom")
        d += elm.Ground().at((4, -4.0))
    else:
        d += elm.Ground().at(bot)

    # ── control side: input source (CS/CD) or ac ground (CG) ──
    xin = (-1.6, G[1])
    if cfg == "CG":
        if not mos and v["RB"] > 0:
            d += elm.Resistor().endpoints(G, xin).label(f"$R_B$\n{v['RB']:g} kΩ", loc="top")
        else:
            d += elm.Line().endpoints(G, xin)
        d += elm.Line().endpoints(xin, (-1.6, G[1] - 0.8))
        d += elm.Ground().at((-1.6, G[1] - 0.8))
    else:
        if not mos and v["RB"] > 0:
            d += elm.Resistor().endpoints(xin, G).label(f"$R_B$\n{v['RB']:g} kΩ", loc="top")
        else:
            d += elm.Line().endpoints(xin, G)
        d += elm.SourceSin().endpoints((-1.6, G[1] - 2.0), xin).label("$v_i$", loc="bottom")
        d += elm.Ground().at((-1.6, G[1] - 2.0))
    return d


@st.cache_data(max_entries=300, show_spinner=False)
def small_signal_svg(fam, cfg, model, vk):
    v = dict(zip(("gm", "rpi", "re", "ro", "Rd", "Rs", "RB"), vk))
    return svg_html(draw_small_signal(fam, cfg, model, v), 460)


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def base_layout(fig, title, xt, yt, height=420, xr=None, yr=None):
    fig.update_layout(template="plotly_white", title=title, height=height,
                      margin=dict(l=60, r=20, t=50, b=60),
                      xaxis=dict(title=xt, range=xr, zeroline=True),
                      yaxis=dict(title=yt, range=yr, zeroline=True),
                      legend=dict(orientation="h", y=-0.2, x=0))
    return fig


def add_region_colored(fig, x, y, classes, labels, width=3):
    """Draw a curve split into segments coloured by operating region."""
    shown = set()
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or classes[i] != classes[start]:
            seg = slice(start, min(i + 1, len(x)))
            c = classes[start]
            fig.add_trace(go.Scatter(x=x[seg], y=y[seg], mode="lines",
                                     line=dict(color=CLASS_COLOR[c], width=width),
                                     name=labels[c], legendgroup=c, showlegend=c not in shown,
                                     hoverinfo="skip"))
            shown.add(c)
            start = i


# ═══════════════════════════════════════════════════════════════════════════════
# Page
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Transistor DC Lab", layout="wide")
st.title("Transistors at DC — Interactive Lab")
st.caption("GUPS2001 Electronics I | Dr. Yi-Chun Chin")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Settings")
    fam = st.radio("Transistor", ["MOS", "BJT"], horizontal=True,
                   format_func=lambda f: "MOSFET" if f == "MOS" else "BJT")
    pol = st.radio("Type", ["n", "p"], horizontal=True,
                   format_func=lambda p: {("MOS", "n"): "NMOS", ("MOS", "p"): "PMOS",
                                          ("BJT", "n"): "npn", ("BJT", "p"): "pnp"}[(fam, p)])
    N = names(fam, pol)
    st.divider()
    st.markdown("**Device parameters**")
    if fam == "MOS":
        Vt = st.slider(("Threshold |V_tp| (V)" if pol == "p" else "Threshold V_tn (V)"),
                       0.2, 1.5, 0.5, step=0.05)
        k = st.slider(("k_p = k′_p(W/L)  (mA/V²)" if pol == "p" else "k_n = k′_n(W/L)  (mA/V²)"),
                      0.2, 10.0, 2.0, step=0.1)
        lam = st.slider("Channel-length modulation λ (V⁻¹)", 0.0, 0.2, 0.0, step=0.005)
        P = dict(Vt=Vt, k=k, lam=lam)
        VA_txt = "∞" if lam == 0 else f"{1 / lam:.0f} V"
        st.markdown(f"""
| Quantity | Value |
|:--|--:|
| V<sub>t</sub> | {"−" if pol == "p" else "+"}{Vt:.2f} V |
| V<sub>A</sub> = 1/λ | {VA_txt} |
| I<sub>D</sub> at V<sub>OV</sub> = 0.2 V | {fmt_I(0.5 * k * 0.04)} |
""", unsafe_allow_html=True)
    else:
        beta = st.slider("Current gain β", 20, 400, 100, step=5)
        early = st.checkbox("Include the Early effect", value=False)
        VA = float(st.slider("Early voltage V_A (V)", 10, 300, 100, step=5)) if early else float("inf")
        IS_A = float(st.select_slider("Saturation current I_S (A)",
                                      options=["1e-16", "1e-15", "1e-14", "1e-13"], value="1e-15"))
        P = dict(IS=IS_A * 1e3, beta=beta, VA=VA)
        st.markdown(f"""
| Quantity | Value |
|:--|--:|
| V<sub>T</sub> | {VT * 1e3:.0f} mV |
| V<sub>A</sub> | {"∞" if math.isinf(VA) else f"{VA:.0f} V"} |
| α = β/(β+1) | {beta / (beta + 1):.4f} |
| {"V<sub>EB</sub>" if pol == "p" else "V<sub>BE</sub>"} at I<sub>C</sub> = 1 mA | {VT * math.log(1 / P["IS"]):.3f} V |
""", unsafe_allow_html=True)
    st.divider()
    st.caption("Units: V, mA, kΩ.  p-type devices use the same equations with "
               "the voltages and the current reversed in sign.")

REGION_LABELS = ({"amp": "saturation (amplifier region)", "ohmic": "triode", "off": "cutoff"}
                 if fam == "MOS" else
                 {"amp": "active (amplifier region)", "ohmic": "saturation", "off": "cutoff"})

tab1, tab2, tab3 = st.tabs([
    "1 · Device I–V characteristics",
    "2 · Amplifier bias & transfer curve",
    "3 · n-type vs p-type",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Device I–V
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment
def render_tab1():
    st.subheader(f"{N['dev']} I–V characteristics")
    if fam == "MOS":
        st.latex(r"""
        I_D=\begin{cases}0 & V_{OV}\le 0\ \text{(cutoff)}\\[2pt]
        k\left(V_{OV}V_{DS}-\tfrac12V_{DS}^2\right) & V_{DS}<V_{OV}\ \text{(triode)}\\[2pt]
        \tfrac12k\,V_{OV}^2\,(1+\lambda V_{DS}) & V_{DS}\ge V_{OV}\ \text{(saturation)}\end{cases}
        \qquad V_{OV}=V_{GS}-V_t""" if pol == "n" else r"""
        I_D=\begin{cases}0 & V_{OV}\le 0\ \text{(cutoff)}\\[2pt]
        k\left(V_{OV}V_{SD}-\tfrac12V_{SD}^2\right) & V_{SD}<V_{OV}\ \text{(triode)}\\[2pt]
        \tfrac12k\,V_{OV}^2\,(1+\lambda V_{SD}) & V_{SD}\ge V_{OV}\ \text{(saturation)}\end{cases}
        \qquad V_{OV}=V_{SG}-|V_{tp}|""")
        st.caption("Note: the app also multiplies the triode current by (1+λV_DS) so that the curves join "
                   "smoothly at the edge of saturation. With λ = 0 it is exactly the textbook model.")
    else:
        st.latex(r"""I_C = I_S\,e^{V_{BE}/V_T}\left(1+\frac{V_{CE}}{V_A}\right),\qquad
        I_B=\frac{I_C}{\beta}\qquad\text{(active: EBJ forward, CBJ reverse)}""" if pol == "n" else
                 r"""I_C = I_S\,e^{V_{EB}/V_T}\left(1+\frac{V_{EC}}{V_A}\right),\qquad
        I_B=\frac{I_C}{\beta}\qquad\text{(active: EBJ forward, CBJ reverse)}""")

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        if fam == "MOS":
            vctl = st.slider(f"{N['ctl_u']} (V)", 0.0, 3.0, 1.5, step=0.01, key=f"t1_ctl_{fam}")
            vout = st.slider(f"{N['out_u']} (V)", 0.0, 5.0, 2.5, step=0.01, key=f"t1_out_{fam}")
            I = float(mos_id(vctl, vout, P))
            region = mos_region(vctl, vout, P)
        else:
            vctl = st.slider(f"{N['ctl_u']} (V)", 0.50, 0.80, 0.68, step=0.002, key=f"t1_ctl_{fam}",
                             format="%.3f")
            vout = st.slider(f"{N['out_u']} (V)", 0.0, 5.0, 3.0, step=0.01, key=f"t1_out_{fam}")
            I, IB = bjt_currents(vctl, vctl - vout, P)
            region = bjt_region(vctl, vctl - vout)
        sgn = 1 if pol == "n" else -1
        rname = REGION_LABELS[region_class(fam, region)]
        region_box(fam, region, f"**{rname.capitalize()}**  ·  {N['I']} = {fmt_I(I)}")
        st.markdown(device_svg(fam, pol, round(sgn * vctl, 3), round(sgn * vout, 2), float(f"{I:.4g}")),
                    unsafe_allow_html=True)

        # numbers substituted into the equation that applies
        if fam == "MOS":
            vov = vctl - P["Vt"]
            o, c = N["out"], N["ctl"]
            if region == "cutoff":
                st.latex(rf"V_{{OV}} = {vctl:.2f}-{P['Vt']:.2f} = {vov:.2f}\ \text{{V}}\le 0"
                         r"\ \Rightarrow\ I_D = 0")
            elif region == "triode":
                st.latex(rf"{o} = {vout:.2f} < V_{{OV}} = {vov:.2f}\ \Rightarrow\ \text{{triode}}")
                st.latex(rf"\begin{{aligned}}I_D &= {P['k']:.1f}\left({vov:.2f}\cdot{vout:.2f}-\tfrac12\,{vout:.2f}^2\right)\\"
                         rf"&\quad\times(1+{P['lam']:.3f}\cdot{vout:.2f})\\ &= {I:.3f}\ \text{{mA}}\end{{aligned}}")
            else:
                st.latex(rf"{o} = {vout:.2f}\ \ge V_{{OV}} = {vov:.2f}\ \Rightarrow\ \text{{sat.}}")
                st.latex(rf"\begin{{aligned}}I_D&=\tfrac12({P['k']:.1f})({vov:.2f})^2(1+{P['lam']:.3f}\cdot{vout:.2f})\\"
                         rf"&= {I:.3f}\ \text{{mA}}\end{{aligned}}")
            gm = P["k"] * vov * (1 + P["lam"] * vout) if region == "saturation" else float("nan")
            ro = 1 / (P["lam"] * I) if (P["lam"] > 0 and I > 0) else float("inf")
            st.markdown(f"""
| V<sub>OV</sub> | g<sub>m</sub> (sat.) | r<sub>o</sub> |
|:-:|:-:|:-:|
| {vov:.2f} V | {"—" if math.isnan(gm) else f"{gm:.3f} mA/V"} | {fmt_R(ro) if region == "saturation" else "—"} |
""", unsafe_allow_html=True)
        else:
            vbc = vctl - vout
            jc = "V_{CB}" if pol == "n" else "V_{BC}"
            st.latex(rf"\text{{CBJ: }}{jc} = {-vbc:+.2f}\ \text{{V}}\ "
                     + (r"\Rightarrow\ \text{reverse biased}" if vbc <= 0.4 else
                        r"\Rightarrow\ \text{forward biased (saturation)}"))
            beff = I / IB if IB > 0 else float("nan")
            st.markdown(f"""
| I<sub>B</sub> | I<sub>C</sub> / I<sub>B</sub> | g<sub>m</sub> = I<sub>C</sub>/V<sub>T</sub> |
|:-:|:-:|:-:|
| {fmt_I(IB)} | {f"{beff:.1f}" if math.isfinite(beff) and I > 1e-9 else "—"} | {f"{I / VT:.2f} mA/V" if region == "active" else "—"} |
""", unsafe_allow_html=True)
            if region == "saturation":
                st.caption(f"In saturation I_C/I_B ({beff:.1f}) falls below β = {P['beta']}: "
                           "the collector can no longer take all the current the base could supply.")

    with col_plot:
        signed = st.checkbox("Signed axes for p-type (show the third-quadrant picture)",
                             value=False, disabled=(pol == "n"), key="t1_signed")
        s = -1 if (pol == "p" and signed) else 1
        if fam == "MOS":
            vx = np.linspace(0, 5, 201)
            fam_vals = [v for v in [1.0, 1.5, 2.0, 2.5, 3.0] if v > P["Vt"]]
            fig = go.Figure()
            for v in fam_vals:
                fig.add_trace(go.Scatter(x=s * vx, y=s * mos_id(v, vx, P), mode="lines",
                                         line=dict(color="lightgray", width=1.5),
                                         name=f"{N['ctl_u']} = {v:.1f} V", showlegend=False,
                                         hovertemplate=f"{N['ctl_u']} = {v:.1f} V<extra></extra>"))
                fig.add_annotation(x=s * 5, y=s * float(mos_id(v, 5, P)), text=f"{v:.1f} V",
                                   showarrow=False, xanchor="left" if s > 0 else "right",
                                   font=dict(color="gray", size=10))
            # edge of saturation locus  V_DS = V_OV
            vov_l = np.linspace(0, 3.0 - P["Vt"], 200)
            fig.add_trace(go.Scatter(x=s * vov_l, y=s * 0.5 * P["k"] * vov_l ** 2 * (1 + P["lam"] * vov_l),
                                     mode="lines", line=dict(color="black", dash="dash", width=1),
                                     name=f"edge of saturation ({N['out_u']} = V_OV)"))
            ycur = mos_id(vctl, vx, P)
            fig.add_trace(go.Scatter(x=s * vx, y=s * ycur, mode="lines",
                                     line=dict(color="royalblue", width=3),
                                     name=f"{N['ctl_u']} = {vctl:.2f} V"))
            ymax = max(float(mos_id(3.0, 5, P)), float(ycur.max())) * 1.1 + 1e-6
        else:
            vx = np.linspace(0, 5, 201)
            fam_vals = [0.64, 0.66, 0.68, 0.70, 0.72]
            fig = go.Figure()

            def ic_curve(vbe):
                # clip the small reverse current near V_CE = 0 so the plot stays readable
                return np.maximum([bjt_currents(vbe, vbe - v, P)[0] for v in vx], 0.0)

            for v in fam_vals:
                yy = ic_curve(v)
                fig.add_trace(go.Scatter(x=s * vx, y=s * yy, mode="lines",
                                         line=dict(color="lightgray", width=1.5), showlegend=False,
                                         hovertemplate=f"{N['ctl_u']} = {v:.2f} V<extra></extra>"))
                fig.add_annotation(x=s * 5, y=s * yy[-1], text=f"{v:.2f} V", showarrow=False,
                                   xanchor="left" if s > 0 else "right", font=dict(color="gray", size=10))
            vb_l = np.linspace(0.6, 0.8, 100)
            edge = np.array([bjt_currents(v, 0.4, P)[0] for v in vb_l])
            fig.add_trace(go.Scatter(x=s * (vb_l - 0.4), y=s * edge, mode="lines",
                                     line=dict(color="black", dash="dash", width=1),
                                     name="edge of saturation (CBJ forward by 0.4 V)"))
            ycur = ic_curve(vctl)
            fig.add_trace(go.Scatter(x=s * vx, y=s * ycur, mode="lines",
                                     line=dict(color="royalblue", width=3),
                                     name=f"{N['ctl_u']} = {vctl:.3f} V"))
            ymax = max(float(ic_curve(0.72)[-1]), float(ycur.max())) * 1.1 + 1e-6
        cls = region_class(fam, region)
        fig.add_trace(go.Scatter(x=[s * vout], y=[s * I], mode="markers",
                                 marker=dict(size=13, color=CLASS_COLOR[cls], line=dict(color="black", width=1)),
                                 name=f"operating point ({rname})"))
        xr = [0, 5.4] if s > 0 else [-5.4, 0]
        yr = [0, ymax] if s > 0 else [-ymax, 0]
        xt = f"{hsub(N['out_u'])} (V)" if s > 0 else ("V<sub>DS</sub> (V)" if fam == "MOS" else "V<sub>CE</sub> (V)")
        yt = f"{hsub(N['I_u'])} (mA)" if s > 0 else (
            "I<sub>D</sub>, into drain (mA)" if fam == "MOS" else "I<sub>C</sub>, into collector (mA)")
        st.plotly_chart(base_layout(fig, "Output characteristics", xt, yt, 430, xr, yr),
                        width="stretch")

        # transfer characteristic
        fig2 = go.Figure()
        if fam == "MOS":
            vg = np.linspace(0, 3, 301)
            yy = mos_id(vg, vout, P)
            fig2.add_trace(go.Scatter(x=s * vg, y=s * yy, mode="lines", line=dict(color="royalblue", width=3),
                                      name=f"{N['out_u']} = {vout:.2f} V"))
            fig2.add_vline(x=s * P["Vt"], line_dash="dot", line_color="gray",
                           annotation_text="|V_t|" if pol == "p" else "V_t")
            ylog = False
            ttl = "Transfer characteristic"
        else:
            ylog = st.checkbox("Log current axis (60 mV per decade)", value=False, key="t1_log")
            vg = np.linspace(0.40, 0.80, 201)
            yy = np.array([bjt_currents(v, v - vout, P)[0] for v in vg])
            yb = np.array([bjt_currents(v, v - vout, P)[1] for v in vg])
            fig2.add_trace(go.Scatter(x=s * vg, y=s * yy if not ylog else yy, mode="lines",
                                      line=dict(color="royalblue", width=3), name=f"{N['I_u']}"))
            fig2.add_trace(go.Scatter(x=s * vg, y=s * yb if not ylog else yb, mode="lines",
                                      line=dict(color="mediumpurple", width=2, dash="dash"), name="I_B"))
            ttl = f"Transfer characteristic at {N['out_u']} = {vout:.2f} V"
        fig2.add_trace(go.Scatter(x=[s * vctl], y=[(s * I) if not ylog else max(I, 1e-12)], mode="markers",
                                  marker=dict(size=13, color=CLASS_COLOR[cls], line=dict(color="black", width=1)),
                                  name="operating point"))
        xt2 = f"{hsub(N['ctl_u'])} (V)" if s > 0 else ("V<sub>GS</sub> (V)" if fam == "MOS" else "V<sub>BE</sub> (V)")
        base_layout(fig2, ttl if fam == "BJT" else f"{ttl} at {N['out_u']} = {vout:.2f} V", xt2, yt, 380)
        if ylog:
            fig2.update_yaxes(type="log", range=[-6, 2], title=f"{hsub(N['I_u'])} (mA, log)")
        elif fam == "BJT":
            fig2.update_yaxes(range=[min(0, s * ymax), max(0, s * ymax)])
        st.plotly_chart(fig2, width="stretch")

    with st.expander("What to try", expanded=False):
        if fam == "MOS":
            st.markdown(f"""
- Hold {N['ctl_u']} fixed and raise {N['out_u']} from 0: the point leaves the triode region exactly where it crosses the dashed curve, {N['out_u']} = V_OV.
- In saturation, I_D does not depend on {N['out_u']} at all with λ = 0 (the default). That flatness is what makes the MOSFET a good current source. Raise λ in the sidebar to add channel-length modulation: the curves tilt and r_o becomes finite.
- Below |V_t| nothing flows, whatever {N['out_u']} is.
- Switch between NMOS and PMOS: the curves are identical in magnitude. Tick *signed axes* to see that every voltage and the current change sign.""")
        else:
            st.markdown(f"""
- Raise {N['ctl_u']} by 18 mV: I_C doubles (V_T ln 2 = 17.3 mV). Turn on the log axis to see a straight line of about 60 mV per decade.
- Lower {N['out_u']} below about 0.3 V: the collector–base junction becomes forward biased, the BJT saturates and I_C/I_B drops below β.
- Careful with names: **BJT saturation is not MOSFET saturation**. For a BJT the amplifier region is *active*.
- The curves are flat in the active region because the Early effect is off by default. Tick *Include the Early effect* in the sidebar to make them tilt (finite r_o).
- Switch between npn and pnp: same magnitudes, opposite signs.""")

with tab1:
    render_tab1()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Amplifier bias and transfer curve
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    ("MOS", "CS"): dict(VDD=3.0, Rd=5.0, Rs=0.0, RB=0.0, vI=1.0, VBIAS=1.5),
    ("MOS", "CD"): dict(VDD=3.0, Rd=0.0, Rs=2.0, RB=0.0, vI=2.0, VBIAS=1.5),
    ("MOS", "CG"): dict(VDD=3.0, Rd=5.0, Rs=1.0, RB=0.0, vI=0.3, VBIAS=1.5),
    ("BJT", "CS"): dict(VDD=5.0, Rd=2.0, Rs=0.0, RB=100.0, vI=1.5, VBIAS=1.5),
    ("BJT", "CD"): dict(VDD=5.0, Rd=0.0, Rs=2.0, RB=10.0, vI=3.0, VBIAS=1.5),
    ("BJT", "CG"): dict(VDD=5.0, Rd=2.0, Rs=1.0, RB=10.0, vI=0.3, VBIAS=1.5),
}
CFG_NAMES = {"MOS": {"CS": "Common source (CS)", "CD": "Common drain / source follower (CD)",
                     "CG": "Common gate (CG)"},
             "BJT": {"CS": "Common emitter (CE)", "CD": "Common collector / emitter follower (CC)",
                     "CG": "Common base (CB)"}}

@st.fragment
def render_tab2():
    st.subheader("Amplifier configurations — DC bias and voltage transfer curve")
    cfg = st.radio("Configuration", ["CS", "CD", "CG"], horizontal=True,
                   format_func=lambda c: CFG_NAMES[fam][c], key=f"cfg_{fam}")
    key = f"{fam}_{pol}_{cfg}"
    dflt = dict(DEFAULTS[(fam, cfg)])
    if pol == "p":   # mirror image of the n-type defaults
        dflt["vI"] = dflt["VDD"] - dflt["vI"]
        dflt["VBIAS"] = dflt["VDD"] - dflt["VBIAS"]
    for nm, v in dflt.items():
        st.session_state.setdefault(f"{key}_{nm}", v)

    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        vdd_u = "V_DD" if fam == "MOS" else "V_CC"
        VDD = st.slider(f"Supply {vdd_u} (V)", 1.0, 12.0, step=0.1, key=f"{key}_VDD")
        for nm in ("vI", "VBIAS"):
            if st.session_state[f"{key}_{nm}"] > VDD:
                st.session_state[f"{key}_{nm}"] = VDD
        in_where = {"CS": N["g"], "CD": N["g"], "CG": N["s"]}[cfg]
        vI = st.slider(f"Input v_I (V), applied at the {'gate' if in_where == 'G' else 'base' if in_where == 'B' else 'source' if in_where == 'S' else 'emitter'}",
                       0.0, VDD, step=0.01, key=f"{key}_vI")
        VBIAS = (st.slider(f"Bias V_BIAS on the {'gate' if fam == 'MOS' else 'base'} (V)", 0.0, VDD,
                           step=0.05, key=f"{key}_VBIAS") if cfg == "CG" else st.session_state[f"{key}_VBIAS"])
        dn, sn = ("R_D", "R_S") if fam == "MOS" else ("R_C", "R_E")
        Rd = st.slider(f"{dn} (kΩ)", 0.0, 20.0, step=0.1, key=f"{key}_Rd")
        Rs = st.slider(f"{sn} (kΩ)", 0.0, 10.0, step=0.1, key=f"{key}_Rs")
        RB = (st.slider("R_B (kΩ)", 0.0, 300.0, step=1.0, key=f"{key}_RB")
              if fam == "BJT" else 0.0)
        if st.button("Reset this circuit", key=f"{key}_reset"):
            for nm, v in dflt.items():
                st.session_state[f"{key}_{nm}"] = v
            st.rerun(scope="fragment")

    s_amp = dict(VDD=VDD, Rd=Rd, Rs=Rs, RB=RB, vI=vI, VBIAS=VBIAS)
    r = solve_amp(fam, pol, cfg, s_amp, P)
    cls = region_class(fam, r["region"])

    # voltage transfer curve (sweep v_I, warm-started for the BJT)
    vis = np.linspace(0, VDD, 201)
    if fam == "MOS":   # whole sweep in one vectorised solve
        rr = solve_amp(fam, pol, cfg, dict(s_amp, vI=vis), P)
        vo = np.asarray(rr["vO"])
        cl = [region_class(fam, g) for g in rr["region"]]
    else:
        vo, cl = [], []
        warm = None
        for v in vis:
            rr = solve_amp(fam, pol, cfg, dict(s_amp, vI=float(v)), P, warm)
            warm = rr.get("x")
            vo.append(rr["vO"])
            cl.append(region_class(fam, rr["region"]))
        vo = np.array(vo)

    # small-signal gain at Q: slope of the transfer curve, and the textbook formula
    dv = 1e-3
    rp = solve_amp(fam, pol, cfg, dict(s_amp, vI=vI + dv), P, r.get("x"))
    rm = solve_amp(fam, pol, cfg, dict(s_amp, vI=vI - dv), P, r.get("x"))
    slope = (rp["vO"] - rm["vO"]) / (2 * dv)
    Iq = abs(r["I"])
    if fam == "MOS":
        vov = r["VGS"] - P["Vt"]
        gm = P["k"] * vov * (1 + P["lam"] * r["VDS"]) if r["region"] == "saturation" else float("nan")
        ro = 1 / (P["lam"] * Iq) if (P["lam"] > 0 and Iq > 0) else float("inf")
        if cfg == "CS":
            A_f, A_tex = -gm * Rd / (1 + gm * Rs), r"A_v=-\dfrac{g_mR_D}{1+g_mR_S}"
        elif cfg == "CD":
            A_f, A_tex = gm * Rs / (1 + gm * Rs), r"A_v=\dfrac{g_mR_S}{1+g_mR_S}"
        else:
            A_f, A_tex = gm * Rd / (1 + gm * Rs), r"A_v=\dfrac{g_mR_D}{1+g_mR_S}"
    else:
        gm = Iq / VT if r["region"] == "active" else float("nan")
        rpi = P["beta"] / gm if gm > 0 else float("nan")
        ro = P["VA"] / Iq if Iq > 0 else float("inf")
        den = RB + rpi + (P["beta"] + 1) * Rs
        if cfg == "CS":
            A_f, A_tex = -P["beta"] * Rd / den, r"A_v=-\dfrac{\beta R_C}{R_B+r_\pi+(\beta+1)R_E}"
        elif cfg == "CD":
            A_f, A_tex = (P["beta"] + 1) * Rs / den, r"A_v=\dfrac{(\beta+1)R_E}{R_B+r_\pi+(\beta+1)R_E}"
        else:
            A_f, A_tex = P["beta"] * Rd / den, r"A_v=\dfrac{\beta R_C}{R_B+r_\pi+(\beta+1)R_E}"

    with col_ctrl:
        rname = REGION_LABELS[cls]
        region_box(fam, r["region"], f"**{rname.capitalize()}**  ·  {N['I']} = {fmt_I(Iq)}  ·  v_O = {r['vO']:.2f} V")
        sk = tuple(round(s_amp[k], 3) for k in ("VDD", "Rd", "Rs", "RB", "vI", "VBIAS"))
        rk = tuple(round(r[k], 2) for k in ("VG", "VD", "VS", "vO")) + (float(f"{Iq:.4g}"),)
        st.markdown(amp_svg(fam, pol, cfg, sk, rk), unsafe_allow_html=True)

    with col_plot:
        figv = go.Figure()
        add_region_colored(figv, vis, vo, cl, REGION_LABELS)
        if cfg == "CD":
            figv.add_trace(go.Scatter(x=[0, VDD], y=[0, VDD], mode="lines", name="v_O = v_I",
                                      line=dict(color="gray", dash="dot", width=1)))
        figv.add_vline(x=vI, line_dash="dot", line_color="gray")
        figv.add_trace(go.Scatter(x=[vI], y=[r["vO"]], mode="markers",
                                  marker=dict(size=14, color=CLASS_COLOR[cls], line=dict(color="black", width=1)),
                                  name=f"Q: v_I = {vI:.2f} V, v_O = {r['vO']:.2f} V"))
        base_layout(figv, f"Voltage transfer characteristic — {CFG_NAMES[fam][cfg]}",
                    "Input v_I (V)", "Output v_O (V)", 420, [0, VDD], [-0.02 * VDD, VDD * 1.03])
        st.plotly_chart(figv, width="stretch")

        # load line on the output characteristics
        figl = go.Figure()
        Vint = r["Vint"]
        vx = np.linspace(0, max(Vint, 0.1), 300)
        vctl_q = r["VGS"]
        if fam == "MOS":
            dev_curve = mos_id(vctl_q, vx, P)
            Rload = Rd + Rs
        else:
            dev_curve = np.maximum([bjt_currents(vctl_q, vctl_q - v, P)[0] for v in vx], 0.0)
            alpha = r["I"] / r["IE"] if r["IE"] > 0 else P["beta"] / (P["beta"] + 1)
            Rload = Rd + Rs / alpha
        figl.add_trace(go.Scatter(x=vx, y=dev_curve, mode="lines", line=dict(color="royalblue", width=3),
                                  name=f"device: {N['ctl_u']} = {vctl_q:.3f} V"))
        if Rload > 0:
            figl.add_trace(go.Scatter(x=[0, Vint], y=[Vint / Rload, 0], mode="lines",
                                      line=dict(color="black", width=2, dash="dash"),
                                      name=f"load line: slope −1/{Rload:.2f} kΩ"))
        ytop = max(float(np.max(dev_curve)), Iq) * 1.3 + 1e-3
        if Rload > 0 and Vint / Rload < 4 * ytop:
            ytop = max(ytop, Vint / Rload * 1.1)
        figl.add_trace(go.Scatter(x=[r["VDS"]], y=[Iq], mode="markers",
                                  marker=dict(size=14, color=CLASS_COLOR[cls], line=dict(color="black", width=1)),
                                  name="Q point"))
        base_layout(figl, "Load line on the output characteristic",
                    f"{hsub(N['out_u'])} (V)", f"{hsub(N['I_u'])} (mA)", 380,
                    [0, max(Vint, 0.1) * 1.05], [0, ytop])
        st.plotly_chart(figl, width="stretch")

    # numbers
    st.markdown("#### Bias point")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(N["I_u"], fmt_I(Iq))
    c2.metric(f"{N['g']} / {N['s']} / {N['d']}  (V)", f"{r['VG']:.2f} / {r['VS']:.2f} / {r['VD']:.2f}")
    c3.metric(N["ctl_u"], f"{r['VGS']:.3f} V")
    c4.metric(N["out_u"], f"{r['VDS']:.3f} V")

    left, right = st.columns(2)
    with left:
        st.markdown("**How the bias point is found**")
        n = pol == "n"
        Xs = ("0" if n else (r"V_{DD}" if fam == "MOS" else r"V_{CC}")) if cfg != "CG" else "v_I"
        if fam == "MOS":
            if n:
                st.latex(rf"V_{{GS}} = V_G - I_D R_S - {Xs} = {r['VGS']:.3f}\ \text{{V}}")
                st.latex(rf"V_{{DS}} = V_{{DD}} - I_D(R_D+R_S) - {Xs} = {r['VDS']:.3f}\ \text{{V}}")
            else:
                st.latex(rf"V_{{SG}} = {Xs} - I_D R_S - V_G = {r['VGS']:.3f}\ \text{{V}}")
                st.latex(rf"V_{{SD}} = {Xs} - I_D(R_S+R_D) = {r['VDS']:.3f}\ \text{{V}}")
            st.caption("Both equations hold together with the I_D equation of the present region; "
                       "the app solves them numerically. Check the region against "
                       f"{N['out_u']} ≥ V_OV for saturation.")
        else:
            if n:
                st.latex(rf"V_{{B,in}} - {Xs} = I_B R_B + V_{{BE}} + I_E R_E")
            else:
                st.latex(rf"{Xs} - V_{{B,in}} = I_E R_E + V_{{EB}} + I_B R_B")
            Vbin = VBIAS if cfg == "CG" else vI
            Xv = (0 if n else VDD) if cfg != "CG" else vI
            drive = (Vbin - Xv) if n else (Xv - Vbin)
            IB7 = (drive - 0.7) / (RB + (P["beta"] + 1) * Rs) if (RB + (P["beta"] + 1) * Rs) > 0 else float("nan")
            if IB7 > 0:
                IC7 = P["beta"] * IB7
                st.markdown("Course estimate with the 0.7 V model:")
                st.latex(rf"I_B=\frac{{{drive:.2f}-0.7}}{{R_B+(\beta+1)R_E}}={IB7 * 1e3:.2f}\ \mu\text{{A}},"
                         rf"\quad I_C=\beta I_B={IC7:.3f}\ \text{{mA}}")
                st.caption(f"Exponential model gives I_C = {Iq:.3f} mA with "
                           f"{N['ctl_u']} = {r['VGS']:.3f} V. The 0.7 V estimate is valid only "
                           "if the transistor is really in the active region.")
            else:
                st.caption("Course estimate (0.7 V): the base–emitter junction does not turn on, so I_C ≈ 0.")
    with right:
        st.markdown("**Small-signal view at this bias point**")
        if cls == "amp":
            if fam == "MOS":
                st.markdown(f"g<sub>m</sub> = {gm:.3f} mA/V &nbsp;·&nbsp; r<sub>o</sub> = {fmt_R(ro)}",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"g<sub>m</sub> = {gm:.2f} mA/V &nbsp;·&nbsp; r<sub>π</sub> = {fmt_R(rpi)}"
                            f" &nbsp;·&nbsp; r<sub>o</sub> = {fmt_R(ro)}", unsafe_allow_html=True)
            st.latex(A_tex + rf"= {A_f:.2f}\ \text{{V/V}}\quad(r_o\ \text{{neglected}})")
            st.markdown(f"Slope of the transfer curve at Q: **{slope:.2f} V/V** "
                        "(exact, includes r<sub>o</sub> and the full nonlinear model).",
                        unsafe_allow_html=True)
        else:
            st.warning(f"The transistor is in **{REGION_LABELS[cls]}**, so this circuit is not "
                       f"amplifying here (slope of the transfer curve = {slope:.2f} V/V). "
                       f"Move v_I until the Q point sits on the green part of the curve.")

    # ── small-signal equivalent circuit: hidden until the student asks for it ──
    st.markdown("#### Small-signal equivalent circuit")
    st.caption("Draw the small-signal model of this circuit yourself first, then click a model to check it. "
               "Click the selected model again to hide it.")
    model = st.segmented_control("Equivalent model", ["pi", "T"], selection_mode="single", default=None,
                                 format_func=lambda m: "Hybrid-π model" if m == "pi" else "T model",
                                 key="ss_model", label_visibility="collapsed")
    if model is not None:
        if cls != "amp":
            st.warning(f"No small-signal model here: the transistor is in {REGION_LABELS[cls]}. "
                       f"The model is valid only in {REGION_LABELS['amp'].split(' ')[0]}.")
        else:
            mos = fam == "MOS"
            beta = P.get("beta", float("nan"))
            alpha = beta / (beta + 1) if not mos else 1.0
            re_ = alpha / gm
            rpi_v = rpi if not mos else float("nan")
            vk = (round(gm, 4), round(rpi_v, 3) if not mos else 0.0, round(re_, 4), round(ro, 2) if math.isfinite(ro) else ro,
                  round(Rd, 2), round(Rs, 2), round(RB, 1))
            c_draw, c_num = st.columns([3, 2])
            with c_draw:
                st.markdown(small_signal_svg(fam, cfg, model, vk), unsafe_allow_html=True)
                st.caption("DC supplies and DC bias sources become ac ground. "
                           + ("The same model is used for PMOS. " if mos else "The same model is used for pnp. ")
                           + ("" if math.isfinite(ro) else "r_o = ∞ here (λ = 0), so it is not drawn."
                              if mos else "r_o = ∞ here (Early effect off), so it is not drawn."))
            with c_num:
                st.markdown("**Model parameters at this bias point**")
                if mos:
                    rows = [("g<sub>m</sub>", f"{gm:.3f} mA/V"), ("1/g<sub>m</sub>", fmt_R(1 / gm)),
                            ("r<sub>o</sub>", fmt_R(ro))]
                else:
                    rows = [("g<sub>m</sub> = I<sub>C</sub>/V<sub>T</sub>", f"{gm:.2f} mA/V"),
                            ("r<sub>π</sub> = β/g<sub>m</sub>", fmt_R(rpi_v)),
                            ("r<sub>e</sub> = α/g<sub>m</sub>", fmt_R(re_)),
                            ("α = β/(β+1)", f"{alpha:.4f}"), ("r<sub>o</sub>", fmt_R(ro))]
                st.markdown("| Parameter | Value |\n|:--|--:|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows),
                            unsafe_allow_html=True)
                inf = float("inf")
                if mos:
                    Rin = {"CS": inf, "CD": inf, "CG": Rs + 1 / gm}[cfg]
                    Rout = {"CS": Rd, "CD": (Rs / (1 + gm * Rs)), "CG": Rd}[cfg]
                    rin_tex = {"CS": r"\infty", "CD": r"\infty", "CG": r"R_S+\tfrac{1}{g_m}"}[cfg]
                    rout_tex = {"CS": "R_D", "CD": r"R_S\parallel\tfrac{1}{g_m}", "CG": "R_D"}[cfg]
                else:
                    Rce = RB + rpi_v + (beta + 1) * Rs
                    Rin = {"CS": Rce, "CD": Rce, "CG": Rs + (RB + rpi_v) / (beta + 1)}[cfg]
                    rx = (RB + rpi_v) / (beta + 1)
                    Rout = {"CS": Rd, "CD": (Rs * rx / (Rs + rx) if Rs > 0 else 0.0), "CG": Rd}[cfg]
                    rin_tex = {"CS": r"R_B+r_\pi+(\beta+1)R_E", "CD": r"R_B+r_\pi+(\beta+1)R_E",
                               "CG": r"R_E+\frac{R_B+r_\pi}{\beta+1}"}[cfg]
                    rout_tex = {"CS": "R_C", "CD": r"R_E\parallel\frac{R_B+r_\pi}{\beta+1}", "CG": "R_C"}[cfg]
                st.markdown("**Results from the model** (r<sub>o</sub> neglected)", unsafe_allow_html=True)
                st.latex(A_tex + rf"= {A_f:.2f}\ \text{{V/V}}")
                if math.isfinite(Rin):
                    st.latex(rf"R_{{in}}={rin_tex}=\text{{{fmt_R(Rin)}}}")
                else:
                    st.latex(r"R_{in}=\infty\quad\text{(no gate current)}")
                st.latex(rf"R_{{out}}={rout_tex}=\text{{{fmt_R(Rout)}}}")
                st.caption(f"Software check: slope of the transfer curve at Q = {slope:.2f} V/V "
                           "(full nonlinear model, includes r_o). R_in is the resistance seen by v_i; "
                           "R_out is seen at the v_o node with v_i = 0.")

    with st.expander("What to try", expanded=False):
        if cfg == "CS":
            st.markdown(f"""
- Sweep v_I from 0 to the supply. The {('CS' if fam == 'MOS' else 'CE')} stage goes cutoff → {REGION_LABELS['amp'].split(' ')[0]} → {REGION_LABELS['ohmic']}. The output swings opposite to the input: the gain is negative.
- The gain is the slope of the curve. It is largest in the green part, and the bias point is normally placed in the middle of it to leave room for the signal to swing both ways.
- Add {sn}: the curve becomes less steep but straighter (more linear). Compare the formula with the slope.
- Switch to p-type: the curve looks the same, but the device sits at the top of the circuit and the transistor turns **on** when v_I goes **down**.""")
        elif cfg == "CD":
            st.markdown(f"""
- The output follows the input, about one {'V_GS' if fam == 'MOS' else '0.7 V'} lower (n-type) or higher (p-type). Compare with the dotted v_O = v_I line.
- The gain is just below 1. Increase {sn}: the gain approaches 1.
- The follower does not amplify voltage; its job is a low output resistance (a buffer).""")
        else:
            st.markdown(f"""
- The input enters at the {'source' if fam == 'MOS' else 'emitter'}. Raising v_I (n-type) reduces {N['ctl_u']}, so the current falls and v_O **rises**: the gain is positive.
- Change V_BIAS to move the whole curve left or right.
- Make {sn} small: the gain grows, because the input current is no longer limited by {sn}.""")

with tab2:
    render_tab2()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — n-type vs p-type
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment
def render_tab3():
    st.subheader(f"{'NMOS vs PMOS' if fam == 'MOS' else 'npn vs pnp'} — the same device with every sign reversed")
    col_ctrl, col_plot = st.columns([1, 2])
    with col_ctrl:
        if fam == "MOS":
            mag_c = st.slider("Gate drive |V_GS| (V)", 0.0, 3.0, 1.5, step=0.01, key="t3_c")
        else:
            mag_c = st.slider("Base drive |V_BE| (V)", 0.50, 0.80, 0.68, step=0.002, key="t3_c", format="%.3f")
        mag_o = st.slider("|V_DS|" if fam == "MOS" else "|V_CE|", 0.0, 5.0, 2.5, step=0.01, key="t3_o")
        if fam == "MOS":
            Im = float(mos_id(mag_c, mag_o, P))
            rg = mos_region(mag_c, mag_o, P)
        else:
            Im = bjt_currents(mag_c, mag_c - mag_o, P)[0]
            rg = bjt_region(mag_c, mag_c - mag_o)
        region_box(fam, rg, f"Both devices: **{REGION_LABELS[region_class(fam, rg)]}**")
        if fam == "MOS":
            st.markdown(f"""
| | NMOS | PMOS |
|:--|--:|--:|
| V<sub>GS</sub> | {mag_c:+.2f} V | {-mag_c:+.2f} V |
| V<sub>DS</sub> | {mag_o:+.2f} V | {-mag_o:+.2f} V |
| V<sub>t</sub> | +{P['Vt']:.2f} V | −{P['Vt']:.2f} V |
| I<sub>D</sub> into drain | {fmt_I(Im)} | −{fmt_I(Im)} |
| Turns on when | V<sub>GS</sub> > V<sub>tn</sub> | V<sub>GS</sub> < V<sub>tp</sub> (V<sub>SG</sub> > \\|V<sub>tp</sub>\\|) |
| Carriers | electrons | holes |
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
| | npn | pnp |
|:--|--:|--:|
| V<sub>BE</sub> | {mag_c:+.3f} V | {-mag_c:+.3f} V |
| V<sub>CE</sub> | {mag_o:+.2f} V | {-mag_o:+.2f} V |
| I<sub>C</sub> into collector | {fmt_I(Im)} | −{fmt_I(Im)} |
| Emitter arrow | points out | points in |
| Carriers through base | electrons | holes |
""", unsafe_allow_html=True)
        st.caption("Same parameters for both devices, so the curves are point-symmetric about the origin. "
                   "In a real process the p-type device is weaker (lower hole mobility) unless it is made wider.")
    with col_plot:
        figc = go.Figure()
        vx = np.linspace(0, 5, 301)
        if fam == "MOS":
            yy = mos_id(mag_c, vx, P)
        else:
            yy = np.maximum([bjt_currents(mag_c, mag_c - v, P)[0] for v in vx], 0.0)
        figc.add_trace(go.Scatter(x=vx, y=yy, mode="lines", line=dict(color="royalblue", width=3),
                                  name="NMOS" if fam == "MOS" else "npn"))
        figc.add_trace(go.Scatter(x=-vx, y=-yy, mode="lines", line=dict(color="crimson", width=3),
                                  name="PMOS" if fam == "MOS" else "pnp"))
        figc.add_trace(go.Scatter(x=[mag_o, -mag_o], y=[Im, -Im], mode="markers",
                                  marker=dict(size=12, color=["royalblue", "crimson"], line=dict(color="black", width=1)),
                                  name="operating points"))
        ym = max(float(yy.max()), 1e-3) * 1.15
        base_layout(figc, "Signed output characteristics",
                    "V<sub>DS</sub> (V)" if fam == "MOS" else "V<sub>CE</sub> (V)",
                    "I<sub>D</sub> into drain (mA)" if fam == "MOS" else "I<sub>C</sub> into collector (mA)",
                    520, [-5.3, 5.3], [-ym, ym])
        figc.add_annotation(x=2.6, y=ym * 0.35, text="1st quadrant: n-type", showarrow=False,
                            font=dict(color="royalblue"))
        figc.add_annotation(x=-2.6, y=-ym * 0.35, text="3rd quadrant: p-type", showarrow=False,
                            font=dict(color="crimson"))
        st.plotly_chart(figc, width="stretch")


with tab3:
    render_tab3()
