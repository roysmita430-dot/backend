import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from scipy.interpolate import UnivariateSpline
import sympy as sp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spline AI Curve Fitter", layout="wide")
st.title("Draw Anything → AI Generates Equation (Spline Exact Fit)")

# -----------------------------
# Canvas for drawing
# -----------------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=2,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Extract points
# -----------------------------
if canvas_result.image_data is not None:
    img_data = canvas_result.image_data[:, :, 0]
    y_idx, x_idx = np.nonzero(img_data < 250)
    if len(x_idx) > 5:
        # normalize x and y to [0,1]
        x = x_idx / max(x_idx)
        y = 1 - (y_idx / max(y_idx))

        st.subheader("Sampled Points")
        st.write("Number of points:", len(x))

        # -----------------------------
        # Fit spline
        # -----------------------------
        try:
            # s=0 for exact fit, increase s>0 to smooth
            spline = UnivariateSpline(x, y, s=0)
            y_fit = spline(x)

            # -----------------------------
            # Symbolic representation using sympy
            # -----------------------------
            xs = sp.Symbol('x')
            # Build a piecewise polynomial approximation (as latex)
            # Using spline's knots and coefficients
            coeffs = spline.get_coeffs()
            t = spline.get_knots()
            # Construct symbolic cubic splines for each interval
            sym_eq_parts = []
            for i in range(len(t)-1):
                # interval i
                x0 = t[i]
                x1 = t[i+1]
                # spline derivative coefficients for interval
                poly = sp.Poly(spline(x0) + ((spline(x1)-spline(x0))/(x1-x0))*(xs-x0), xs)
                sym_eq_parts.append(sp.Piecewise((poly.as_expr(), (xs>=x0) & (xs<=x1))))

            final_sym_eq = sum(sym_eq_parts)
            st.subheader("Equation in LaTeX (Piecewise Spline)")
            st.latex(sp.latex(final_sym_eq))

            # -----------------------------
            # Plot
            # -----------------------------
            fig, ax = plt.subplots()
            ax.scatter(x, y, s=2, color='blue', label='Drawn points')
            x_fit = np.linspace(0, 1, 400)
            y_fit_curve = spline(x_fit)
            ax.plot(x_fit, y_fit_curve, color='red', label='Fitted Spline')
            ax.set_title("Drawn Points vs Fitted Spline")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Spline fitting failed: {e}")
