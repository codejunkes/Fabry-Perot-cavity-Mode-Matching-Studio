# Fabry-Perot-cavity-Mode-Matching-Studio
Lens setup for Fabry Perot cavity Mode Matching and coupling efficiency optimization
<div align="center">
  <h1>Advanced FP Mode-Matching Studio</h1>
  <p><b>A computational toolkit for optical bench layout optimization and Fabry-Perot cavity mode-matching.</b></p>
</div>

## 🔬 Overview
The **Advanced FP Mode-Matching Studio** is a dedicated optoelectronics engineering tool designed to automate the design and tolerance analysis of lens configurations. By leveraging rigorous ABCD matrix propagation, overlap integrals, and Monte Carlo tolerance simulations, this suite provides a robust platform for preparing physical optical bench layouts. 

It is specifically engineered to establish a solid mode-matched foundation before engaging highly sensitive frequency stabilization protocols, such as establishing **Pound-Drever-Hall (PDH) locking** between a commercial laser and a Fabry-Perot reference cavity.

## ✨ Key Features
- **Multithreaded Optimization**: Utilizes combinatorial grid search paired with L-BFGS-B gradient descent to find optimal lens placements that maximize coupling efficiency.
- **Monte Carlo Tolerance Analysis**: Simulates real-world mechanical and optical imperfections (e.g., machining defects, alignment drift) to compute worst-case efficiency limits.
- **Interactive GUI**: Built with Tkinter and Matplotlib, offering dynamic rendering of beam envelopes ($w(z)$) and direct comparison of top-ranked configurations.
- **Comprehensive Optical Modeling**:
  - Supports free-space and fiber-coupled laser sources.
  - Calculates stable cavity eigenmodes for hemispherical Fabry-Perot geometries.
  - Automatically corrects for optical path length shifts induced by tilted vacuum viewports.
- **State Persistence**: Serializes bench settings into XML (`last_settings.xml`) for seamless continuity across experimental sessions.

## 📂 Project Structure
- `gui.py` — The primary Tkinter-based graphical front-end for user interaction, visual rendering, and state management.
- `optimization.py` — The physics and mathematics engine containing complex beam parameter ($q$) propagation, eigenmode derivation, and the optimization algorithms.
- `setup_data.py` — Defines the fundamental data structures and default optical parameters using Python `dataclasses`.
- `last_settings.xml` — (Auto-generated) Stores the most recent system configuration to restore your workspace.

## 🚀 Installation & Usage
This project requires Python 3.8+ and standard scientific libraries.

### 1. Install Dependencies
Ensure you have `numpy`, `scipy`, and `matplotlib` installed:
```bash
pip install numpy scipy matplotlib
```

### 2. Run the Application
Execute the main GUI script to launch the Mode-Matching Studio:
```bash
python gui.py
```

## 🧠 Physics Engine Details
The computational core relies on the **paraxial approximation of Gaussian beams**. 
- **Beam Tracking**: Tracked along the optical axis ($z$) using the complex parameter $q$.
- **Propagation**: Transformed via ABCD matrices for free space and thin lenses.
- **Efficiency metric**: Mode-matching quality is quantified by the spatial overlap integral ($\eta$) between the incident beam and the FP cavity's stable eigenmode.

## 🛠️ Built With
- **Python** (Core Logic)
- **Tkinter** (GUI)
- **SciPy** (L-BFGS-B Optimization)
- **NumPy & Matplotlib** (Data structures & visualization)

---
*Designed for applied physics and optoelectronics research applications.*
