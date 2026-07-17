# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import xml.etree.ElementTree as ET
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from setup_data import OpticalSetup
from optimization import run_multithreaded_optimization, generate_beam_profile_data

class ModeMatchingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced FP Mode-Matching Studio")
        self.root.geometry("1100x950")
        self.setup = OpticalSetup()
        self.settings_file = "fp_cav_beam_coupling/last_settings.xml"
        
        self.use_fiber_var = tk.BooleanVar(value=False)
        self.use_vp_var = tk.BooleanVar(value=False)
        self.lens_enabled = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), 
                             tk.BooleanVar(value=False), tk.BooleanVar(value=False)]
        self.lens_fixed = [tk.BooleanVar(value=False) for _ in range(4)]
        self.lens_f_entries = []
        self.dist_min_entries = []
        self.dist_max_entries = []
        
        self.create_widgets()
        self.load_settings_from_xml()
        self.toggle_source_mode()
        self.toggle_viewport_mode()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.main_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_frame, text="System Configuration & Console")
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Best Gaussian Beam Sketch")
        self.comp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comp_frame, text="Tolerance Analysis & Comparisons")
        
        self.setup_plot_area()
        self.setup_comp_area()
        self.setup_config_area()

    def setup_config_area(self):
        # Allow vertical scrolling for the config area
        canvas = tk.Canvas(self.main_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row_idx = 0
        
        # --- Section 1: Source ---
        ttk.Label(frame, text="1. Source Parameters (Nominal ± Tolerance)", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=6, sticky='w', pady=(0, 5)); row_idx+=1
        ttk.Label(frame, text="Wavelength (nm):").grid(row=row_idx, column=0, sticky='w')
        self.ent_wl = ttk.Entry(frame, width=8); self.ent_wl.insert(0, "1550"); self.ent_wl.grid(row=row_idx, column=1, sticky='w')
        ttk.Label(frame, text="±").grid(row=row_idx, column=2)
        self.ent_twl = ttk.Entry(frame, width=6); self.ent_twl.insert(0, "0.1"); self.ent_twl.grid(row=row_idx, column=3, sticky='w'); row_idx+=1
        
        ttk.Checkbutton(frame, text="Use Fiber Coupler Calculator", variable=self.use_fiber_var, command=self.toggle_source_mode).grid(row=row_idx, column=0, columnspan=6, sticky='w', pady=(5,0)); row_idx+=1
        
        self.man_frame = ttk.LabelFrame(frame, text="Manual Source Input")
        self.man_frame.grid(row=row_idx, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        ttk.Label(self.man_frame, text="Beam Waist w0 (mm):").grid(row=0, column=0, sticky='w')
        self.ent_w0 = ttk.Entry(self.man_frame, width=6); self.ent_w0.insert(0, "0.5"); self.ent_w0.grid(row=0, column=1)
        ttk.Label(self.man_frame, text="±").grid(row=0, column=2)
        self.ent_tw0 = ttk.Entry(self.man_frame, width=5); self.ent_tw0.insert(0, "0.01"); self.ent_tw0.grid(row=0, column=3)
        
        self.fib_frame = ttk.LabelFrame(frame, text="Fiber Coupler Input")
        self.fib_frame.grid(row=row_idx, column=2, columnspan=4, sticky='nsew', padx=5, pady=5); row_idx+=1
        ttk.Label(self.fib_frame, text="Core / MFD (µm):").grid(row=0, column=0, sticky='w')
        self.ent_fcore = ttk.Entry(self.fib_frame, width=6); self.ent_fcore.insert(0, "10.4"); self.ent_fcore.grid(row=0, column=1)
        ttk.Label(self.fib_frame, text="±").grid(row=0, column=2)
        self.ent_tfcore = ttk.Entry(self.fib_frame, width=5); self.ent_tfcore.insert(0, "0.1"); self.ent_tfcore.grid(row=0, column=3)
        
        ttk.Label(self.fib_frame, text="Coupler f (mm):").grid(row=1, column=0, sticky='w')
        self.ent_fcoup = ttk.Entry(self.fib_frame, width=6); self.ent_fcoup.insert(0, "11.0"); self.ent_fcoup.grid(row=1, column=1)
        ttk.Label(self.fib_frame, text="±").grid(row=1, column=2)
        self.ent_tfcoup = ttk.Entry(self.fib_frame, width=5); self.ent_tfcoup.insert(0, "0.1"); self.ent_tfcoup.grid(row=1, column=3)

        # --- Section 1.5: Viewport ---
        ttk.Separator(frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=6, sticky='ew', pady=5); row_idx+=1
        ttk.Label(frame, text="2. Vacuum Viewport Window", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=6, sticky='w'); row_idx+=1
        
        ttk.Checkbutton(frame, text="Include Viewport Window", variable=self.use_vp_var, command=self.toggle_viewport_mode).grid(row=row_idx, column=0, columnspan=6, sticky='w'); row_idx+=1
        
        self.vp_frame = ttk.Frame(frame)
        self.vp_frame.grid(row=row_idx, column=0, columnspan=6, sticky='w', pady=2); row_idx+=1
        
        ttk.Label(self.vp_frame, text="Thickness (mm):").grid(row=0, column=0, sticky='w')
        self.ent_thick = ttk.Entry(self.vp_frame, width=6); self.ent_thick.insert(0, "4.0"); self.ent_thick.grid(row=0, column=1)
        ttk.Label(self.vp_frame, text="±").grid(row=0, column=2)
        self.ent_tthick = ttk.Entry(self.vp_frame, width=5); self.ent_tthick.insert(0, "0.1"); self.ent_tthick.grid(row=0, column=3, padx=(0,10))
        
        ttk.Label(self.vp_frame, text="Tilt Angle (deg):").grid(row=0, column=4, sticky='w')
        self.ent_ang = ttk.Entry(self.vp_frame, width=6); self.ent_ang.insert(0, "3.0"); self.ent_ang.grid(row=0, column=5)
        ttk.Label(self.vp_frame, text="±").grid(row=0, column=6)
        self.ent_tang = ttk.Entry(self.vp_frame, width=5); self.ent_tang.insert(0, "0.1"); self.ent_tang.grid(row=0, column=7, padx=(0,10))
        
        ttk.Label(self.vp_frame, text="Refractive Index (n):").grid(row=1, column=0, sticky='w', pady=(5,0))
        self.ent_n = ttk.Entry(self.vp_frame, width=6); self.ent_n.insert(0, "1.5168"); self.ent_n.grid(row=1, column=1, pady=(5,0))
        
        ttk.Label(self.vp_frame, text="Dist. from Cavity (mm):").grid(row=1, column=4, sticky='w', pady=(5,0))
        self.ent_vdist = ttk.Entry(self.vp_frame, width=6); self.ent_vdist.insert(0, "20.0"); self.ent_vdist.grid(row=1, column=5, pady=(5,0))
        ttk.Label(self.vp_frame, text="±").grid(row=1, column=6, pady=(5,0))
        self.ent_tvdist = ttk.Entry(self.vp_frame, width=5); self.ent_tvdist.insert(0, "1.0"); self.ent_tvdist.grid(row=1, column=7, pady=(5,0), padx=(0,10))

        # --- Section 2: Cavity ---
        ttk.Separator(frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=6, sticky='ew', pady=5); row_idx+=1
        ttk.Label(frame, text="3. Hemispherical Cavity Parameters", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=6, sticky='w'); row_idx+=1
        
        ttk.Label(frame, text="Cavity Length (mm):").grid(row=row_idx, column=0, sticky='w')
        self.ent_cl = ttk.Entry(frame, width=8); self.ent_cl.insert(0, "50.0"); self.ent_cl.grid(row=row_idx, column=1, sticky='w')
        ttk.Label(frame, text="±").grid(row=row_idx, column=2)
        self.ent_tcl = ttk.Entry(frame, width=6); self.ent_tcl.insert(0, "0.1"); self.ent_tcl.grid(row=row_idx, column=3, sticky='w')
        
        ttk.Label(frame, text="Curved Mirror ROC (mm):").grid(row=row_idx, column=4, sticky='e')
        self.ent_roc = ttk.Entry(frame, width=8); self.ent_roc.insert(0, "100.0"); self.ent_roc.grid(row=row_idx, column=5, sticky='w')
        ttk.Label(frame, text="±").grid(row=row_idx, column=6)
        self.ent_troc = ttk.Entry(frame, width=6); self.ent_troc.insert(0, "1.0"); self.ent_troc.grid(row=row_idx, column=7, sticky='w'); row_idx+=1
        
        ttk.Label(frame, text="Flat Mirror Flatness Defect (± Min ROC mm):").grid(row=row_idx, column=0, columnspan=2, sticky='w')
        self.ent_tr1 = ttk.Entry(frame, width=8); self.ent_tr1.insert(0, "10000"); self.ent_tr1.grid(row=row_idx, column=2, columnspan=2, sticky='w'); row_idx+=1

        # --- Section 3: Lenses ---
        ttk.Separator(frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=6, sticky='ew', pady=5); row_idx+=1
        
        hdr_frame = ttk.Frame(frame)
        hdr_frame.grid(row=row_idx, column=0, columnspan=6, sticky='w')
        ttk.Label(hdr_frame, text="4. Lens Configuration", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=(0, 20))
        ttk.Label(hdr_frame, text="Focal Length Tolerance (± %):").grid(row=0, column=1, sticky='e')
        self.ent_tf = ttk.Entry(hdr_frame, width=6); self.ent_tf.insert(0, "1.0"); self.ent_tf.grid(row=0, column=2, sticky='w')
        row_idx+=1
        
        lens_frame = ttk.Frame(frame)
        lens_frame.grid(row=row_idx, column=0, columnspan=6, sticky='w', pady=2); row_idx+=1
        for i in range(4):
            cb_en = ttk.Checkbutton(lens_frame, text=f"Enable Lens {i+1}", variable=self.lens_enabled[i], command=self.refresh_distances)
            cb_en.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            cb_fix = ttk.Checkbutton(lens_frame, text="Lock Focal Length:", variable=self.lens_fixed[i])
            cb_fix.grid(row=i, column=1, sticky='w', padx=5)
            ent_f = ttk.Entry(lens_frame, width=8); ent_f.insert(0, "50.0")
            ent_f.grid(row=i, column=2, sticky='w', padx=5)
            self.lens_f_entries.append(ent_f)

        # --- Section 4: Distances ---
        ttk.Separator(frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=6, sticky='ew', pady=5); row_idx+=1
        dist_hdr = ttk.Frame(frame)
        dist_hdr.grid(row=row_idx, column=0, columnspan=6, sticky='w')
        ttk.Label(dist_hdr, text="5. Travel Bounds (Min, Max) mm", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=(0, 20))
        ttk.Label(dist_hdr, text="Positional Tolerance (± mm):").grid(row=0, column=1, sticky='e')
        self.ent_tpos = ttk.Entry(dist_hdr, width=6); self.ent_tpos.insert(0, "0.2"); self.ent_tpos.grid(row=0, column=2, sticky='w')
        row_idx+=1
        
        self.dist_frame = ttk.Frame(frame)
        self.dist_frame.grid(row=row_idx, column=0, columnspan=6, sticky='w'); row_idx+=1
        self.refresh_distances() 

        # --- Section 5: Config ---
        ttk.Separator(frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=6, sticky='ew', pady=5); row_idx+=1
        ttk.Label(frame, text="Optimization Pool (CSV mm):").grid(row=row_idx, column=0, columnspan=2, sticky='w')
        self.ent_lenses = ttk.Entry(frame, width=50); self.ent_lenses.insert(0, "25, 50, 75, 100, 150, 200")
        self.ent_lenses.grid(row=row_idx, column=2, columnspan=4, sticky='w'); row_idx+=1
        
        ttk.Label(frame, text="Timeout (sec):").grid(row=row_idx, column=0, sticky='w')
        self.ent_time = ttk.Entry(frame, width=8); self.ent_time.insert(0, "15"); self.ent_time.grid(row=row_idx, column=1, sticky='w'); row_idx+=1

        self.btn_run = ttk.Button(frame, text="EXECUTE OVERLAP INTEGRAL OPTIMIZATION", command=self.start_optimization)
        self.btn_run.grid(row=row_idx, column=0, columnspan=6, pady=5); row_idx+=1
        
        self.progress_bar = ttk.Progressbar(frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=row_idx, column=0, columnspan=6, sticky='ew'); row_idx+=1
        
        self.txt_output = tk.Text(frame, height=12, width=140, font=('Consolas', 9), background='#f8f9fa', wrap='none')
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=self.txt_output.xview)
        self.txt_output.configure(xscrollcommand=scroll_x.set)
        self.txt_output.grid(row=row_idx, column=0, columnspan=6, pady=5)
        row_idx+=1
        scroll_x.grid(row=row_idx, column=0, columnspan=6, sticky='ew')

    def toggle_source_mode(self):
        state = 'normal' if self.use_fiber_var.get() else 'disabled'
        inv_state = 'disabled' if self.use_fiber_var.get() else 'normal'
        self.ent_w0.config(state=inv_state); self.ent_tw0.config(state=inv_state)
        self.ent_fcore.config(state=state); self.ent_tfcore.config(state=state)
        self.ent_fcoup.config(state=state); self.ent_tfcoup.config(state=state)
        
    def toggle_viewport_mode(self):
        state = 'normal' if self.use_vp_var.get() else 'disabled'
        for child in self.vp_frame.winfo_children():
            child.configure(state=state)

    def setup_plot_area(self):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw(); self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_comp_area(self):
        self.fig_comp = Figure(figsize=(8, 6), dpi=100)
        self.ax_comp = self.fig_comp.add_subplot(111)
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=self.comp_frame)
        self.canvas_comp.draw(); self.canvas_comp.get_tk_widget().pack(fill='both', expand=True)

    def draw_sketch(self, result):
        self.ax.clear()
        z, w, lenses_z, flat_z, curved_z, vp_s, vp_e = generate_beam_profile_data(self.setup, result['focal_lengths'], result['distances'])
        
        self.ax.fill_between(z, w, -w, color='cyan', alpha=0.3, label='Best Nominal Beam')
        self.ax.plot(z, w, color='blue', linewidth=1)
        self.ax.plot(z, -w, color='blue', linewidth=1)
        
        for i, l_z in enumerate(lenses_z):
            self.ax.axvline(x=l_z, color='green', linestyle='-', linewidth=2, label=f'Lens {i+1} (f={result["focal_lengths"][i]})')
            
        if vp_s is not None and vp_e is not None:
            self.ax.axvspan(vp_s, vp_e, color='gray', alpha=0.4, label='Viewport Window')
            
        self.ax.axvline(x=flat_z, color='black', linestyle='-', linewidth=3, label='Flat Input Mirror')
        arc_y = np.linspace(-max(w)*1.5, max(w)*1.5, 100)
        arc_x = curved_z + (arc_y**2) / (2*self.setup.cavity_R2) 
        self.ax.plot(arc_x, arc_y, color='black', linewidth=3, label='Curved End Mirror')
        
        y_text_pos = max(w) * 1.2
        R_text = f"{result['achieved_R_flat']:.1f}" if result['achieved_R_flat'] != np.inf else "Infinity"
        flat_text = (f"FLAT MIRROR\n"
                     f"Nom. Eff: {result['nom_eff']*100:.2f}%\n"
                     f"Wrst Eff: {result['worst_eff']*100:.2f}%\n"
                     f"Worst Dev: ±{result['worst_dev_flat']*1000:.1f}µm")
        self.ax.text(flat_z - (max(z)*0.02), y_text_pos, flat_text, bbox=dict(facecolor='white', alpha=0.8), ha='right', va='bottom', fontsize=8)

        curved_text = (f"CURVED MIRROR\n"
                       f"Target w: {result['target_w_curved']:.4f} mm\n"
                       f"Achieved w: {result['achieved_w_curved']:.4f} mm\n"
                       f"Worst Dev: ±{result['worst_dev_curved']*1000:.1f}µm")
        self.ax.text(curved_z + (max(z)*0.02), y_text_pos, curved_text, bbox=dict(facecolor='white', alpha=0.8), ha='left', va='bottom', fontsize=8)

        self.ax.set_title(f"Optimized Alignment (Worst-Case MC Efficiency: {result['worst_eff']*100:.2f}%)")
        self.ax.set_xlabel("Distance Z (mm)"); self.ax.set_ylabel("Beam Radius w(z) (mm)")
        self.ax.legend(loc='lower right', fontsize='small')
        self.ax.set_ylim(-max(w)*2.0, max(w)*2.5) 
        self.ax.grid(True, alpha=0.4)
        self.canvas.draw()

    def draw_comparison(self, top_results):
        self.ax_comp.clear()
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        if not top_results: return

        best = top_results[0]
        _, _, _, best_flat_z, best_curved_z, vp_s, vp_e = generate_beam_profile_data(self.setup, best['focal_lengths'], best['distances'])
        target_w0 = best['target_w_flat']
        zr = (np.pi * target_w0**2) / self.setup.wavelength
        
        ideal_z = np.linspace(best_flat_z, best_curved_z, 100)
        ideal_w = target_w0 * np.sqrt(1 + ((ideal_z - best_flat_z) / zr)**2)
        
        self.ax_comp.plot(ideal_z, ideal_w, color='black', linestyle='--', linewidth=2.5, label='Nominal Cavity Mode')
        self.ax_comp.plot(ideal_z, -ideal_w, color='black', linestyle='--', linewidth=2.5)

        for idx, res in enumerate(top_results):
            if idx >= len(colors): break
            
            z, w, lenses_z, flat_z, curved_z, vp_s, vp_e = generate_beam_profile_data(self.setup, res['focal_lengths'], res['distances'])
            worst_err_um = max(res['worst_dev_flat'], res['worst_dev_curved']) * 1000
            
            w_err_ratio = worst_err_um / (res['target_w_flat']*1000)
            w_env_max = w * (1 + w_err_ratio)
            w_env_min = w * (1 - w_err_ratio)
            self.ax_comp.fill_between(z, w_env_max, w_env_min, color=colors[idx], alpha=0.15, linewidth=0)
            self.ax_comp.fill_between(z, -w_env_min, -w_env_max, color=colors[idx], alpha=0.15, linewidth=0)
            
            label_str = (f"Rank {idx+1}: {res['variable_lenses']} "
                         f"(Eff: {res['nom_eff']*100:.2f}% | "
                         f"Wrst: {res['worst_eff']*100:.2f}%)")
            self.ax_comp.plot(z, w, color=colors[idx], label=label_str, linewidth=1.5)
            self.ax_comp.plot(z, -w, color=colors[idx], linewidth=1.5)
            
            for l_z in lenses_z: self.ax_comp.axvline(x=l_z, color=colors[idx], linestyle=':', alpha=0.4)
            self.ax_comp.axvline(x=flat_z, color=colors[idx], linestyle=':', alpha=0.4)
            self.ax_comp.axvline(x=curved_z, color=colors[idx], linestyle=':', alpha=0.4)
            
            if vp_s is not None:
                self.ax_comp.axvspan(vp_s, vp_e, color=colors[idx], alpha=0.1)
            
        self.ax_comp.set_title("Top Combinations & Worst-Case Tolerance Boundaries")
        self.ax_comp.set_xlabel("Optical Axis Z (mm)"); self.ax_comp.set_ylabel("Beam Radius w(z) (mm)")
        self.ax_comp.legend(loc='lower right', fontsize='x-small')
        self.ax_comp.grid(True, linestyle='--', alpha=0.4)
        self.canvas_comp.draw()

    def refresh_distances(self):
        if hasattr(self, 'dist_frame') and self.dist_frame:
            for widget in self.dist_frame.winfo_children(): widget.destroy()
        self.dist_min_entries = []; self.dist_max_entries = []
        
        if not hasattr(self, 'lens_enabled'): return
        active_lenses = [i+1 for i in range(4) if self.lens_enabled[i].get()]
        num_distances = len(active_lenses) + 1
        
        for i in range(num_distances):
            if i == 0:
                target = f"L{active_lenses[0]}" if active_lenses else "Cavity"
                lbl_text = f"Dist {i+1} (Src -> {target}):"
            elif i == num_distances - 1:
                src = f"L{active_lenses[-1]}"
                lbl_text = f"Dist {i+1} ({src} -> Cavity):"
            else:
                src = f"L{active_lenses[i-1]}"
                target = f"L{active_lenses[i]}"
                lbl_text = f"Dist {i+1} ({src} -> {target}):"
                
            ttk.Label(self.dist_frame, text=lbl_text).grid(row=i//2, column=(i%2)*3, sticky='w', padx=(0, 5), pady=2)
            ent_min = ttk.Entry(self.dist_frame, width=6); ent_min.insert(0, "20")
            ent_max = ttk.Entry(self.dist_frame, width=6); ent_max.insert(0, "300")
            ent_min.grid(row=i//2, column=(i%2)*3 + 1, padx=2)
            ent_max.grid(row=i//2, column=(i%2)*3 + 2, padx=(2, 15))
            
            self.dist_min_entries.append(ent_min); self.dist_max_entries.append(ent_max)

    def update_entry(self, entry_widget, value):
        state = entry_widget.cget('state')
        if state == 'disabled': entry_widget.config(state='normal')
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, str(value))
        if state == 'disabled': entry_widget.config(state='disabled')

    def save_settings_to_xml(self):
        root_el = ET.Element("OpticsSetup")
        src = ET.SubElement(root_el, "Source", wl=self.ent_wl.get(), twl=self.ent_twl.get(), use_fiber=str(self.use_fiber_var.get()))
        ET.SubElement(src, "Manual", w0=self.ent_w0.get(), tw0=self.ent_tw0.get())
        ET.SubElement(src, "Fiber", core=self.ent_fcore.get(), tfcore=self.ent_tfcore.get(), coup=self.ent_fcoup.get(), tcoup=self.ent_tfcoup.get())
        
        vp = ET.SubElement(root_el, "Viewport", use=str(self.use_vp_var.get()), n=self.ent_n.get())
        ET.SubElement(vp, "Dims", t=self.ent_thick.get(), tt=self.ent_tthick.get(), a=self.ent_ang.get(), ta=self.ent_tang.get(), d=self.ent_vdist.get(), td=self.ent_tvdist.get())
        
        ET.SubElement(root_el, "Cavity", L=self.ent_cl.get(), tL=self.ent_tcl.get(), r1=self.ent_tr1.get(), r2=self.ent_roc.get(), tr2=self.ent_troc.get())
        
        lenses_el = ET.SubElement(root_el, "Lenses", tf=self.ent_tf.get())
        for i in range(4):
            ET.SubElement(lenses_el, "Lens", index=str(i), en=str(self.lens_enabled[i].get()), fix=str(self.lens_fixed[i].get()), f=self.lens_f_entries[i].get())
        dists_el = ET.SubElement(root_el, "Distances", tpos=self.ent_tpos.get())
        for i, (emin, emax) in enumerate(zip(self.dist_min_entries, self.dist_max_entries)):
            ET.SubElement(dists_el, "Dist", i=str(i), min=emin.get(), max=emax.get())
        ET.SubElement(root_el, "Opt", pool=self.ent_lenses.get(), time=self.ent_time.get())
        ET.ElementTree(root_el).write(self.settings_file)

    def load_settings_from_xml(self):
        if not os.path.exists(self.settings_file): return
        try:
            r = ET.parse(self.settings_file).getroot()
            if (s := r.find("Source")) is not None:
                self.update_entry(self.ent_wl, s.get("wl", "1550")); self.update_entry(self.ent_twl, s.get("twl", "0.1"))
                self.use_fiber_var.set(s.get("use_fiber") == "True")
                if (m := s.find("Manual")) is not None:
                    self.update_entry(self.ent_w0, m.get("w0", "0.5")); self.update_entry(self.ent_tw0, m.get("tw0", "0.01"))
                if (f := s.find("Fiber")) is not None:
                    self.update_entry(self.ent_fcore, f.get("core", "10.4")); self.update_entry(self.ent_tfcore, f.get("tfcore", "0.1"))
                    self.update_entry(self.ent_fcoup, f.get("coup", "11.0")); self.update_entry(self.ent_tfcoup, f.get("tcoup", "0.1"))
            
            if (vp := r.find("Viewport")) is not None:
                self.use_vp_var.set(vp.get("use") == "True"); self.update_entry(self.ent_n, vp.get("n", "1.5168"))
                if (dims := vp.find("Dims")) is not None:
                    self.update_entry(self.ent_thick, dims.get("t", "4.0")); self.update_entry(self.ent_tthick, dims.get("tt", "0.1"))
                    self.update_entry(self.ent_ang, dims.get("a", "3.0")); self.update_entry(self.ent_tang, dims.get("ta", "0.1"))
                    self.update_entry(self.ent_vdist, dims.get("d", "20.0")); self.update_entry(self.ent_tvdist, dims.get("td", "1.0"))
            
            if (c := r.find("Cavity")) is not None:
                self.update_entry(self.ent_cl, c.get("L", "50.0")); self.update_entry(self.ent_tcl, c.get("tL", "0.1"))
                self.update_entry(self.ent_tr1, c.get("r1", "10000"))
                self.update_entry(self.ent_roc, c.get("r2", "100.0")); self.update_entry(self.ent_troc, c.get("tr2", "1.0"))
            if (ls := r.find("Lenses")) is not None:
                self.update_entry(self.ent_tf, ls.get("tf", "1.0"))
                for l in ls.findall("Lens"):
                    i = int(l.get("index"))
                    self.lens_enabled[i].set(l.get("en", "False") == "True")
                    self.lens_fixed[i].set(l.get("fix", "False") == "True")
                    self.update_entry(self.lens_f_entries[i], l.get("f", "50.0"))
            self.refresh_distances()
            if (ds := r.find("Distances")) is not None:
                self.update_entry(self.ent_tpos, ds.get("tpos", "0.2"))
                for d in ds.findall("Dist"):
                    i = int(d.get("i"))
                    if i < len(self.dist_min_entries):
                        self.update_entry(self.dist_min_entries[i], d.get("min", "20"))
                        self.update_entry(self.dist_max_entries[i], d.get("max", "300"))
            if (o := r.find("Opt")) is not None:
                self.update_entry(self.ent_lenses, o.get("pool", "25, 50, 75, 100, 150, 200")); self.update_entry(self.ent_time, o.get("time", "15"))
        except Exception as e: print(e)
        self.toggle_viewport_mode()

    def on_closing(self): self.save_settings_to_xml(); self.root.destroy()

    def start_optimization(self):
        self.save_settings_to_xml(); threading.Thread(target=self.execute_optimization, daemon=True).start()

    def execute_optimization(self):
        try:
            self.setup.wavelength = float(self.ent_wl.get()) * 1e-6
            self.setup.tol_wl = float(self.ent_twl.get())
            self.setup.use_fiber_coupler = self.use_fiber_var.get()
            self.setup.source_w0 = float(self.ent_w0.get())
            self.setup.tol_w0 = float(self.ent_tw0.get())
            self.setup.fiber_core_mfd = float(self.ent_fcore.get()) * 1e-3  
            self.setup.tol_fcore = float(self.ent_tfcore.get()) * 1e-3  
            self.setup.coupler_f = float(self.ent_fcoup.get())
            self.setup.tol_coupler_f = float(self.ent_tfcoup.get())
            
            self.setup.use_viewport = self.use_vp_var.get()
            self.setup.window_thickness = float(self.ent_thick.get())
            self.setup.tol_thickness = float(self.ent_tthick.get())
            self.setup.window_angle = float(self.ent_ang.get())
            self.setup.tol_angle = float(self.ent_tang.get())
            self.setup.window_n = float(self.ent_n.get())
            self.setup.window_dist = float(self.ent_vdist.get())
            self.setup.tol_window_dist = float(self.ent_tvdist.get())
            
            self.setup.cavity_length = float(self.ent_cl.get())
            self.setup.tol_cl = float(self.ent_tcl.get())
            self.setup.tol_r1 = float(self.ent_tr1.get())
            self.setup.cavity_R2 = float(self.ent_roc.get())
            self.setup.tol_r2 = float(self.ent_troc.get())
            
            self.setup.tol_f = float(self.ent_tf.get())
            self.setup.tol_pos = float(self.ent_tpos.get())
            
            active_lenses = []
            for i in range(4):
                if self.lens_enabled[i].get():
                    if self.lens_fixed[i].get(): active_lenses.append(float(self.lens_f_entries[i].get()))
                    else: active_lenses.append(None)
            self.setup.lenses = active_lenses
            
            bounds = []
            for d_min, d_max in zip(self.dist_min_entries, self.dist_max_entries):
                bounds.append((float(d_min.get()), float(d_max.get())))
            self.setup.distance_bounds = bounds
            self.setup.available_lenses = [float(x.strip()) for x in self.ent_lenses.get().split(",") if x.strip()]
            
            self.txt_output.delete("1.0", tk.END)
            self.txt_output.insert(tk.END, "Running multithreaded Grid Search & Monte Carlo Edge-Case simulations...\n")
            self.btn_run.config(state='disabled')
            
            results = run_multithreaded_optimization(self.setup, float(self.ent_time.get()), self.update_progress)
            self.btn_run.config(state='normal')
            
            if results:
                best = results[0]
                self.txt_output.insert(tk.END, "\n" + "="*50 + " GRID SEARCH RESULTS " + "="*50 + "\n")
                header = "{:<4} | {:<18} | {:<35} | {:<12} | {:<12} | {:<16} | {:<16}\n".format(
                    "Rank", "Var. Lenses (f)", "Distances (mm)", "Nom. Eff(%)", "Wrst Eff(%)", "Max Dev F (µm)", "Max Dev C (µm)"
                )
                self.txt_output.insert(tk.END, header)
                self.txt_output.insert(tk.END, "-"*132 + "\n")
                
                for idx, res in enumerate(results):
                    var_lens_str = str(res['variable_lenses'])
                    dist_str = ", ".join([f"{d:.1f}" for d in res['distances']])
                    nom_eff = res['nom_eff'] * 100
                    wrst_eff = res['worst_eff'] * 100
                    wrst_f = res['worst_dev_flat'] * 1000
                    wrst_c = res['worst_dev_curved'] * 1000
                    
                    row_str = "{:<4} | {:<18} | {:<35} | {:<12.2f} | {:<12.2f} | ±{:<15.3f} | ±{:<15.3f}\n".format(
                        idx + 1, var_lens_str, f"[{dist_str}]", nom_eff, wrst_eff, wrst_f, wrst_c
                    )
                    self.txt_output.insert(tk.END, row_str)

                self.txt_output.insert(tk.END, "\n>> BEST RESULT SUMMARY (Rank 1) <<\n")
                self.txt_output.insert(tk.END, f"Target Flat Waist: {best['target_w_flat']:.5f} mm | Achieved: {best['achieved_w_flat']:.5f} mm | Worst Defect Limits: ± {best['worst_dev_flat']*1000:.1f} µm\n")
                self.txt_output.insert(tk.END, f"Target Curved Waist: {best['target_w_curved']:.5f} mm | Achieved: {best['achieved_w_curved']:.5f} mm | Worst Defect Limits: ± {best['worst_dev_curved']*1000:.1f} µm\n")
                
                self.draw_sketch(best)
                self.draw_comparison(results)
            else:
                self.txt_output.insert(tk.END, "\nFailed to find any viable physical solution in time bounds.\n")
                
        except Exception as err:
            self.btn_run.config(state='normal')
            import traceback
            traceback_str = traceback.format_exc()
            self.txt_output.insert(tk.END, f"\nExecution crashed:\n{traceback_str}")
            messagebox.showerror("Execution Error", str(err))

    def update_progress(self, val):
        self.progress_bar['value'] = val
        self.root.update_idletasks()

if __name__ == "__main__":
    app = ModeMatchingGUI(tk.Tk())
    app.root.mainloop()