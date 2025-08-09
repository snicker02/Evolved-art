import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw
import math
import traceback
import os
import random
import numpy as np
from datetime import datetime

try:
    from effects import AVAILABLE_EFFECTS
    from generators import AVAILABLE_GENERATORS, parse_color_string
    from gradient_loader import load_gradient_from_file
except ImportError as e:
    messagebox.showerror("Missing File", f"A required file is missing: {e}. Please ensure effects.py, generators.py, and gradient_loader.py are in the same directory.")
    exit()

try:
    import torch
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: PyTorch not found. GPU acceleration will be disabled. Only basic CPU functions will work.")

if GPU_AVAILABLE:
    def pil_to_tensor(pil_image):
        if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
        numpy_array = np.array(pil_image, dtype=np.float32) / 255.0
        return torch.from_numpy(numpy_array).permute(2, 0, 1).unsqueeze(0)

    def tensor_to_pil(tensor):
        tensor = tensor.clamp(0, 1)
        numpy_array = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(numpy_array, 'RGB')

    class GPUManager:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"GPU Manager initialized on device: {self.device}")

class EvolverApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title(f"Modular Image Evolver v5.2 {'(GPU Enabled)' if GPU_AVAILABLE else '(CPU Mode)'}")
        self.gpu_evolver = GPUManager() if GPU_AVAILABLE else None

        # --- Instance Variables ---
        self.input_image_loaded = None; self.current_evolving_image = None; self.photo_image = None
        self.input_image_filename_var = tk.StringVar(value="No image loaded.")
        self.displacement_map_image = None; self.displacement_map_filename_var = tk.StringVar(value="No map loaded.")
        self.target_display_width = 500; self.target_display_height = 500
        self.after_id_preview = None; self.interactive_update_delay = 75
        self.hold_evolve_active = False; self.hold_evolve_after_id = None
        self.hold_evolve_delay = 50
        self.stop_evolution_requested = False
        self.save_animation_frames_var = tk.BooleanVar(value=False)

        self.zoom_factor = tk.DoubleVar(value=1.0)
        self.view_center_x_norm = tk.DoubleVar(value=0.5); self.view_center_y_norm = tk.DoubleVar(value=0.5)
        self.panning_active = False
        self.pan_start_mouse_x = 0; self.pan_start_mouse_y = 0
        self.pan_start_view_cx = 0.5; self.pan_start_view_cy = 0.5
        self.panning_mode_var = tk.StringVar(value="Tile")
        self.panning_options = ["Off", "Tile", "Mirror"]

        self.symmetry_options = ["None", "Horizontal (Left Master)", "Horizontal (Right Master)", "Vertical (Top Master)", "Vertical (Bottom Master)", "4-Way Mirror (Top-Left Master)", "2-Fold Rotational (Average)", "Kaleidoscope (6-fold)", "Kaleidoscope (8-fold)", "Diagonal Mirror (TL-BR Master)", "Diagonal Mirror (TR-BL Master)"]

        self.entries = {}; self.op_params = {}; self.anim_params = {}
        self.param_frames = {}; self.active_pipeline = []

        self.generator_params = {}; self.generator_param_frames = {}; self.selected_generator_key = tk.StringVar()
        self.generator_swatch_frames = {}

        self.enable_brightness_boost_var = tk.BooleanVar(value=False)
        self.brightness_boost_nth_frame_var = tk.IntVar(value=50)
        self.brightness_boost_amount_var = tk.DoubleVar(value=1.02)

        self.anim_enabled = {'pan_x': tk.BooleanVar(value=False), 'pan_y': tk.BooleanVar(value=False)}
        self.anim_params['pan_x_amp'] = tk.DoubleVar(value=0.1); self.anim_params['pan_x_per'] = tk.IntVar(value=100)
        self.anim_params['pan_y_amp'] = tk.DoubleVar(value=0.1); self.anim_params['pan_y_per'] = tk.IntVar(value=120)

        self.anim_modes = {}
        self.drunken_walk_values = {}
        self.anim_base_values = {}

        self.enable_periodic_random_var = tk.BooleanVar(value=False)
        self.periodic_random_nth_frame_var = tk.IntVar(value=100)
        self.periodic_random_strength_var = tk.DoubleVar(value=0.1)

        self.effects_config = {
            "blur": {"params": {"radius": {"var_key": "blur_radius", "default": 0.0, "min": 0.0, "max": 10.0, "label": "Radius:", "animatable": True, "amp_default": 1.0, "period_default": 100, "hold_default": 1}}},
            "shape": {"params": {
                "shape_type": {"var_key": "shape_type", "default": "Circle", "type": "dropdown", "options": ["Circle", "Rectangle", "Diamond", "Triangle"], "label": "Type:"},
                "shape_x": {"var_key": "shape_x", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Center X:", "animatable": True, "amp_default": 0.1, "period_default": 120, "hold_default": 1},
                "shape_y": {"var_key": "shape_y", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Center Y:", "animatable": True, "amp_default": 0.1, "period_default": 140, "hold_default": 1},
                "shape_size": {"var_key": "shape_size", "default": 0.2, "min": 0.01, "max": 1.0, "label": "Size:", "animatable": True, "amp_default": 0.05, "period_default": 100, "hold_default": 1},
                "shape_color": {"var_key": "shape_color", "default": "#ffffff", "type": "color", "label": "Color:"},
                "shape_outline": {"var_key": "shape_outline", "default": True, "type": "checkbox", "label": "Outline"},
                "shape_blend": {"var_key": "shape_blend", "default": 0.5, "min": 0.0, "max": 1.0, "label": "Blend:"},
                "inner_effect": {"var_key": "shape_inner_effect", "default": "None", "type": "dropdown", "options": [], "label": "Inner Effect:"},
                "inner_strength": {"var_key": "shape_inner_strength", "default": 1.0, "min": 0.0, "max": 3.0, "label": "Inner Strength:"},
            }},
            "unsharp_mask": {"params": {"radius": {"var_key": "unsharp_radius", "default": 2, "min": 0, "max": 10, "label": "Rad:", "is_int": True}, "percent": {"var_key": "unsharp_percent", "default": 150, "min": 50, "max": 300, "label": "%:", "is_int": True}}},
            "sharpen": {"params": {"strength": {"var_key": "sharpen_strength", "default": 0.0, "min": 0.0, "max": 1.0, "label": "Strength:", "animatable": True, "amp_default": 0.5, "period_default": 80, "hold_default": 1}}},
            "pixelate": {"params": {"block_size": {"var_key": "pixelate_block_size", "default": 8, "min": 2, "max": 64, "label": "Block:", "is_int": True}}},
            "channel_shift": {"params": {"r_x": {"var_key": "chanshift_rx", "default": 0, "min": -20, "max": 20, "label": "R X:", "is_int": True, "animatable": True, "amp_default": 5, "period_default": 120, "hold_default": 1}, "r_y": {"var_key": "chanshift_ry", "default": 0, "min": -20, "max": 20, "label": "R Y:", "is_int": True, "animatable": True, "amp_default": 5, "period_default": 140, "hold_default": 1}, "b_x": {"var_key": "chanshift_bx", "default": 0, "min": -20, "max": 20, "label": "B X:", "is_int": True, "animatable": True, "amp_default": -5, "period_default": 130, "hold_default": 1}, "b_y": {"var_key": "chanshift_by", "default": 0, "min": -20, "max": 20, "label": "B Y:", "is_int": True, "animatable": True, "amp_default": -5, "period_default": 150, "hold_default": 1}}},
            "displacement_map": {"params": {"x_scale": {"var_key": "displace_x_scale", "default": 10.0, "min": -100.0, "max": 100.0, "label": "X Scale:", "animatable": True, "amp_default": 20, "period_default": 200, "hold_default": 1}, "y_scale": {"var_key": "displace_y_scale", "default": 10.0, "min": -100.0, "max": 100.0, "label": "Y Scale:", "animatable": True, "amp_default": 20, "period_default": 220, "hold_default": 1}}},
            "wave_displace": {"params": {
                "wave_type_x": {"var_key": "wave_type_x", "default": "Sine", "type": "dropdown", "options": ["Sine", "Triangle", "Sawtooth"], "label": "Wave X:"},
                "amp_x": {"var_key": "wave_amp_x", "default": 0, "min": 0, "max": 100, "label": "Amp X:", "is_int": True, "animatable": True, "amp_default": 20, "period_default": 100, "hold_default": 1},
                "freq_x": {"var_key": "wave_freq_x", "default": 4, "min": 0, "max": 50, "label": "Freq X:", "animatable": True, "amp_default": 10, "period_default": 120, "hold_default": 1},
                "wave_type_y": {"var_key": "wave_type_y", "default": "Sine", "type": "dropdown", "options": ["Sine", "Triangle", "Sawtooth"], "label": "Wave Y:"},
                "amp_y": {"var_key": "wave_amp_y", "default": 0, "min": 0, "max": 100, "label": "Amp Y:", "is_int": True, "animatable": True, "amp_default": 20, "period_default": 110, "hold_default": 1},
                "freq_y": {"var_key": "wave_freq_y", "default": 4, "min": 0, "max": 50, "label": "Freq Y:", "animatable": True, "amp_default": 10, "period_default": 130, "hold_default": 1},
            }},
            "pixel_jitter": {"params": {"strength": {"var_key": "jitter_strength", "default": 0.0, "min": 0.0, "max": 50.0, "label": "Strength:", "animatable": True, "amp_default": 10, "period_default": 60, "hold_default": 1}}},
            "swirl": {"params": {"strength": {"var_key": "swirl_strength", "default": 0.0, "min": -10.0, "max": 10.0, "label": "Strength:", "animatable": True, "amp_default": 2.0, "period_default": 150, "hold_default": 1}}},
            "chromatic_aberration": {"params": {
                "red_offset": {"var_key": "ca_red_offset", "default": 0.0, "min": -20.0, "max": 20.0, "label": "Red Offset:", "animatable": True, "amp_default": 5, "period_default": 120, "hold_default": 1},
                "red_angle": {"var_key": "ca_red_angle", "default": 0.0, "min": 0, "max": 360, "label": "Red Angle:", "animatable": True, "amp_default": 180, "period_default": 200, "hold_default": 1},
                "blue_offset": {"var_key": "ca_blue_offset", "default": 0.0, "min": -20.0, "max": 20.0, "label": "Blue Offset:", "animatable": True, "amp_default": -5, "period_default": 120, "hold_default": 1},
                "blue_angle": {"var_key": "ca_blue_angle", "default": 180.0, "min": 0, "max": 360, "label": "Blue Angle:", "animatable": True, "amp_default": 180, "period_default": 200, "hold_default": 1},
            }},
            "spherical_distort": {"params": {
                "strength": {"var_key": "sphere_strength", "default": 1.0, "min": 0.1, "max": 4.0, "label": "Strength:", "animatable": True, "amp_default": 0.5, "period_default": 100, "hold_default": 1},
                "zoom": {"var_key": "sphere_zoom", "default": 1.0, "min": 0.1, "max": 5.0, "label": "Zoom:", "animatable": True, "amp_default": 0.5, "period_default": 120, "hold_default": 1},
            }},
            "perspective_transform": {"params": {
                "tl_x": {"var_key": "pt_tl_x", "default": 0.0, "min": -1.0, "max": 1.0, "label": "TopLeft X:"},
                "tl_y": {"var_key": "pt_tl_y", "default": 0.0, "min": -1.0, "max": 1.0, "label": "TopLeft Y:"},
                "tr_x": {"var_key": "pt_tr_x", "default": 1.0, "min": 0.0, "max": 2.0, "label": "TopRight X:"},
                "tr_y": {"var_key": "pt_tr_y", "default": 0.0, "min": -1.0, "max": 1.0, "label": "TopRight Y:"},
                "bl_x": {"var_key": "pt_bl_x", "default": 0.0, "min": -1.0, "max": 1.0, "label": "BotLeft X:"},
                "bl_y": {"var_key": "pt_bl_y", "default": 1.0, "min": 0.0, "max": 2.0, "label": "BotLeft Y:"},
                "br_x": {"var_key": "pt_br_x", "default": 1.0, "min": 0.0, "max": 2.0, "label": "BotRight X:"},
                "br_y": {"var_key": "pt_br_y", "default": 1.0, "min": 0.0, "max": 2.0, "label": "BotRight Y:"},
                "tiled": {"var_key": "pt_tiled", "default": True, "type": "checkbox", "label": "Tiled"},
            }},
            "turbulence": {"params": {
                "strength": {"var_key": "turbulence_strength", "default": 0.0, "min": 0.0, "max": 100.0, "label": "Strength:", "animatable": True, "amp_default": 20, "period_default": 150, "hold_default": 1},
                "scale": {"var_key": "turbulence_scale", "default": 50.0, "min": 1.0, "max": 300.0, "label": "Scale:", "animatable": True, "amp_default": 50, "period_default": 200, "hold_default": 1},
                "octaves": {"var_key": "turbulence_octaves", "default": 4, "min": 1, "max": 8, "label": "Octaves:", "is_int": True},
            }},
            "voronoi_noise": {"params": {
                "points": {"var_key": "voronoi_points", "default": 50, "min": 2, "max": 500, "label": "# Points:", "is_int": True},
                "strength": {"var_key": "voronoi_strength", "default": 0.0, "min": 0.0, "max": 100.0, "label": "Strength:", "animatable": True, "amp_default": 20, "period_default": 150, "hold_default": 1},
            }},
            "julia_distort": {"params": {
                "c_real": {"var_key": "julia_c_real", "default": -0.8, "min": -2.0, "max": 2.0, "label": "C Real:", "animatable": True, "amp_default": 0.1, "period_default": 120, "hold_default": 1},
                "c_imag": {"var_key": "julia_c_imag", "default": 0.156, "min": -2.0, "max": 2.0, "label": "C Imaginary:", "animatable": True, "amp_default": 0.1, "period_default": 140, "hold_default": 1},
                "iterations": {"var_key": "julia_iterations", "default": 20, "min": 1, "max": 100, "label": "Iterations:", "is_int": True},
                "strength": {"var_key": "julia_strength", "default": 10.0, "min": 0.0, "max": 100.0, "label": "Strength:", "animatable": True, "amp_default": 20, "period_default": 100, "hold_default": 1},
            }},
            "mandelbrot_distort": {"params": {
                "iterations": {"var_key": "mandel_iterations", "default": 20, "min": 1, "max": 100, "label": "Iterations:", "is_int": True},
                "strength": {"var_key": "mandel_strength", "default": 10.0, "min": 0.0, "max": 100.0, "label": "Strength:", "animatable": True, "amp_default": 20, "period_default": 100, "hold_default": 1},
            }},
            "burning_ship_distort": {"params": {
                "iterations": {"var_key": "ship_iterations", "default": 20, "min": 1, "max": 100, "label": "Iterations:", "is_int": True},
                "strength": {"var_key": "ship_strength", "default": 10.0, "min": 0.0, "max": 100.0, "label": "Strength:", "animatable": True, "amp_default": 20, "period_default": 100, "hold_default": 1},
            }},
            "reaction_diffusion": {"params": {
                "Du": {"var_key": "rd_Du", "default": 1.0, "min": 0.0, "max": 2.0, "label": "Diffusion U:"},
                "Dv": {"var_key": "rd_Dv", "default": 0.5, "min": 0.0, "max": 2.0, "label": "Diffusion V:"},
                "feed": {"var_key": "rd_feed", "default": 0.055, "min": 0.0, "max": 0.1, "label": "Feed Rate:"},
                "kill": {"var_key": "rd_kill", "default": 0.062, "min": 0.0, "max": 0.1, "label": "Kill Rate:"},
                "strength": {"var_key": "rd_strength", "default": 10.0, "min": -50.0, "max": 50.0, "label": "Strength:"},
            }},
            "color": {"params": {
                "brightness": {"var_key": "brightness_factor", "default": 1.0, "min": 0.5, "max": 1.5, "label": "Bright:", "animatable": True, "amp_default": 0.1, "period_default": 80, "hold_default": 1},
                "contrast": {"var_key": "contrast_factor", "default": 1.0, "min": 0.5, "max": 1.5, "label": "Contrast:", "animatable": True, "amp_default": 0.1, "period_default": 90, "hold_default": 1},
                "saturation": {"var_key": "saturation_factor", "default": 1.0, "min": 0.0, "max": 3.0, "label": "Satur:", "animatable": True, "amp_default": 0.2, "period_default": 100, "hold_default": 1},
                "hue": {"var_key": "hue_amount", "default": 0.0, "min": -0.1, "max": 0.1, "label": "Hue Shift:", "animatable": True, "amp_default": 0.05, "period_default": 150, "hold_default": 1}
            }},
            "simple_tile": {"params": {"tile_scale_factor": {"var_key": "tile_scale_factor", "default": 0.5, "min": 0.1, "max": 2.0, "label": "Tile Scale:", "animatable": True, "amp_default": 0.2, "period_default": 100, "hold_default": 1}}},
            "shear": {"params": {"shear_x_factor": {"var_key": "shear_x_factor", "default": 0.0, "min": -1.0, "max": 1.0, "label": "X Factor:", "animatable": True, "amp_default": 0.2, "period_default": 120, "hold_default": 1}}},
            "rotate": {"params": {"rotate_angle_value": {"var_key": "rotate_angle_value", "default": 0.0, "min": -45.0, "max": 45.0, "label": "Angle(°):", "animatable": True, "amp_default": 10, "period_default": 180, "hold_default": 1}}},
            "symmetry": {"params": {"symmetry_type": {"var_key": "symmetry_type", "default": "None", "type": "dropdown", "options": self.symmetry_options, "label": "Type:"}}},
            "random_block": {"params": {
                "num_to_pick": {"var_key": "rand_num_to_pick", "default": 1, "min": 1, "max": 5, "label": "# Effects:", "is_int": True},
                "randomize_params": {"var_key": "rand_randomize_params", "default": True, "type": "checkbox", "label": "Randomize Parameters"},
            }},
        }
        
        inner_effect_options = ["None"] + sorted([e.name for e in AVAILABLE_EFFECTS.values() if e.name not in ["Shape", "Symmetry", "Random Effects Block"]])
        self.effects_config["shape"]["params"]["inner_effect"]["options"] = inner_effect_options

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        controls_area_frame = ttk.Frame(main_frame, padding=5)
        controls_area_frame.grid(row=0, column=0, sticky="nswe")
        main_frame.columnconfigure(0, weight=0, minsize=500)
        main_frame.rowconfigure(0, weight=1)
        self._setup_gui(controls_area_frame)
        self.image_display_label = ttk.Label(main_frame, relief="sunken", anchor="center", background="#2B2B2B")
        self.image_display_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        self.image_display_label.bind("<ButtonPress-1>", self.on_pan_start)
        self.image_display_label.bind("<B1-Motion>", self.on_pan_drag)
        self.image_display_label.bind("<ButtonRelease-1>", self.on_pan_end)
        self.image_display_label.bind("<Button-3>", self.on_shape_place)

        self.root.after(200, self._capture_initial_display_size)

    # FIXED: The methods below were incorrectly indented inside __init__
    # They are now properly defined as methods of the EvolverApp class.

    def on_shape_place(self, event):
        if "shape" not in self.active_pipeline:
            return
        
        if not self.photo_image: return
        display_w = self.photo_image.width()
        display_h = self.photo_image.height()

        label_w = self.image_display_label.winfo_width()
        label_h = self.image_display_label.winfo_height()
        
        offset_x = (label_w - display_w) / 2
        offset_y = (label_h - display_h) / 2

        norm_x = (event.x - offset_x) / display_w
        norm_y = (event.y - offset_y) / display_h

        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        if "shape_x" in self.op_params and "shape_y" in self.op_params:
            self.op_params["shape_x"].set(norm_x)
            self.op_params["shape_y"].set(norm_y)
            self.schedule_interactive_update()
            
    def _setup_gui(self, parent_frame):
        canvas = tk.Canvas(parent_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="10")
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"), width=e.width))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        csr = 0

        input_frame = ttk.LabelFrame(scrollable_frame, text="Image Input & Generation", padding="10")
        input_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        file_frame = ttk.Frame(input_frame); file_frame.pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Image File", command=self.load_input_image).pack(side=tk.LEFT, padx=5)
        ttk.Label(file_frame, textvariable=self.input_image_filename_var).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Load Gradient (.map, .ugr)", command=self.load_gradient_file).pack(side=tk.LEFT, padx=10)
        gen_frame = ttk.Frame(input_frame); gen_frame.pack(fill=tk.X, pady=5)
        self.generator_listbox = tk.Listbox(gen_frame, exportselection=False, height=len(AVAILABLE_GENERATORS))
        for key in AVAILABLE_GENERATORS: self.generator_listbox.insert(tk.END, AVAILABLE_GENERATORS[key].name)
        self.generator_listbox.pack(side=tk.LEFT, padx=5)
        self.generator_listbox.bind("<<ListboxSelect>>", self.on_generator_selection_change)
        self.generator_params_container = ttk.Frame(gen_frame)
        self.generator_params_container.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self._create_all_generator_param_widgets()
        ttk.Button(input_frame, text="Generate New Image", command=self.generate_from_selected_generator).pack(fill=tk.X, padx=5, pady=5)

        config_frame = ttk.LabelFrame(scrollable_frame, text="Configuration", padding="10")
        config_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        ttk.Label(config_frame, text="Steps:").grid(row=0, column=0); self.entries["steps"] = tk.StringVar(value="200")
        ttk.Entry(config_frame, textvariable=self.entries["steps"], width=7).grid(row=0, column=1)
        ttk.Label(config_frame, text="Width:").grid(row=0, column=2); self.entries["output_width"] = tk.StringVar(value="512")
        ttk.Entry(config_frame, textvariable=self.entries["output_width"], width=7).grid(row=0, column=3)
        ttk.Label(config_frame, text="Height:").grid(row=0, column=4); self.entries["output_height"] = tk.StringVar(value="512")
        ttk.Entry(config_frame, textvariable=self.entries["output_height"], width=7).grid(row=0, column=5)

        view_frame = ttk.LabelFrame(scrollable_frame, text="View Controls", padding=10)
        view_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        zoom_btn_frame = ttk.Frame(view_frame); zoom_btn_frame.pack(fill=tk.X)
        ttk.Button(zoom_btn_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_btn_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_btn_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        panning_mode_frame = ttk.Frame(zoom_btn_frame); panning_mode_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(panning_mode_frame, text="Panning:").pack(side=tk.LEFT)
        ttk.Combobox(panning_mode_frame, textvariable=self.panning_mode_var, values=self.panning_options, state="readonly", width=8).pack(side=tk.LEFT)
        pan_anim_x_frame = ttk.Frame(view_frame); pan_anim_x_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Checkbutton(pan_anim_x_frame, text="Animate Pan X", variable=self.anim_enabled['pan_x']).pack(side=tk.LEFT)
        ttk.Label(pan_anim_x_frame, text="Amp:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Entry(pan_anim_x_frame, textvariable=self.anim_params['pan_x_amp'], width=5).pack(side=tk.LEFT)
        ttk.Label(pan_anim_x_frame, text="Per:").pack(side=tk.LEFT, padx=(5,0))
        ttk.Entry(pan_anim_x_frame, textvariable=self.anim_params['pan_x_per'], width=5).pack(side=tk.LEFT)
        pan_anim_y_frame = ttk.Frame(view_frame); pan_anim_y_frame.pack(fill=tk.X)
        ttk.Checkbutton(pan_anim_y_frame, text="Animate Pan Y", variable=self.anim_enabled['pan_y']).pack(side=tk.LEFT)
        ttk.Label(pan_anim_y_frame, text="Amp:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Entry(pan_anim_y_frame, textvariable=self.anim_params['pan_y_amp'], width=5).pack(side=tk.LEFT)
        ttk.Label(pan_anim_y_frame, text="Per:").pack(side=tk.LEFT, padx=(5,0))
        ttk.Entry(pan_anim_y_frame, textvariable=self.anim_params['pan_y_per'], width=5).pack(side=tk.LEFT)

        pipeline_frame = ttk.LabelFrame(scrollable_frame, text="Effects Pipeline", padding=10)
        pipeline_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        available_frame = ttk.Frame(pipeline_frame); available_frame.grid(row=0, column=0, padx=5)
        ttk.Label(available_frame, text="Available Effects").pack()
        self.available_listbox = tk.Listbox(available_frame, exportselection=False, height=10)
        
        for key in sorted(AVAILABLE_EFFECTS.keys()): self.available_listbox.insert(tk.END, AVAILABLE_EFFECTS[key].name)
        
        self.available_listbox.pack()
        add_remove_frame = ttk.Frame(pipeline_frame); add_remove_frame.grid(row=0, column=1, padx=5, sticky="ns")
        ttk.Button(add_remove_frame, text="Add >>", command=self.add_effect_to_pipeline).pack(pady=10)
        ttk.Button(add_remove_frame, text="<< Remove", command=self.remove_effect_from_pipeline).pack()
        active_frame = ttk.Frame(pipeline_frame); active_frame.grid(row=0, column=2, padx=5)
        ttk.Label(active_frame, text="Active Pipeline").pack()
        self.active_listbox = tk.Listbox(active_frame, exportselection=False, height=10)
        self.active_listbox.bind("<<ListboxSelect>>", self.on_pipeline_selection_change)
        self.active_listbox.pack()
        move_frame = ttk.Frame(pipeline_frame); move_frame.grid(row=0, column=3, padx=5, sticky="ns")
        ttk.Button(move_frame, text="Move Up", command=self.move_effect_up).pack(pady=10)
        ttk.Button(move_frame, text="Move Down", command=self.move_effect_down).pack()

        self.params_container = ttk.LabelFrame(scrollable_frame, text="Effect Parameters", padding=10)
        self.params_container.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        self._create_all_param_widgets()

        final_ops_frame = ttk.LabelFrame(scrollable_frame, text="Final Operations", padding=10)
        final_ops_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        self.op_params["feedback_amount"] = tk.DoubleVar(value=1.0)
        ttk.Label(final_ops_frame, text="Feedback:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(final_ops_frame, variable=self.op_params["feedback_amount"], from_=0.8, to=1.0, length=120, command=self.schedule_interactive_update).grid(row=0, column=1)
        ttk.Label(final_ops_frame, textvariable=self.op_params["feedback_amount"], width=5).grid(row=0, column=2)
        boost_frame = ttk.LabelFrame(scrollable_frame, text="Periodic Boosts", padding=10)
        boost_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        ttk.Checkbutton(boost_frame, text="Enable Brightness Boost", variable=self.enable_brightness_boost_var).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(boost_frame, text="Every Nth Frame:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(boost_frame, textvariable=self.brightness_boost_nth_frame_var, width=7).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(boost_frame, text="Boost Amount:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Scale(boost_frame, variable=self.brightness_boost_amount_var, from_=1.0, to=1.1, length=120).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(boost_frame, textvariable=self.brightness_boost_amount_var, width=5).grid(row=2, column=2, sticky=tk.W)
        ttk.Separator(boost_frame, orient='horizontal').grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)
        ttk.Checkbutton(boost_frame, text="Enable Periodic Randomization", variable=self.enable_periodic_random_var).grid(row=4, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(boost_frame, text="Every Nth Frame:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(boost_frame, textvariable=self.periodic_random_nth_frame_var, width=7).grid(row=5, column=1, sticky=tk.W)
        ttk.Label(boost_frame, text="Randomization Strength:").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Scale(boost_frame, variable=self.periodic_random_strength_var, from_=0.0, to=1.0, length=120).grid(row=6, column=1, sticky=tk.W)
        ttk.Label(boost_frame, textvariable=self.periodic_random_strength_var, width=5).grid(row=6, column=2, sticky=tk.W)

        output_frame = ttk.LabelFrame(scrollable_frame, text="Output", padding=10)
        output_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        ttk.Checkbutton(output_frame, text="Save Animation Frames (on Multi-Step)", variable=self.save_animation_frames_var).pack(side=tk.LEFT)

        buttons_frame = ttk.Frame(scrollable_frame); buttons_frame.grid(row=csr, column=0, columnspan=4, pady=15, sticky="ew"); csr+=1
        ttk.Button(buttons_frame, text="Preview Step", command=self.schedule_interactive_update).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(buttons_frame, text="Multi-Step Evolve", command=self.trigger_multistep_evolution).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.request_stop_evolution, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        hold_button = ttk.Button(buttons_frame, text="Hold to Evolve")
        hold_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        hold_button.bind("<ButtonPress-1>", self.on_hold_evolve_press); hold_button.bind("<ButtonRelease-1>", self.on_hold_evolve_release)

        file_buttons_frame = ttk.Frame(scrollable_frame); file_buttons_frame.grid(row=csr, column=0, columnspan=4, pady=5, sticky="ew"); csr+=1
        ttk.Button(file_buttons_frame, text="Reset Image", command=self.reset_image_to_original).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(file_buttons_frame, text="Save Image", command=self.save_image_as).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.status_label = ttk.Label(scrollable_frame, text="Load an image or generate one to begin.", wraplength=450)
        # FIXED: Removed duplicate grid placement of the status label
        self.status_label.grid(row=csr, column=0, columnspan=4, pady=5, sticky=tk.W)

    def _create_all_param_widgets(self, *args):
        for key, config in self.effects_config.items():
            frame = ttk.Frame(self.params_container)
            self.param_frames[key] = frame
            if key == "displacement_map":
                ttk.Button(frame, text="Load D-Map", command=self.load_displacement_map_image).pack(side=tk.LEFT, padx=5)
                ttk.Label(frame, textvariable=self.displacement_map_filename_var).pack(side=tk.LEFT)
            for param_name, p_config in config.get("params", {}).items():
                p_frame = ttk.Frame(frame); p_frame.pack(fill=tk.X, expand=True, pady=1)
                var_key = p_config["var_key"]

                if p_config.get("type") == "dropdown":
                    self.op_params[var_key] = tk.StringVar(value=p_config["default"])
                    ttk.Label(p_frame, text=p_config.get("label", param_name.title() + ":"), width=12).pack(side=tk.LEFT)
                    ttk.Combobox(p_frame, textvariable=self.op_params[var_key], values=p_config["options"], state="readonly").pack(side=tk.LEFT, expand=True, fill=tk.X)
                elif p_config.get("type") == "checkbox":
                    self.op_params[var_key] = tk.BooleanVar(value=p_config["default"])
                    ttk.Checkbutton(p_frame, text=p_config["label"], variable=self.op_params[var_key]).pack(side=tk.LEFT, padx=5)
                elif p_config.get("type") == "color":
                    self.op_params[var_key] = tk.StringVar(value=p_config["default"])
                    ttk.Label(p_frame, text=p_config.get("label", "Color:"), width=12).pack(side=tk.LEFT)
                    
                    color_btn = tk.Button(p_frame, text="", bg=p_config["default"], width=3, relief="sunken")
                    color_btn.pack(side=tk.LEFT)
                    
                    def choose_color(v=self.op_params[var_key], btn=color_btn):
                        color_code = colorchooser.askcolor(title="Choose color", initialcolor=v.get())
                        if color_code and color_code[1]:
                            v.set(color_code[1])
                            btn.config(bg=color_code[1])
                            self.schedule_interactive_update()
                    color_btn.config(command=choose_color)
                else:
                    var_type = tk.IntVar if p_config.get("is_int") else tk.DoubleVar
                    self.op_params[var_key] = var_type(value=p_config["default"])
                    ttk.Label(p_frame, text=p_config["label"], width=10).pack(side=tk.LEFT)
                    ttk.Scale(p_frame, variable=self.op_params[var_key], from_=p_config["min"], to=p_config["max"], command=self.schedule_interactive_update).pack(side=tk.LEFT, expand=True, fill=tk.X)
                    entry = ttk.Entry(p_frame, textvariable=self.op_params[var_key], width=7)
                    entry.pack(side=tk.LEFT, padx=(2, 10))
                    entry.bind("<Return>", self.schedule_interactive_update)
                    entry.bind("<FocusOut>", self.schedule_interactive_update)
                    if p_config.get("animatable"):
                        anim_mode_key = f"{var_key}_anim_mode"
                        self.anim_modes[anim_mode_key] = tk.StringVar(value="Sine Wave")

                        amp_var_key = f"{var_key}_amp"; per_var_key = f"{var_key}_per"; hold_var_key = f"{var_key}_hold"
                        self.anim_params[amp_var_key] = tk.DoubleVar(value=p_config.get("amp_default", 0.0))
                        self.anim_params[per_var_key] = tk.IntVar(value=p_config.get("period_default", 100))
                        self.anim_params[hold_var_key] = tk.IntVar(value=p_config.get("hold_default", 1))

                        anim_controls_frame = ttk.Frame(p_frame)
                        anim_controls_frame.pack(side=tk.LEFT)

                        mode_combo = ttk.Combobox(anim_controls_frame, textvariable=self.anim_modes[anim_mode_key], values=["Sine Wave", "Drunken Walk"], state="readonly", width=12)
                        mode_combo.pack(side=tk.LEFT)

                        amp_label = ttk.Label(anim_controls_frame, text="Amp/Step:")
                        amp_entry = ttk.Entry(anim_controls_frame, textvariable=self.anim_params[amp_var_key], width=5)

                        per_label = ttk.Label(anim_controls_frame, text="Per:")
                        per_entry = ttk.Entry(anim_controls_frame, textvariable=self.anim_params[per_var_key], width=5)

                        hold_label = ttk.Label(anim_controls_frame, text="Hold:")
                        hold_entry = ttk.Entry(anim_controls_frame, textvariable=self.anim_params[hold_var_key], width=5)

                        def _update_anim_widgets(event=None, mode_var=self.anim_modes[anim_mode_key], hl=hold_label, he=hold_entry, al=amp_label, ae=amp_entry, pl=per_label, pe=per_entry):
                            mode = mode_var.get()
                            if mode == "Sine Wave":
                                hl.pack_forget(); he.pack_forget()
                                al.pack(side=tk.LEFT, padx=(5,0)); ae.pack(side=tk.LEFT)
                                pl.pack(side=tk.LEFT, padx=(5,0)); pe.pack(side=tk.LEFT)
                            elif mode == "Drunken Walk":
                                pl.pack_forget(); pe.pack_forget()
                                al.pack(side=tk.LEFT, padx=(5,0)); ae.pack(side=tk.LEFT)
                                hl.pack(side=tk.LEFT, padx=(5,0)); he.pack(side=tk.LEFT)

                        mode_combo.bind("<<ComboboxSelected>>", _update_anim_widgets)
                        _update_anim_widgets()

    def _create_all_generator_param_widgets(self, *args):
        for gen_key, gen_class in AVAILABLE_GENERATORS.items():
            frame = ttk.Frame(self.generator_params_container)
            self.generator_param_frames[gen_key] = frame
            for param_key, p_config in gen_class.params.items():
                p_frame = ttk.Frame(frame); p_frame.pack(fill=tk.X, expand=True, pady=1)
                if p_config['type'] == 'multicolor':
                    self.generator_params[param_key] = tk.StringVar(value=p_config['default'])
                    var = self.generator_params[param_key]
                    ttk.Label(p_frame, text=p_config['label'], width=12).pack(side=tk.LEFT)
                    entry_frame = ttk.Frame(p_frame); entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    entry = ttk.Entry(entry_frame, textvariable=var)
                    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    btn_frame = ttk.Frame(entry_frame); btn_frame.pack(side=tk.LEFT, padx=2)
                    ttk.Button(btn_frame, text="+", width=2, command=lambda k=gen_key, pk=param_key: self._add_color(k, pk)).pack()
                    ttk.Button(btn_frame, text="-", width=2, command=lambda k=gen_key, pk=param_key: self._remove_color(k, pk)).pack()
                    
                    canvas_container = ttk.Frame(frame)
                    canvas_container.pack(fill=tk.X, pady=2, padx=(95, 0))
                    swatch_canvas = tk.Canvas(canvas_container, height=30, highlightthickness=0)
                    swatch_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=swatch_canvas.xview)
                    swatch_canvas.configure(xscrollcommand=swatch_scrollbar.set)
                    scrollable_swatch_frame = ttk.Frame(swatch_canvas)
                    swatch_canvas.create_window((0, 0), window=scrollable_swatch_frame, anchor="nw")
                    scrollable_swatch_frame.bind(
                        "<Configure>",
                        lambda e, canvas=swatch_canvas: canvas.configure(scrollregion=canvas.bbox("all"))
                    )
                    swatch_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
                    swatch_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
                    self.generator_swatch_frames[gen_key] = {'frame': scrollable_swatch_frame, 'container': canvas_container}
                    
                    def update_swatches_callback(event, v=var, gk=gen_key): self._update_color_swatches(gk, v.get())
                    entry.bind("<KeyRelease>", update_swatches_callback)
                    self._update_color_swatches(gen_key, p_config['default'])

                elif p_config['type'] in ['slider', 'slider_int']:
                    var_type = tk.IntVar if p_config['type'] == 'slider_int' else tk.DoubleVar
                    self.generator_params[param_key] = var_type(value=p_config['default'])
                    ttk.Label(p_frame, text=p_config['label'], width=12).pack(side=tk.LEFT)
                    ttk.Scale(p_frame, variable=self.generator_params[param_key], from_=p_config['min'], to=p_config['max']).pack(side=tk.LEFT, expand=True, fill=tk.X)
                    ttk.Entry(p_frame, textvariable=self.generator_params[param_key], width=7).pack(side=tk.LEFT)

    def _update_color_swatches(self, gen_key, color_string):
        if gen_key not in self.generator_swatch_frames: return
        
        swatch_info = self.generator_swatch_frames[gen_key]
        swatch_frame = swatch_info['frame']
        container = swatch_info['container']

        for widget in swatch_frame.winfo_children():
            widget.destroy()

        colors = [c.strip() for c in color_string.split(',') if c.strip()]
        
        if not colors:
            container.pack_forget()
            return
        else:
            container.pack(fill=tk.X, pady=2, padx=(95, 0))

        param_key = next((pk for pk, pv in AVAILABLE_GENERATORS[gen_key].params.items() if pv['type'] == 'multicolor'), None)
        if not param_key: return
        
        for i, color_hex in enumerate(colors):
            try:
                swatch = tk.Label(swatch_frame, text="", background=color_hex, width=4, relief='sunken', cursor="hand2")
                swatch.pack(side=tk.LEFT, padx=2)
                swatch.bind("<Button-1>", lambda e, gk=gen_key, pk=param_key, idx=i: self._edit_color(gk, pk, idx))
            except: pass

    def _add_color(self, gen_key, param_key):
        if not param_key: return
        color_var = self.generator_params[param_key]
        color_code = colorchooser.askcolor(title="Choose color")
        if not (color_code and color_code[1]): return
        current_colors = color_var.get(); new_color = color_code[1]
        new_color_string = f"{current_colors},{new_color}" if current_colors else new_color
        color_var.set(new_color_string)
        self._update_color_swatches(gen_key, new_color_string)

    def _remove_color(self, gen_key, param_key):
        if not param_key: return
        color_var = self.generator_params[param_key]
        colors = [c.strip() for c in color_var.get().split(',') if c.strip()]
        if len(colors) > 1:
            new_color_string = ",".join(colors[:-1])
        else:
            new_color_string = ""
        color_var.set(new_color_string)
        self._update_color_swatches(gen_key, new_color_string)

    def _edit_color(self, gen_key, param_key, index_to_edit):
        if not gen_key: return
        color_var = self.generator_params[param_key]
        colors = [c.strip() for c in color_var.get().split(',') if c.strip()]
        if index_to_edit >= len(colors): return
        initial_color = colors[index_to_edit]
        new_color_code = colorchooser.askcolor(initialcolor=initial_color, title="Choose color")
        if new_color_code and new_color_code[1]:
            colors[index_to_edit] = new_color_code[1]
            new_color_string = ",".join(colors)
            color_var.set(new_color_string)
            self._update_color_swatches(gen_key, new_color_string)

    def _get_active_multicolor_param(self):
        gen_key = self.selected_generator_key.get()
        if not gen_key: return None, None
        for param_key, p_config in AVAILABLE_GENERATORS[gen_key].params.items():
            if p_config['type'] == 'multicolor': return gen_key, param_key
        return gen_key, None
    
    def on_generator_selection_change(self, event=None):
        for frame in self.generator_param_frames.values(): frame.pack_forget()
        selection_indices = self.generator_listbox.curselection()
        if not selection_indices: return
        selected_name = self.generator_listbox.get(selection_indices[0])
        for key, gen_class in AVAILABLE_GENERATORS.items():
            if gen_class.name == selected_name:
                self.selected_generator_key.set(key)
                if key in self.generator_param_frames: self.generator_param_frames[key].pack(fill=tk.X, expand=True)
                break

    def generate_from_selected_generator(self):
        gen_key = self.selected_generator_key.get()
        if not gen_key: messagebox.showwarning("Warning", "Please select a generator from the list."); return
        gen_class = AVAILABLE_GENERATORS[gen_key]
        params = {}
        for param_key in gen_class.params.keys():
            if param_key in self.generator_params: params[param_key] = self.generator_params[param_key].get()
        try:
            w = int(self.entries["output_width"].get()); h = int(self.entries["output_height"].get())
            if w <= 0 or h <= 0: raise ValueError("Width and Height must be positive.")
            self.status_label.config(text=f"Generating {gen_class.name}..."); self.root.update_idletasks()
            generated_image = gen_class().generate(w, h, params)
            self.input_image_loaded = generated_image
            self.current_evolving_image = None
            self.input_image_filename_var.set(f"Generated: {gen_class.name}")
            self.reset_view()
            self.display_image(self.input_image_loaded)
            self.status_label.config(text="Generated image loaded.")
        except Exception as e:
            messagebox.showerror("Generation Error", f"Could not generate image:\n{e}")
            self.status_label.config(text="Generation failed."); traceback.print_exc()

    def schedule_interactive_update(self, event=None):
        if not self.input_image_loaded: return
        if self.after_id_preview: self.root.after_cancel(self.after_id_preview)
        self.after_id_preview = self.root.after(self.interactive_update_delay, self._perform_interactive_update)

    def on_pipeline_selection_change(self, event=None):
        for frame in self.param_frames.values(): frame.pack_forget()
        selection_indices = self.active_listbox.curselection()
        if not selection_indices: return
        selected_index = selection_indices[0]
        effect_key = self.active_pipeline[selected_index]
        if effect_key in self.param_frames: self.param_frames[effect_key].pack(fill=tk.X, expand=True)

    def add_effect_to_pipeline(self, *args):
        selection_indices = self.available_listbox.curselection()
        if not selection_indices: return
        selected_name = self.available_listbox.get(selection_indices[0])
        for key, effect_class in AVAILABLE_EFFECTS.items():
            if effect_class.name == selected_name:
                self.active_pipeline.append(key)
                self.active_listbox.insert(tk.END, selected_name)
                break

    def remove_effect_from_pipeline(self, *args):
        selection_indices = self.active_listbox.curselection()
        if not selection_indices: return
        idx = selection_indices[0]
        self.active_listbox.delete(idx)
        del self.active_pipeline[idx]
        self.on_pipeline_selection_change()

    def move_effect_up(self, *args):
        selection_indices = self.active_listbox.curselection()
        if not selection_indices or selection_indices[0] == 0: return
        idx = selection_indices[0]
        self.active_pipeline[idx], self.active_pipeline[idx - 1] = self.active_pipeline[idx - 1], self.active_pipeline[idx]
        name = self.active_listbox.get(idx)
        self.active_listbox.delete(idx); self.active_listbox.insert(idx - 1, name); self.active_listbox.selection_set(idx - 1)

    def move_effect_down(self, *args):
        selection_indices = self.active_listbox.curselection()
        if not selection_indices or selection_indices[0] == self.active_listbox.size() - 1: return
        idx = selection_indices[0]
        self.active_pipeline[idx], self.active_pipeline[idx + 1] = self.active_pipeline[idx + 1], self.active_pipeline[idx]
        name = self.active_listbox.get(idx)
        self.active_listbox.delete(idx); self.active_listbox.insert(idx + 1, name); self.active_listbox.selection_set(idx + 1)
        
    def _apply_cpu_effect(self, image_in, effect_key, params):
        """Applies a single effect using CPU (Pillow) logic."""
        try:
            if effect_key == "blur":
                radius = params.get("blur_radius", 0)
                if radius > 0:
                    return image_in.filter(ImageFilter.GaussianBlur(radius=radius))
            
            elif effect_key == "shape":
                w, h = image_in.size
                image_for_compositing = image_in.copy()
                inner_effect_name = params.get("shape_inner_effect", "None")

                # Step 1: Apply inner effect if one is selected
                if inner_effect_name != "None":
                    inner_effect_key = next((k for k, v in AVAILABLE_EFFECTS.items() if v.name == inner_effect_name), None)
                    if inner_effect_key:
                        temp_params = params.copy()
                        inner_strength = params.get("shape_inner_strength", 1.0)
                        
                        inner_config = self.effects_config.get(inner_effect_key, {})
                        for p_name, p_config in inner_config.get("params", {}).items():
                            if p_config.get("animatable"):
                                var_key = p_config["var_key"]
                                base_val = temp_params.get(var_key, p_config["default"])
                                temp_params[var_key] = base_val * inner_strength
                                break
                        
                        warped_image = self._apply_cpu_effect(image_in, inner_effect_key, temp_params)
                        
                        mask = Image.new('L', (w,h), 0)
                        draw_mask = ImageDraw.Draw(mask)
                        
                        center_x = params.get("shape_x", 0.5) * w
                        center_y = params.get("shape_y", 0.5) * h
                        size = params.get("shape_size", 0.2) * min(w, h)
                        box = [center_x - size/2, center_y - size/2, center_x + size/2, center_y + size/2]
                        shape_type = params.get("shape_type")

                        if shape_type == "Circle":
                            draw_mask.ellipse(box, fill=255)
                        elif shape_type == "Rectangle":
                            draw_mask.rectangle(box, fill=255)
                        elif shape_type == "Diamond":
                            p = [(center_x, center_y - size/2), (center_x + size/2, center_y), (center_x, center_y + size/2), (center_x - size/2, center_y)]
                            draw_mask.polygon(p, fill=255)
                        elif shape_type == "Triangle":
                            p = [(center_x, center_y - size/2), (center_x + size/2, center_y + size/2), (center_x - size/2, center_y + size/2)]
                            draw_mask.polygon(p, fill=255)
                        
                        image_for_compositing = Image.composite(warped_image, image_in, mask)

                # Step 2: Draw the shape and outline on top of the (possibly warped) image
                shape_layer = Image.new('RGBA', (w,h), (0,0,0,0))
                draw = ImageDraw.Draw(shape_layer)
                
                center_x = params.get("shape_x", 0.5) * w
                center_y = params.get("shape_y", 0.5) * h
                size = params.get("shape_size", 0.2) * min(w, h)
                color = params.get("shape_color", "#ffffff")
                has_outline = params.get("shape_outline", True)
                box = [center_x - size/2, center_y - size/2, center_x + size/2, center_y + size/2]
                
                fill_color = color
                outline_color = "black" if has_outline else None
                shape_type = params.get("shape_type")

                if shape_type == "Circle":
                    draw.ellipse(box, fill=fill_color, outline=outline_color, width=2)
                elif shape_type == "Rectangle":
                    draw.rectangle(box, fill=fill_color, outline=outline_color, width=2)
                elif shape_type == "Diamond":
                    p = [(center_x, center_y - size/2), (center_x + size/2, center_y), (center_x, center_y + size/2), (center_x - size/2, center_y)]
                    draw.polygon(p, fill=fill_color, outline=outline_color, width=2)
                elif shape_type == "Triangle":
                    p = [(center_x, center_y - size/2), (center_x + size/2, center_y + size/2), (center_x - size/2, center_y + size/2)]
                    draw.polygon(p, fill=fill_color, outline=outline_color, width=2)
                
                # Step 3: Blend final result
                final_image = Image.alpha_composite(image_for_compositing.convert('RGBA'), shape_layer).convert('RGB')
                blend_alpha = params.get("shape_blend", 0.5)
                return Image.blend(image_in, final_image, blend_alpha)
            
            # FIXED: De-indented the following elif blocks to be at the same level as the 'if effect_key == "blur":'
            elif effect_key == "unsharp_mask":
                radius = params.get("unsharp_radius", 2)
                percent = params.get("unsharp_percent", 150)
                return image_in.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent))
            elif effect_key == "sharpen":
                return image_in.filter(ImageFilter.SHARPEN)
            elif effect_key == "color":
                img = image_in
                brightness = params.get("brightness_factor", 1.0)
                contrast = params.get("contrast_factor", 1.0)
                saturation = params.get("saturation_factor", 1.0)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(contrast)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(saturation)
                return img
            elif effect_key == "rotate":
                angle = params.get("rotate_angle_value", 0.0)
                return image_in.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
            else:
                return image_in
        except Exception as e:
            print(f"CPU Effect Error for '{effect_key}': {e}")
            return image_in

    def _apply_evolution_pipeline_once(self, image_in, step_num=0):
        if step_num > 0:
            if self.enable_periodic_random_var.get():
                nth_frame = self.periodic_random_nth_frame_var.get()
                if nth_frame > 0 and step_num % nth_frame == 0:
                    strength = self.periodic_random_strength_var.get()
                    self.status_label.config(text=f"Randomizing parameters at step {step_num}...")
                    for effect_key in self.active_pipeline:
                        if effect_key in self.effects_config:
                            for param_name, p_config in self.effects_config[effect_key].get("params", {}).items():
                                if p_config.get("animatable"):
                                    var_key = p_config["var_key"]
                                    p_min, p_max = p_config["min"], p_config["max"]
                                    change_range = (p_max - p_min) * strength
                                    current_value = self.op_params[var_key].get()
                                    new_value = current_value + random.uniform(-change_range / 2, change_range / 2)
                                    new_value = max(p_min, min(p_max, new_value))
                                    self.op_params[var_key].set(new_value)
            
            for effect_key, config in self.effects_config.items():
                for param_name, p_config in config.get("params", {}).items():
                    if p_config.get("animatable"):
                        var_key = p_config["var_key"]
                        anim_mode_key = f"{var_key}_anim_mode"
                        anim_mode = self.anim_modes.get(anim_mode_key, tk.StringVar(value="Sine Wave")).get()
                        
                        base_val = self.anim_base_values.get(var_key, self.op_params[var_key].get())
                        final_val = base_val

                        if anim_mode == "Sine Wave":
                            amp_key = f"{var_key}_amp"; per_key = f"{var_key}_per"
                            amp = self.anim_params[amp_key].get(); period = self.anim_params[per_key].get()
                            if period > 0:
                                offset = amp * math.sin(2 * math.pi * step_num / period)
                                final_val = base_val + offset
                        # FIXED: De-indented this elif to be on the same level as the if above
                        elif anim_mode == "Drunken Walk":
                            if step_num == 1:
                                self.drunken_walk_values[var_key] = base_val
                            hold_key = f"{var_key}_hold"
                            hold_val = self.anim_params[hold_key].get()
                            if hold_val > 0 and step_num % hold_val == 0:
                                amp_key = f"{var_key}_amp"
                                max_step = self.anim_params[amp_key].get()
                                current_val = self.drunken_walk_values.get(var_key, base_val)
                                final_val = current_val + random.uniform(-max_step, max_step)
                                self.drunken_walk_values[var_key] = final_val
                            else:
                                final_val = self.drunken_walk_values.get(var_key, base_val)

                        clamped_val = max(p_config["min"], min(p_config["max"], final_val))
                        self.op_params[var_key].set(clamped_val)

        current_params = {key: var.get() for key, var in self.op_params.items()}
        processed_image = None
        
        if not self.gpu_evolver:
            processed_image = image_in.copy()
            for effect_key in self.active_pipeline:
                processed_image = self._apply_cpu_effect(processed_image, effect_key, current_params)
            
            if self.enable_brightness_boost_var.get() and step_num > 0:
                nth_frame = self.brightness_boost_nth_frame_var.get()
                if nth_frame > 0 and step_num % nth_frame == 0:
                    amount = self.brightness_boost_amount_var.get()
                    enhancer = ImageEnhance.Brightness(processed_image)
                    processed_image = enhancer.enhance(amount)
        else:
            img_tensor = pil_to_tensor(image_in).to(self.gpu_evolver.device)
            for effect_key in self.active_pipeline:
                effect_class = AVAILABLE_EFFECTS[effect_key]
                effect_instance = effect_class(self)
                img_tensor = effect_instance.apply(img_tensor, current_params)

            if self.enable_brightness_boost_var.get() and step_num > 0:
                nth_frame = self.brightness_boost_nth_frame_var.get()
                if nth_frame > 0 and step_num % nth_frame == 0:
                    amount = self.brightness_boost_amount_var.get()
                    img_tensor = TF.adjust_brightness(img_tensor, amount)
            
            processed_image = tensor_to_pil(img_tensor)
        
        feedback = current_params.get("feedback_amount", 1.0)
        final_image = Image.blend(image_in, processed_image, alpha=feedback) if feedback < 1.0 else processed_image
        return final_image

    def _perform_interactive_update(self, step_num=0):
        if not self.input_image_loaded: return
        source = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
        try:
            w = int(self.entries["output_width"].get()); h = int(self.entries["output_height"].get())
            if w <= 0 or h <= 0: return
            roi = self._get_image_roi_at_output_resolution(source, w, h, step_num)
            evolved = self._apply_evolution_pipeline_once(roi, step_num)
            self.current_evolving_image = evolved
            self.display_image(self.current_evolving_image)
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            traceback.print_exc()
            
    def trigger_multistep_evolution(self, *args):
        if not self.input_image_loaded: return
        self.stop_evolution_requested = False
        self.stop_button.config(state=tk.NORMAL)
        self.drunken_walk_values = {}
        self.anim_base_values = {key: var.get() for key, var in self.op_params.items()}
        frames_to_save = []
        should_save_frames = self.save_animation_frames_var.get()
        try:
            steps = int(self.entries["steps"].get()); w = int(self.entries["output_width"].get()); h = int(self.entries["output_height"].get())
            source_canvas = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
            current_frame = self._get_image_roi_at_output_resolution(source_canvas, w, h, step_num=0)
            if should_save_frames: frames_to_save.append(current_frame.copy())
            i = 0
            for i in range(steps):
                if self.stop_evolution_requested: break
                view_of_last_frame = self._get_image_roi_at_output_resolution(current_frame, w, h, step_num=i + 1)
                current_frame = self._apply_evolution_pipeline_once(view_of_last_frame, step_num=i + 1)
                if should_save_frames: frames_to_save.append(current_frame.copy())
                self.current_evolving_image = current_frame
                self.display_image(self.current_evolving_image)
                self.status_label.config(text=f"Step {i + 1}/{steps}")
                self.root.update_idletasks()
            self.status_label.config(text=f"Evolution finished at step {i + 1}.")
            if should_save_frames and frames_to_save:
                self.save_animation_frame_sequence(frames_to_save)
        # FIXED: Correctly indented 'finally' block
        finally:
            self.stop_button.config(state=tk.DISABLED)

    def save_animation_frame_sequence(self, frames):
        if not frames: return
        save_dir = filedialog.askdirectory(title="Select Parent Folder for Animation Frames")
        if not save_dir:
            self.status_label.config(text="Animation save cancelled.")
            return
        subfolder_name = f"evolved_frames_{random.randint(100, 999)}"
        frames_subdir = os.path.join(save_dir, subfolder_name)
        os.makedirs(frames_subdir, exist_ok=True)
        base_name = "frame"; num_digits = len(str(len(frames)))
        for i, frame_img in enumerate(frames):
            frame_fname = f"{base_name}_{i:0{num_digits}d}.png"
            frame_path = os.path.join(frames_subdir, frame_fname)
            try: frame_img.save(frame_path)
            except Exception as e: messagebox.showerror("Save Error", f"Could not save frame {frame_path}:\n{e}"); return
        messagebox.showinfo("Save Complete", f"{len(frames)} frames saved to:\n{frames_subdir}")
        self.status_label.config(text=f"Animation saved to {subfolder_name}")

    def on_hold_evolve_press(self, event):
        if not self.input_image_loaded: return
        self.hold_evolve_active = True
        self.drunken_walk_values = {}
        self.anim_base_values = {key: var.get() for key, var in self.op_params.items()}
        self.continuous_evolve_step()

    def on_hold_evolve_release(self, event):
        self.hold_evolve_active = False
        if self.hold_evolve_after_id: self.root.after_cancel(self.hold_evolve_after_id)

    def continuous_evolve_step(self):
        if not self.hold_evolve_active: return
        source = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
        w = int(self.entries["output_width"].get()); h = int(self.entries["output_height"].get())
        if w > 0 and h > 0:
            view = self._get_image_roi_at_output_resolution(source, w, h, step_num=0)
            evolved = self._apply_evolution_pipeline_once(view, step_num=0)
            self.current_evolving_image = evolved
            self.display_image(self.current_evolving_image)
        self.hold_evolve_after_id = self.root.after(self.hold_evolve_delay, self.continuous_evolve_step)

    def request_stop_evolution(self): self.stop_evolution_requested = True
    
    def reset_image_to_original(self):
        self.current_evolving_image = None
        if self.input_image_loaded: self.display_image(self.input_image_loaded)
        self.reset_view()
        self.status_label.config(text="Image and view reset.")
        
    def save_image_as(self):
        if not self.current_evolving_image: return
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        initial_filename = f"image_evolver_{timestamp}.png"
        path = filedialog.asksaveasfilename(
            initialfile=initial_filename,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
        )
        if path:
            try:
                self.current_evolving_image.save(path)
                self.status_label.config(text=f"Saved to {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
                
    def load_input_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.input_image_loaded = Image.open(path).convert("RGB")
            self.input_image_filename_var.set(os.path.basename(path))
            self.reset_image_to_original()
            
    def load_displacement_map_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.displacement_map_image = Image.open(path).convert("RGB")
            self.displacement_map_filename_var.set(os.path.basename(path))
            
    def load_gradient_file(self):
        path = filedialog.askopenfilename(filetypes=[("Gradient Files", "*.map *.ugr"), ("All Files", "*.*")])
        if not path: return

        hex_string = load_gradient_from_file(path)
        if not hex_string: return

        gen_key, param_key = self._get_active_multicolor_param()
        if not param_key:
            messagebox.showwarning("No Generator Selected", "Please select a generator with a color parameter first.")
            return

        self.generator_params[param_key].set(hex_string)
        self._update_color_swatches(gen_key, hex_string)
        self.status_label.config(text=f"Loaded gradient: {os.path.basename(path)}")
        
    def display_image(self, img):
        if not img: return
        try:
            display_copy = img.copy()
            display_copy.thumbnail((self.target_display_width, self.target_display_height))
            self.photo_image = ImageTk.PhotoImage(display_copy)
            self.image_display_label.config(image=self.photo_image)
        except Exception as e:
            print(f"Display error: {e}")
            
    def _capture_initial_display_size(self):
        self.root.update_idletasks()
        if self.image_display_label.winfo_width() > 10:
            self.target_display_width = self.image_display_label.winfo_width() - 10
            self.target_display_height = self.image_display_label.winfo_height() - 10
            
    def _get_image_roi_at_output_resolution(self, source_image, target_w, target_h, step_num=0):
        if not source_image: return None
        src_w, src_h = source_image.size

        final_cx = self.view_center_x_norm.get()
        final_cy = self.view_center_y_norm.get()
        if step_num > 0:
            if self.anim_enabled['pan_x'].get():
                amp = self.anim_params['pan_x_amp'].get(); period = self.anim_params['pan_x_per'].get()
                if period > 0: final_cx += amp * math.sin(2 * math.pi * step_num / period)
            if self.anim_enabled['pan_y'].get():
                amp = self.anim_params['pan_y_amp'].get(); period = self.anim_params['pan_y_per'].get()
                if period > 0: final_cy += amp * math.sin(2 * math.pi * step_num / period)

        panning_mode = self.panning_mode_var.get()

        if panning_mode == "Off":
            final_cx = max(0.0, min(1.0, final_cx))
            final_cy = max(0.0, min(1.0, final_cy))
            zoom = max(0.01, self.zoom_factor.get())
            center_x_abs = final_cx * src_w; center_y_abs = final_cy * src_h
            view_w = src_w / zoom; view_h = src_h / zoom
            x0 = center_x_abs - view_w / 2; y0 = center_y_abs - view_h / 2
            crop_box = (int(round(x0)), int(round(y0)), int(round(x0 + view_w)), int(round(y0 + view_h)))
            return source_image.crop(crop_box).resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            zoom = max(0.01, self.zoom_factor.get())
            view_w_on_src = src_w / zoom; view_h_on_src = src_h / zoom
            x0 = (final_cx * src_w) - (view_w_on_src / 2)
            y0 = (final_cy * src_h) - (view_h_on_src / 2)
            new_image = Image.new('RGB', (target_w, target_h))
            source_pixels = source_image.load(); new_pixels = new_image.load()
            for yt in range(target_h):
                for xt in range(target_w):
                    src_x = x0 + (xt / target_w) * view_w_on_src
                    src_y = y0 + (yt / target_h) * view_h_on_src
                    if panning_mode == "Tile":
                        wrapped_src_x = int(round(src_x)) % src_w
                        wrapped_src_y = int(round(src_y)) % src_h
                    else: # Mirror
                        rev_x = (int(round(src_x)) // src_w) % 2 == 1
                        rev_y = (int(round(src_y)) // src_h) % 2 == 1
                        wrapped_src_x = (src_w - 1) - (int(round(src_x)) % src_w) if rev_x else int(round(src_x)) % src_w
                        wrapped_src_y = (src_h - 1) - (int(round(src_y)) % src_h) if rev_y else int(round(src_y)) % src_h
                    new_pixels[xt, yt] = source_pixels[wrapped_src_x, wrapped_src_y]
            return new_image

    def on_pan_start(self, event):
        if not self.input_image_loaded: return
        self.panning_active = True
        self.pan_start_mouse_x = event.x; self.pan_start_mouse_y = event.y
        self.pan_start_view_cx = self.view_center_x_norm.get(); self.pan_start_view_cy = self.view_center_y_norm.get()
        self.image_display_label.config(cursor="fleur")
        
    def on_pan_drag(self, event):
        if not self.panning_active: return
        dx = event.x - self.pan_start_mouse_x; dy = event.y - self.pan_start_mouse_y
        label_w = self.image_display_label.winfo_width(); label_h = self.image_display_label.winfo_height()
        if label_w <= 1 or label_h <= 1: return
        zoom = self.zoom_factor.get()
        delta_cx = (dx / label_w) / zoom; delta_cy = (dy / label_h) / zoom
        new_cx = self.pan_start_view_cx - delta_cx; new_cy = self.pan_start_view_cy - delta_cy

        if self.panning_mode_var.get() == "Off":
            new_cx = max(0.0, min(1.0, new_cx))
            new_cy = max(0.0, min(1.0, new_cy))

        self.view_center_x_norm.set(new_cx)
        self.view_center_y_norm.set(new_cy)
        self.schedule_interactive_update()
        
    def on_pan_end(self, event):
        self.panning_active = False
        self.image_display_label.config(cursor=""); self.schedule_interactive_update()
        
    def zoom_in(self):
        self.zoom_factor.set(self.zoom_factor.get() * 1.2); self.schedule_interactive_update()
        
    def zoom_out(self):
        self.zoom_factor.set(self.zoom_factor.get() / 1.2); self.schedule_interactive_update()
        
    def reset_view(self):
        self.zoom_factor.set(1.0); self.view_center_x_norm.set(0.5); self.view_center_y_norm.set(0.5)
        self.schedule_interactive_update()

if __name__ == "__main__":
    root = tk.Tk()
    app = EvolverApp(root)
    root.mainloop()