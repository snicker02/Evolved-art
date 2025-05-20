# 1. IMPORTS
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageChops, ImageDraw, ImageColor
import colorsys 
import math
import traceback
import os 
import random 

# 2. HELPER FUNCTIONS 
def compose_affine(trans1, trans2):
    """Composes two affine transformations (Pillow's 6-tuple format)."""
    a1, b1, c1, d1, e1, f1 = trans1
    a2, b2, c2, d2, e2, f2 = trans2
    return (
        a1*a2 + b1*d2, a1*b2 + b1*e2, a1*c2 + b1*f2 + c1,
        d1*a2 + e1*d2, d1*b2 + e1*e2, d1*c2 + e1*f2 + f1
    )

def transform_point(p, trans):
    """Applies an affine transform to a point."""
    # Pillow affine: x_new = a*x + b*y + c, y_new = d*x + e*y + f
    x, y = p
    a, b, c, d, e, f = trans
    return (a*x + b*y + c, d*x + e*y + f)

def transform_polygon(poly, trans):
    """Applies an affine transform to a list of points (polygon)."""
    return [transform_point(p, trans) for p in poly]

# 3. MAIN APPLICATION CLASS DEFINITION
class ImageEvolverApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Image Evolver & Rep-Tiler v3.0.86") 
        self.current_evolving_image = None 
        self.current_reptile_image = None  
        self.photo_image = None
        self.input_image_loaded = None    
        self.input_image_filename_var = tk.StringVar(value="No image loaded for Evolver.")
        self.texture_image_for_reptile = None 
        self.texture_image_filename_var = tk.StringVar(value="No texture loaded for Rep-Tiler.")


        self.displacement_map_image = None 
        self.displacement_map_filename_var = tk.StringVar(value="No map loaded.")

        self.target_display_width = 500
        self.target_display_height = 500
        
        self.after_id_preview = None 
        self.interactive_update_delay = 75 

        self.symmetry_type_var = tk.StringVar(value="None")
        self.symmetry_options = [
            "None", "Horizontal (Left Master)", "Horizontal (Right Master)",
            "Vertical (Top Master)", "Vertical (Bottom Master)",
            "4-Way Mirror (Top-Left Master)", "2-Fold Rotational (Average)",
            "Kaleidoscope (6-fold)", "Kaleidoscope (8-fold)", 
            "Diagonal Mirror (TL-BR Master)", "Diagonal Mirror (TR-BL Master)" 
        ]

        self.zoom_factor = tk.DoubleVar(value=1.0)
        self.view_center_x_norm = tk.DoubleVar(value=0.5) 
        self.view_center_y_norm = tk.DoubleVar(value=0.5) 
        self.zoom_increment = 1.2
        self.pan_increment_base = 0.1

        self.panning_active = False
        self.pan_start_mouse_x = 0
        self.pan_start_mouse_y = 0
        self.pan_start_view_cx = 0.5
        self.pan_start_view_cy = 0.5
        
        self.hold_evolve_active = False
        self.hold_evolve_after_id = None
        self.hold_evolve_delay = 75

        self.save_animation_frames_var = tk.BooleanVar(value=False)

        self.post_pan_anim_enabled_var = tk.BooleanVar(value=False)
        self.post_pan_drift_steps_var = tk.IntVar(value=10) 
        self.post_pan_drift_delay_var = tk.IntVar(value=40) 
        self.post_pan_drift_amount_var = tk.DoubleVar(value=0.01)
        self.is_post_pan_anim_running = False
        self.post_pan_anim_dx_factor_dir = 0.0 
        self.post_pan_anim_dy_factor_dir = 0.0 
        self.post_pan_anim_current_step = 0
        self.post_pan_after_id = None

        self.starter_shape_type_var = tk.StringVar(value="Circle")
        self.starter_shape_options = ["Circle", "Square", "Horizontal Lines", "Vertical Lines", "Linear Gradient (H)", "Linear Gradient (V)", "Radial Gradient", "Noise (Grayscale)"]

        self.zoom_pan_timing_var = tk.StringVar(value="Process ROI (Pre-Effects)")
        self.zoom_pan_timing_options = ["Process ROI (Pre-Effects)", "View Full Image (Post-Effects)"]

        self.stop_evolution_requested = False

        self.app_mode_var = tk.StringVar(value="Image Evolver")
        self.app_mode_options = ["Image Evolver", "Rep-Tile Patterner"]

        self.reptile_type_var = tk.StringVar(value="L-Tromino")
        self.reptile_options = ["L-Tromino", "Ammann Rhombus", "Ammann Thin Rhombus", "Custom IFS Fractal"] 
        self.reptile_recursion_depth = tk.IntVar(value=2) 
        self.reptile_output_width = tk.IntVar(value=512)
        self.reptile_output_height = tk.IntVar(value=512)
        self.reptile_use_texture_var = tk.BooleanVar(value=False)
        
        self.reptile_texture_effect_var = tk.StringVar(value="Tile (Global Coords)")
        self.reptile_texture_effect_options = [
            "Tile (Global Coords)", 
            "Stretch to Unit", 
            "Rotate & Tile (per Unit)"
        ]
        self.reptile_background_type_var = tk.StringVar(value="Default Canvas (Black)") 
        self.reptile_background_options = [                                         
            "Default Canvas (Black)", "Solid Color (Light Gray)", 
            "Transparent (PNG only)", "Tile with Texture"
        ]

        # IFS Parameters
        self.ifs_iterations_var = tk.IntVar(value=50000) 
        self.ifs_unit_canvas_size_var = tk.IntVar(value=256) 
        self.ifs_scale_var = tk.DoubleVar(value=50.0) 
        self.ifs_offset_x_var = tk.DoubleVar(value=128.0) 
        self.ifs_offset_y_var = tk.DoubleVar(value=128.0) 
        self.ifs_point_color_var = tk.StringVar(value="palette") 
        self.ifs_tile_rotation_var = tk.DoubleVar(value=0.0)
        self.selected_ifs_example_var = tk.StringVar(value="Example 1 (Sierpinski-like)")


        # Define sets of IFS transformations
        self.ifs_example_sets = {
            "Example 1 (Sierpinski-like)": [
                {'coefs_xml': (0, 0.7071067811865476, -0.7071067811865475, 0, 0, 0), 'weight': 0.5, 'color_idx': 0},
                {'coefs_xml': (-0.5, 0.0, 0.0, -0.5, -1.0, -0.7071067811865476), 'weight': 0.25, 'color_idx': 1},
                {'coefs_xml': (0.5, 0.0, 0.0, 0.5, -0.5, 0), 'weight': 0.25, 'color_idx': 2}
            ],
            "Example 2 (User Provided)": [
                {'coefs_xml': (0, 0.7071067811865476, 0.7071067811865475, 0, 0, 0), 'weight': 0.5, 'color_idx': 0},
                {'coefs_xml': (0.5, 0.0, 0.0, 0.5, 0.5, 0), 'weight': 0.25, 'color_idx': 1},
                {'coefs_xml': (-0.5, 0.0, 0.0, -0.5, 0.5, 0.7071067811865476), 'weight': 0.25, 'color_idx': 2}
            ]
        }
        
        self.ifs_transform_vars = [] 
        default_example_data = self.ifs_example_sets[self.selected_ifs_example_var.get()]
        for i in range(max(len(data_set) for data_set in self.ifs_example_sets.values())): 
            if i < len(default_example_data):
                data = default_example_data[i]
                xml_c = data['coefs_xml']
                self.ifs_transform_vars.append({
                    'a': tk.DoubleVar(value=xml_c[0]),      
                    'b': tk.DoubleVar(value=xml_c[1]),      
                    'c': tk.DoubleVar(value=xml_c[4]),      
                    'd': tk.DoubleVar(value=xml_c[2]),      
                    'e': tk.DoubleVar(value=xml_c[3]),      
                    'f': tk.DoubleVar(value=xml_c[5]),      
                    'weight': tk.DoubleVar(value=data['weight']),
                    'color_idx': tk.IntVar(value=data.get('color_idx', i)) 
                })
            else: 
                 self.ifs_transform_vars.append({
                    'a': tk.DoubleVar(value=0.0), 'b': tk.DoubleVar(value=0.0), 'c': tk.DoubleVar(value=0.0),
                    'd': tk.DoubleVar(value=0.0), 'e': tk.DoubleVar(value=0.0), 'f': tk.DoubleVar(value=0.0),
                    'weight': tk.DoubleVar(value=0.0),'color_idx': tk.IntVar(value=i) 
                })

        self.ifs_color_palette = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]


        self.entries = {}
        self.op_vars = {} 
        self.op_params = {} 
        self.anim_vars = {} 
        self.anim_params = {}
        
        self.operations_config = {
            "blur": {"var_key": "blur_enabled", "label": "Blur", "params": {"radius": {"var_key": "blur_radius", "default": 0.0, "min": 0.0, "max": 5.0, "label": "Radius:", "anim_config": {"amp_default": 0.5, "period_default": 40, "amp_min":0, "amp_max":2, "period_min":10, "period_max":100}}}},
            "unsharp_mask": {"var_key": "unsharp_enabled", "label": "Unsharp Mask", "params": {"radius": {"var_key": "unsharp_radius", "default": 2, "min": 0, "max": 10, "label": "Rad:", "is_int": True},"percent": {"var_key": "unsharp_percent", "default": 150, "min": 50, "max": 300, "label": "%:", "is_int": True},"threshold": {"var_key": "unsharp_threshold", "default": 3, "min": 0, "max": 10, "label": "Thr:", "is_int": True}}},
            "edge_blend": {"var_key": "edge_blend_enabled", "label": "Edge Blend", "params": {"alpha": {"var_key": "edge_blend_alpha", "default": 0.1, "min": 0.0, "max": 1.0, "label": "Alpha:"}}},
            "pixelate": {"var_key": "pixelate_enabled", "label": "Pixelate", "params": {"block_size": {"var_key": "pixelate_block_size", "default": 8, "min": 2, "max": 64, "label": "Block:", "is_int": True}}},
            "channel_shift": {"var_key": "chanshift_enabled", "label": "Channel Shift", "params": {"r_x": {"var_key": "chanshift_rx", "default": 0, "min": -10, "max": 10, "label": "R X:", "is_int": True}, "r_y": {"var_key": "chanshift_ry", "default": 0, "min": -10, "max": 10, "label": "R Y:", "is_int": True},"b_x": {"var_key": "chanshift_bx", "default": 0, "min": -10, "max": 10, "label": "B X:", "is_int": True}, "b_y": {"var_key": "chanshift_by", "default": 0, "min": -10, "max": 10, "label": "B Y:", "is_int": True}}},
            "shear": {"var_key": "shear_enabled", "label": "Shear (Tiled)", "params": {"x_factor": {"var_key": "shear_x_factor", "default": 0.0, "min": -0.5, "max": 0.5, "label": "X Fact:"}}},
            "displacement_map": {"var_key": "displace_enabled", "label": "Displace Map", "params": {
                "x_scale": {"var_key": "displace_x_scale", "default": 10.0, "min": -50.0, "max": 50.0, "label": "X Scale:"},
                "y_scale": {"var_key": "displace_y_scale", "default": 10.0, "min": -50.0, "max": 50.0, "label": "Y Scale:"},
            }},
            "simple_tile": {"var_key": "tile_enabled", "label": "Simple Tile", "params": { 
                 "tile_scale": {"var_key": "tile_scale_factor", "default": 0.5, "min": 0.1, "max": 2.0, "label": "Tile Scale:"}
            }},
            "hue_shift": {"var_key": "hue_enabled", "label": "Hue Shift", "params": {"amount": {"var_key": "hue_amount", "default": 0.0, "min": -0.05, "max": 0.05, "label": "Amt/Stp:", "anim_config": {"amp_default": 0.02, "period_default": 50, "amp_min":0, "amp_max":0.1, "period_min":10, "period_max":100}}}},
            "rotate": {"var_key": "rotate_enabled", "label": "Rotate (Tiled)", "params": {"angle": {"var_key": "rotate_angle_value", "default": 0.0, "min": -45.0, "max": 45.0, "label": "Ang/Stp(°):", "anim_config": {"amp_default": 15.0, "period_default": 60, "amp_min":0, "amp_max":45, "period_min":10, "period_max":200}}}},
            "brightness": {"var_key": "brightness_enabled", "label": "Brightness", "params": {"factor": {"var_key": "brightness_factor", "default": 1.0, "min": 0.7, "max": 1.3, "label": "Factor:", "anim_config": {"amp_default": 0.1, "period_default": 30, "amp_min":0, "amp_max":0.3, "period_min":10, "period_max":100}}}},
            "contrast": {"var_key": "contrast_enabled", "label": "Contrast", "params": {"factor": {"var_key": "contrast_factor", "default": 1.0, "min": 0.7, "max": 1.3, "label": "Factor:"}}},
            "saturation": {"var_key": "saturation_enabled", "label": "Saturation", "params": {"factor": {"var_key": "saturation_factor", "default": 1.0, "min": 0.0, "max": 2.0, "label": "Factor:"}}}
        }
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)

        self.controls_area_frame = ttk.Frame(main_frame, padding=5)
        self.controls_area_frame.grid(row=0, column=0, sticky="nswe")
        main_frame.columnconfigure(0, weight=0, minsize=470) 
        main_frame.rowconfigure(0, weight=1)

        mode_frame = ttk.LabelFrame(self.controls_area_frame, text="Application Mode", padding=5)
        mode_frame.pack(fill="x", pady=5, anchor="n") 
        self.mode_combo = ttk.Combobox(mode_frame, textvariable=self.app_mode_var, values=self.app_mode_options, state="readonly", width=25)
        self.mode_combo.pack(padx=5, pady=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_app_mode_change)

        self.evolver_controls_frame = ttk.Frame(self.controls_area_frame)
        self.reptile_controls_frame = ttk.Frame(self.controls_area_frame)
        
        self._setup_evolver_gui(self.evolver_controls_frame) 
        self._setup_reptile_gui(self.reptile_controls_frame)
        
        self.image_display_label = ttk.Label(main_frame, relief="sunken", anchor="center", background="#2B2B2B")
        self.image_display_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew") 
        main_frame.columnconfigure(1, weight=1); main_frame.rowconfigure(0, weight=1) 
        
        self.image_display_label.bind("<Motion>", self.on_mouse_move_over_image)
        self.image_display_label.bind("<Leave>", self.on_mouse_leave_image)
        self.image_display_label.bind("<ButtonPress-1>", self.on_pan_start)
        self.image_display_label.bind("<B1-Motion>", self.on_pan_drag)
        self.image_display_label.bind("<ButtonRelease-1>", self.on_pan_end)

        self.root.after(200, self._capture_initial_display_size)
        self.on_app_mode_change() 

    def on_app_mode_change(self, event=None):
        selected_mode = self.app_mode_var.get()
        if selected_mode == "Image Evolver":
            self.reptile_controls_frame.pack_forget()
            self.evolver_controls_frame.pack(fill="both", expand=True)
            display_img = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
            if display_img: self.display_image(display_img)
            else: self.image_display_label.config(image='') 
            if hasattr(self, 'status_label'): self.status_label.config(text="Image Evolver mode.")
        elif selected_mode == "Rep-Tile Patterner":
            self.evolver_controls_frame.pack_forget()
            self.reptile_controls_frame.pack(fill="both", expand=True)
            if self.current_reptile_image: self.display_image(self.current_reptile_image)
            else: self.image_display_label.config(image='')
            if hasattr(self, 'status_label_reptile'): self.status_label_reptile.config(text="Rep-Tile Patterner mode.")
        self._toggle_ifs_sphinx_controls() 


    def _setup_evolver_gui(self, parent_frame):
        canvas = tk.Canvas(parent_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        self.evolver_scrollable_frame = ttk.Frame(canvas, padding="10") 
        self.evolver_scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"), width=e.width))
        canvas.create_window((0, 0), window=self.evolver_scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        csr = 0 
        input_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="Image Input (Evolver)", padding="10")
        input_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=5); csr += 1
        self.load_image_button_evolver = ttk.Button(input_frame, text="Load Image File", command=self.load_input_image) 
        self.load_image_button_evolver.grid(row=0, column=0, columnspan=2, sticky="we", padx=5, pady=2)
        self.loaded_image_label_evolver = ttk.Label(input_frame, textvariable=self.input_image_filename_var, wraplength=180) 
        self.loaded_image_label_evolver.grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(input_frame, text="Starter Shape:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.starter_shape_combo_evolver = ttk.Combobox(input_frame, textvariable=self.starter_shape_type_var, values=self.starter_shape_options, state="readonly", width=18) 
        self.starter_shape_combo_evolver.grid(row=1, column=1, sticky="we", padx=5, pady=2)
        self.generate_starter_button_evolver = ttk.Button(input_frame, text="Generate Starter", command=self.generate_and_load_starter_shape) 
        self.generate_starter_button_evolver.grid(row=1, column=2, columnspan=2, sticky="we", padx=5, pady=2)

        ttk.Label(self.evolver_scrollable_frame, text="Evolution Steps:").grid(row=csr, column=0, sticky=tk.W, padx=5, pady=3)
        if "steps" not in self.entries: self.entries["steps"] = tk.StringVar(value="10")
        ttk.Entry(self.evolver_scrollable_frame, textvariable=self.entries["steps"], width=10).grid(row=csr, column=1, sticky="we", padx=5, pady=3)
        ttk.Label(self.evolver_scrollable_frame, text="Proc. Width:").grid(row=csr, column=2, sticky=tk.W, padx=5, pady=3) 
        if "output_width" not in self.entries: self.entries["output_width"] = tk.StringVar(value="512")
        ttk.Entry(self.evolver_scrollable_frame, textvariable=self.entries["output_width"], width=10).grid(row=csr, column=3, sticky="we", padx=5, pady=3); csr += 1
        ttk.Label(self.evolver_scrollable_frame, text="Proc. Height:").grid(row=csr, column=0, sticky=tk.W, padx=5, pady=3) 
        if "output_height" not in self.entries: self.entries["output_height"] = tk.StringVar(value="512")
        ttk.Entry(self.evolver_scrollable_frame, textvariable=self.entries["output_height"], width=10).grid(row=csr, column=1, sticky="we", padx=5, pady=3); csr += 1

        op_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="Evolution Operations Pipeline", padding="10")
        op_frame.grid(row=csr, column=0, columnspan=4, sticky="new", padx=5, pady=10); csr +=1
        op_cr = 0
        for op_key_cfg, config in self.operations_config.items():
            default_op_state = op_key_cfg in [] 
            self.op_vars[config["var_key"]] = tk.BooleanVar(value=default_op_state);
            op_row_frame = ttk.Frame(op_frame); op_row_frame.grid(row=op_cr, column=0, columnspan=4, sticky=tk.W, pady=1)
            ttk.Checkbutton(op_row_frame, text=config["label"], variable=self.op_vars[config["var_key"]]).pack(side=tk.LEFT, padx=(0,5)) 
            param_controls_frame = ttk.Frame(op_row_frame); param_controls_frame.pack(side=tk.LEFT)
            param_gui_row = 0
            if op_key_cfg == "displacement_map": 
                ttk.Button(param_controls_frame, text="Load DMap", command=self.load_displacement_map_image, width=10).grid(row=param_gui_row, column=0, padx=(5,2))
                ttk.Label(param_controls_frame, textvariable=self.displacement_map_filename_var, width=15, wraplength=100).grid(row=param_gui_row, column=1, columnspan=2, sticky=tk.W, padx=2)
                param_gui_row +=1
                self.op_params["displace_map_channel"] = tk.StringVar(value="Luminance")
                displace_map_channels = ["Luminance", "Red", "Green", "Blue", "Alpha"]
                ttk.Label(param_controls_frame, text="Map Chan:").grid(row=param_gui_row, column=0, sticky=tk.E, padx=(5,0))
                dmap_channel_combo = ttk.Combobox(param_controls_frame, textvariable=self.op_params["displace_map_channel"], values=displace_map_channels, state="readonly", width=10)
                dmap_channel_combo.grid(row=param_gui_row, column=1, sticky=tk.W)
                dmap_channel_combo.bind("<<ComboboxSelected>>", self.schedule_interactive_update) 
                param_gui_row +=1
            for param_key_cfg, p_config in config["params"].items():
                param_var_key = p_config["var_key"]; var_type = tk.IntVar if p_config.get("is_int") else tk.DoubleVar
                self.op_params[param_var_key] = var_type(value=p_config["default"])
                current_col_param = 0
                if op_key_cfg == "displacement_map" and param_gui_row > 1 : current_col_param = 0 
                ttk.Label(param_controls_frame, text=p_config["label"]).grid(row=param_gui_row, column=current_col_param, sticky=tk.E, padx=(5,0)); current_col_param+=1
                scale_length = 50 
                ttk.Scale(param_controls_frame, variable=self.op_params[param_var_key], from_=p_config["min"], to=p_config["max"], orient=tk.HORIZONTAL, length=scale_length).grid(row=param_gui_row, column=current_col_param, sticky=tk.W, padx=(0,2)); current_col_param+=1
                ttk.Label(param_controls_frame, textvariable=self.op_params[param_var_key], width=5).grid(row=param_gui_row, column=current_col_param, sticky=tk.W, padx=(0,5)); current_col_param+=1
                if "anim_config" in p_config:
                    anim_var_key = f"{param_var_key}_anim_enabled"; self.anim_vars[anim_var_key] = tk.BooleanVar(value=False)
                    anim_amp_key = f"{param_var_key}_anim_amp"; self.anim_params[anim_amp_key] = tk.DoubleVar(value=p_config["anim_config"]["amp_default"])
                    anim_period_key = f"{param_var_key}_anim_period"; self.anim_params[anim_period_key] = tk.IntVar(value=p_config["anim_config"]["period_default"])
                    ttk.Checkbutton(param_controls_frame, text="A", variable=self.anim_vars[anim_var_key], width=2).grid(row=param_gui_row, column=current_col_param, sticky=tk.W, padx=(10,0)); current_col_param+=1
                    ttk.Entry(param_controls_frame, textvariable=self.anim_params[anim_amp_key], width=4).grid(row=param_gui_row, column=current_col_param, padx=(0,1)); current_col_param+=1
                    ttk.Entry(param_controls_frame, textvariable=self.anim_params[anim_period_key], width=4).grid(row=param_gui_row, column=current_col_param, padx=(0,1)); current_col_param+=1
                param_gui_row +=1
            op_cr += 1
        
        symmetry_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="Symmetry (Applied Last)", padding="10"); symmetry_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr +=1
        ttk.Label(symmetry_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.symmetry_combo_evolver = ttk.Combobox(symmetry_frame, textvariable=self.symmetry_type_var, values=self.symmetry_options, state="readonly", width=25); self.symmetry_combo_evolver.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5, pady=2) 
        
        view_controls_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="View & ROI Controls", padding="10"); view_controls_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr +=1
        vc_row = 0
        ttk.Label(view_controls_frame, text="Zoom/Pan Mode:").grid(row=vc_row, column=0, sticky=tk.W, padx=5, pady=2)
        self.zoom_pan_timing_combo_evolver = ttk.Combobox(view_controls_frame, textvariable=self.zoom_pan_timing_var, values=self.zoom_pan_timing_options, state="readonly", width=25) 
        self.zoom_pan_timing_combo_evolver.grid(row=vc_row, column=1, columnspan=3, sticky="ew", padx=5, pady=2)
        vc_row+=1
        zoom_btn_subframe = ttk.Frame(view_controls_frame); zoom_btn_subframe.grid(row=vc_row, column=0, columnspan=4, pady=2); vc_row+=1
        ttk.Button(zoom_btn_subframe, text="Zoom In (+)", command=self.zoom_in_view).pack(side=tk.LEFT, padx=5); ttk.Button(zoom_btn_subframe, text="Zoom Out (-)", command=self.zoom_out_view).pack(side=tk.LEFT, padx=5); ttk.Button(zoom_btn_subframe, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        pan_btn_subframe_top = ttk.Frame(view_controls_frame); pan_btn_subframe_top.grid(row=vc_row, column=0, columnspan=4, pady=1); vc_row+=1; ttk.Button(pan_btn_subframe_top, text="Pan Up (↑)", command=lambda: self.pan_view(0, -1)).pack()
        pan_btn_subframe_mid = ttk.Frame(view_controls_frame); pan_btn_subframe_mid.grid(row=vc_row, column=0, columnspan=4, pady=1); vc_row+=1; ttk.Button(pan_btn_subframe_mid, text="Pan Left (←)", command=lambda: self.pan_view(-1, 0)).pack(side=tk.LEFT, padx=40); ttk.Button(pan_btn_subframe_mid, text="Pan Right (→)", command=lambda: self.pan_view(1, 0)).pack(side=tk.RIGHT, padx=40)
        pan_btn_subframe_bot = ttk.Frame(view_controls_frame); pan_btn_subframe_bot.grid(row=vc_row, column=0, columnspan=4, pady=1); vc_row+=1; ttk.Button(pan_btn_subframe_bot, text="Pan Down (↓)", command=lambda: self.pan_view(0, 1)).pack()
        self.anim_vars["pan_x_anim_enabled"] = tk.BooleanVar(value=False); self.anim_params["pan_x_anim_amp"] = tk.DoubleVar(value=0.1); self.anim_params["pan_x_anim_period"] = tk.IntVar(value=50)
        pan_x_lfo_frame = ttk.Frame(view_controls_frame); pan_x_lfo_frame.grid(row=vc_row, column=0, columnspan=4, pady=2, sticky=tk.W); vc_row+=1; ttk.Checkbutton(pan_x_lfo_frame, text="Animate Pan X", variable=self.anim_vars["pan_x_anim_enabled"]).pack(side=tk.LEFT); ttk.Label(pan_x_lfo_frame, text="Amp:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_x_lfo_frame, textvariable=self.anim_params["pan_x_anim_amp"], width=5).pack(side=tk.LEFT); ttk.Label(pan_x_lfo_frame, text="Per:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_x_lfo_frame, textvariable=self.anim_params["pan_x_anim_period"], width=5).pack(side=tk.LEFT)
        self.anim_vars["pan_y_anim_enabled"] = tk.BooleanVar(value=False); self.anim_params["pan_y_anim_amp"] = tk.DoubleVar(value=0.1); self.anim_params["pan_y_anim_period"] = tk.IntVar(value=60)
        pan_y_lfo_frame = ttk.Frame(view_controls_frame); pan_y_lfo_frame.grid(row=vc_row, column=0, columnspan=4, pady=2, sticky=tk.W); vc_row+=1; ttk.Checkbutton(pan_y_lfo_frame, text="Animate Pan Y", variable=self.anim_vars["pan_y_anim_enabled"]).pack(side=tk.LEFT); ttk.Label(pan_y_lfo_frame, text="Amp:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_y_lfo_frame, textvariable=self.anim_params["pan_y_anim_amp"], width=5).pack(side=tk.LEFT); ttk.Label(pan_y_lfo_frame, text="Per:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_y_lfo_frame, textvariable=self.anim_params["pan_y_anim_period"], width=5).pack(side=tk.LEFT)
        
        post_pan_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="Post-Pan Animation", padding="10"); post_pan_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr += 1; ttk.Checkbutton(post_pan_frame, text="Enable Drift", variable=self.post_pan_anim_enabled_var).grid(row=0, column=0, sticky=tk.W); ttk.Label(post_pan_frame, text="Steps:").grid(row=0, column=1, sticky=tk.E, padx=(10,0)); ttk.Entry(post_pan_frame, textvariable=self.post_pan_drift_steps_var, width=5).grid(row=0, column=2, sticky=tk.W); ttk.Label(post_pan_frame, text="Delay(ms):").grid(row=1, column=1, sticky=tk.E, padx=(10,0)); ttk.Entry(post_pan_frame, textvariable=self.post_pan_drift_delay_var, width=5).grid(row=1, column=2, sticky=tk.W); ttk.Label(post_pan_frame, text="Amount:").grid(row=0, column=3, sticky=tk.E, padx=(10,0)); ttk.Scale(post_pan_frame, variable=self.post_pan_drift_amount_var, from_=0.001, to=0.05, orient=tk.HORIZONTAL, length=80).grid(row=0, column=4, sticky=tk.W); ttk.Label(post_pan_frame, textvariable=self.post_pan_drift_amount_var, width=6).grid(row=0, column=5, sticky=tk.W)
        
        feedback_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="Feedback Options", padding="10"); feedback_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr +=1; self.op_vars["blend_original_enabled"] = tk.BooleanVar(value=False); self.op_params["blend_alpha_value"] = tk.DoubleVar(value=0.01); 
        ttk.Checkbutton(feedback_frame, text="Blend with Original", variable=self.op_vars["blend_original_enabled"]).grid(row=0, column=0, sticky=tk.W); ttk.Label(feedback_frame, text="Alpha:").grid(row=0, column=1, sticky=tk.E, padx=2); ttk.Scale(feedback_frame, variable=self.op_params["blend_alpha_value"], from_=0.0, to=0.2, orient=tk.HORIZONTAL, length=100).grid(row=0, column=2, sticky=tk.W); ttk.Label(feedback_frame,textvariable=self.op_params["blend_alpha_value"], width=4).grid(row=0, column=3, sticky=tk.W)
        
        output_options_frame = ttk.LabelFrame(self.evolver_scrollable_frame, text="Output Options", padding="10"); output_options_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr += 1; ttk.Label(output_options_frame, text="Default Filename:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3); self.entries["default_filename"] = tk.StringVar(value="evolved_art.png"); ttk.Entry(output_options_frame, textvariable=self.entries["default_filename"], width=30).grid(row=0, column=1, columnspan=3, sticky="we", padx=5, pady=3); ttk.Checkbutton(output_options_frame, text="Save All Frames (Multi-Step)", variable=self.save_animation_frames_var).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=3)
        
        buttons_frame = ttk.Frame(self.evolver_scrollable_frame); buttons_frame.grid(row=csr, column=0, columnspan=4, pady=15, sticky="ew"); csr +=1
        buttons_frame.columnconfigure(0, weight=1); buttons_frame.columnconfigure(1, weight=1); buttons_frame.columnconfigure(2, weight=1); buttons_frame.columnconfigure(3, weight=1); buttons_frame.columnconfigure(4, weight=1) 
        self.preview_button = ttk.Button(buttons_frame, text="Preview Step", command=self.schedule_interactive_update) 
        self.preview_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.evolve_button_evolver = ttk.Button(buttons_frame, text="Multi-Step Evolve", command=self.trigger_multistep_evolution); self.evolve_button_evolver.grid(row=0, column=1, padx=2, pady=2, sticky="ew") 
        self.stop_button_evolver = ttk.Button(buttons_frame, text="Stop Evolve", command=self.request_stop_evolution, state=tk.DISABLED) 
        self.stop_button_evolver.grid(row=0, column=2, padx=2, pady=2, sticky="ew") 
        self.hold_evolve_button_evolver = ttk.Button(buttons_frame, text="Hold to Evolve"); self.hold_evolve_button_evolver.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="ew"); self.hold_evolve_button_evolver.bind("<ButtonPress-1>", self.on_hold_evolve_press); self.hold_evolve_button_evolver.bind("<ButtonRelease-1>", self.on_hold_evolve_release) 
        self.reset_button_evolver = ttk.Button(buttons_frame, text="Reset Image", command=self.reset_image_to_original); self.reset_button_evolver.grid(row=1, column=2, padx=2, pady=2, sticky="ew") 
        self.save_button_evolver = ttk.Button(buttons_frame, text="Save Image", command=self.save_image_as); self.save_button_evolver.grid(row=1, column=3, padx=2, pady=2, sticky="ew") 
        
        self.status_label = ttk.Label(self.evolver_scrollable_frame, text="Load an image. Adjust parameters and click 'Preview Step'.", wraplength=380); self.status_label.grid(row=csr, column=0, columnspan=4, pady=5, sticky=tk.W); csr+=1
        # End Evolver GUI

    def _setup_reptile_gui(self, parent_frame):
        csr_rep = 0
        ttk.Label(parent_frame, text="Rep-Tile Type:").grid(row=csr_rep, column=0, sticky=tk.W, padx=5, pady=5)
        self.reptile_type_combo = ttk.Combobox(parent_frame, textvariable=self.reptile_type_var, values=self.reptile_options, state="readonly", width=20)
        self.reptile_type_combo.grid(row=csr_rep, column=1,columnspan=3, sticky="ew", padx=5, pady=5); csr_rep+=1
        self.reptile_type_combo.bind("<<ComboboxSelected>>", self._toggle_ifs_sphinx_controls)

        # Controls common to most rep-tiles (L-Tromino, Sphinx)
        self.common_reptile_controls_frame = ttk.Frame(parent_frame)
        self.common_reptile_controls_frame.grid(row=csr_rep, column=0, columnspan=4, sticky="ew"); csr_rep+=1
        
        ttk.Label(self.common_reptile_controls_frame, text="Recursion Depth:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.recursion_depth_entry = ttk.Entry(self.common_reptile_controls_frame, textvariable=self.reptile_recursion_depth, width=5)
        self.recursion_depth_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(self.common_reptile_controls_frame, text="Output Width:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.common_reptile_controls_frame, textvariable=self.reptile_output_width, width=7).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.common_reptile_controls_frame, text="Output Height:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.common_reptile_controls_frame, textvariable=self.reptile_output_height, width=7).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Background Options
        bg_options_frame = ttk.LabelFrame(parent_frame, text="Background Options", padding=5)
        bg_options_frame.grid(row=csr_rep, column=0, columnspan=4, sticky="ew", pady=5); csr_rep+=1
        ttk.Label(bg_options_frame, text="Fill Empty Space:").pack(side=tk.LEFT, padx=5)
        self.reptile_bg_combo = ttk.Combobox(bg_options_frame, textvariable=self.reptile_background_type_var, values=self.reptile_background_options, state="readonly", width=25)
        self.reptile_bg_combo.pack(side=tk.LEFT, padx=5)


        texture_options_frame = ttk.LabelFrame(parent_frame, text="Texture Options", padding=5) 
        texture_options_frame.grid(row=csr_rep, column=0, columnspan=4, sticky="ew", pady=5); csr_rep+=1
        
        ttk.Checkbutton(texture_options_frame, text="Use Texture on Tiles", variable=self.reptile_use_texture_var).pack(side=tk.LEFT, padx=5) 
        self.reptile_texture_effect_combo = ttk.Combobox(texture_options_frame, textvariable=self.reptile_texture_effect_var, values=self.reptile_texture_effect_options, state="readonly", width=25)
        self.reptile_texture_effect_combo.pack(side=tk.LEFT, padx=5)
        
        load_texture_frame = ttk.Frame(parent_frame)
        load_texture_frame.grid(row=csr_rep, column=0, columnspan=4, sticky="ew", pady=2); csr_rep+=1
        self.reptile_load_texture_button = ttk.Button(load_texture_frame, text="Load Custom Texture", command=self.load_texture_for_reptile) 
        self.reptile_load_texture_button.pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Label(load_texture_frame, textvariable=self.texture_image_filename_var).pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)

        # --- Dynamic Parameter Frames ---
        # Sphinx parameters LabelFrame (kept for structure, but won't be shown as Sphinx is removed)
        self.sphinx_params_lf = ttk.LabelFrame(parent_frame, text="Sphinx Child Affine Transforms (a,b,c,d,e,f)", padding=5)
        # No actual controls for Sphinx are created here anymore, but the frame variable is kept for _toggle_ifs_sphinx_controls

        self.ifs_params_lf = ttk.LabelFrame(parent_frame, text="Custom IFS Parameters", padding=5)
        
        # IFS Example Loader
        ifs_example_loader_frame = ttk.Frame(self.ifs_params_lf)
        ifs_example_loader_frame.pack(fill="x", pady=(0,5))
        ttk.Label(ifs_example_loader_frame, text="Load Example:").pack(side=tk.LEFT, padx=5)
        self.ifs_example_combo = ttk.Combobox(ifs_example_loader_frame, textvariable=self.selected_ifs_example_var, 
                                              values=list(self.ifs_example_sets.keys()), state="readonly", width=25)
        self.ifs_example_combo.pack(side=tk.LEFT, padx=5)
        self.ifs_example_combo.bind("<<ComboboxSelected>>", self._load_selected_ifs_example)
        
        ifs_f1 = ttk.Frame(self.ifs_params_lf); ifs_f1.pack(fill="x")
        ttk.Label(ifs_f1, text="Iterations:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(ifs_f1, textvariable=self.ifs_iterations_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(ifs_f1, text="Point Color:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(ifs_f1, textvariable=self.ifs_point_color_var, width=10).pack(side=tk.LEFT, padx=5)

        ifs_f2 = ttk.Frame(self.ifs_params_lf); ifs_f2.pack(fill="x", pady=2)
        ttk.Label(ifs_f2, text="Unit Size:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(ifs_f2, textvariable=self.ifs_unit_canvas_size_var, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Label(ifs_f2, text="IFS Scale:").pack(side=tk.LEFT, padx=5) 
        ttk.Entry(ifs_f2, textvariable=self.ifs_scale_var, width=7).pack(side=tk.LEFT, padx=5)
        
        ifs_f3 = ttk.Frame(self.ifs_params_lf); ifs_f3.pack(fill="x", pady=2)
        ttk.Label(ifs_f3, text="Offset X:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(ifs_f3, textvariable=self.ifs_offset_x_var, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Label(ifs_f3, text="Offset Y:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(ifs_f3, textvariable=self.ifs_offset_y_var, width=7).pack(side=tk.LEFT, padx=5)
        
        ifs_f4 = ttk.Frame(self.ifs_params_lf); ifs_f4.pack(fill="x", pady=2) 
        ttk.Label(ifs_f4, text="Tile Rotation (deg):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(ifs_f4, textvariable=self.ifs_tile_rotation_var, width=7).pack(side=tk.LEFT, padx=5)

        self.ifs_coeffs_lf = ttk.LabelFrame(self.ifs_params_lf, text="IFS Transform Coefficients (a,b,c,d,e,f) & Weight", padding=3)
        self.ifs_coeffs_lf.pack(fill="x", pady=5)

        ifs_param_labels = ['a', 'b', 'c(tx)', 'd', 'e', 'f(ty)', 'w']
        for i in range(len(self.ifs_transform_vars)): 
            transform_f = ttk.Frame(self.ifs_coeffs_lf, padding=1)
            transform_f.pack(fill="x")
            ttk.Label(transform_f, text=f"T{i}:").pack(side=tk.LEFT, padx=(0,1))
            keys_ordered = ['a','b','c','d','e','f', 'weight']
            for k_idx, key in enumerate(keys_ordered):
                ttk.Label(transform_f, text=f"{ifs_param_labels[k_idx]}:").pack(side=tk.LEFT, padx=(2 if k_idx > 0 else 0,0))
                ttk.Entry(transform_f, textvariable=self.ifs_transform_vars[i][key], width=6).pack(side=tk.LEFT)
        
        self.root.after(100, self._toggle_ifs_sphinx_controls) 

        # Buttons for Rep-Tiler
        self.reptile_action_frame = ttk.Frame(parent_frame) 
        
        self.generate_reptile_button = ttk.Button(self.reptile_action_frame, text="Generate Rep-Tile Pattern", command=self.trigger_reptile_generation)
        self.generate_reptile_button.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.save_reptile_button = ttk.Button(self.reptile_action_frame, text="Save Rep-Tile", command=self.save_reptile_image)
        self.save_reptile_button.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.send_to_evolver_button = ttk.Button(self.reptile_action_frame, text="Send to Evolver", command=self.send_reptile_to_evolver)
        self.send_to_evolver_button.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.status_label_reptile = ttk.Label(parent_frame, text="Select Rep-Tile options and generate.")
        
        self._toggle_ifs_sphinx_controls() 


    def _toggle_ifs_sphinx_controls(self, event=None):
        """Shows/hides IFS or Sphinx specific controls based on reptile_type_var."""
        tile_type = self.reptile_type_var.get()
        
        if not hasattr(self, 'sphinx_params_lf') or not hasattr(self, 'ifs_params_lf') or \
           not hasattr(self, 'common_reptile_controls_frame') or not hasattr(self, 'recursion_depth_entry') or \
           not hasattr(self, 'generate_reptile_button') or not hasattr(self, 'status_label_reptile') or \
           not hasattr(self, 'reptile_action_frame'): 
            return 

        current_row = self._get_next_reptile_gui_row() 

        self.sphinx_params_lf.grid_remove() 
        self.ifs_params_lf.grid_remove()
        self.recursion_depth_entry.config(state='normal') 
        self.reptile_texture_effect_combo.config(state='readonly') 


        if tile_type == "Custom IFS Fractal":
            self.ifs_params_lf.grid(row=current_row, column=0, columnspan=4, sticky="ew", pady=5)
            current_row += 1 
        elif tile_type == "Sphinx Tile": # This option is no longer in self.reptile_options
            self.reptile_texture_effect_combo.config(state='disabled') 
        else: # L-Tromino or Ammann Rhombus
             self.reptile_texture_effect_combo.config(state='readonly') 
            
        self.reptile_action_frame.grid(row=current_row, column=0, columnspan=4, pady=5, sticky="ew")
        current_row +=1
        
        self.status_label_reptile.grid(row=current_row, column=0, columnspan=4, pady=5, sticky=tk.W)


    def _get_next_reptile_gui_row(self):
        """Helper to find the next available row in reptile_controls_frame for dynamic controls."""
        return 5 

    def _load_selected_ifs_example(self, event=None):
        """Loads the selected IFS example coefficients into the GUI variables."""
        selected_example_name = self.selected_ifs_example_var.get()
        if selected_example_name in self.ifs_example_sets:
            example_data = self.ifs_example_sets[selected_example_name]
            for i in range(len(self.ifs_transform_vars)):
                if i < len(example_data):
                    data = example_data[i]
                    xml_c = data['coefs_xml']
                    self.ifs_transform_vars[i]['a'].set(xml_c[0])
                    self.ifs_transform_vars[i]['b'].set(xml_c[1])
                    self.ifs_transform_vars[i]['c'].set(xml_c[4]) # tx
                    self.ifs_transform_vars[i]['d'].set(xml_c[2])
                    self.ifs_transform_vars[i]['e'].set(xml_c[3])
                    self.ifs_transform_vars[i]['f'].set(xml_c[5]) # ty
                    self.ifs_transform_vars[i]['weight'].set(data['weight'])
                    self.ifs_transform_vars[i]['color_idx'].set(data.get('color_idx', i))
                else: # Clear any extra vars if current example is smaller
                    self.ifs_transform_vars[i]['a'].set(0.0)
                    self.ifs_transform_vars[i]['b'].set(0.0)
                    self.ifs_transform_vars[i]['c'].set(0.0)
                    self.ifs_transform_vars[i]['d'].set(0.0)
                    self.ifs_transform_vars[i]['e'].set(0.0)
                    self.ifs_transform_vars[i]['f'].set(0.0)
                    self.ifs_transform_vars[i]['weight'].set(0.0)
                    self.ifs_transform_vars[i]['color_idx'].set(i)


    # --- METHOD DEFINITIONS ---
    def _capture_initial_display_size(self):
        self.root.update_idletasks();label_w=self.image_display_label.winfo_width();label_h=self.image_display_label.winfo_height()
        if label_w > 50 and label_h > 50: self.target_display_width=label_w-10;self.target_display_height=label_h-10

    def update_mouse_param_options(self): pass

    def schedule_interactive_update(self, event=None): 
        if self.app_mode_var.get() != "Image Evolver": return 
        if self.is_post_pan_anim_running: return 
        if not self.input_image_loaded: return
        if self.after_id_preview: self.root.after_cancel(self.after_id_preview)
        self.after_id_preview = self.root.after(self.interactive_update_delay, self._perform_interactive_update) 

    def _perform_interactive_update(self): 
        if self.app_mode_var.get() != "Image Evolver": return
        if not self.input_image_loaded: return
        source_for_pipeline = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
        if not source_for_pipeline : return 
        try:
            output_w = int(self.entries["output_width"].get()); output_h = int(self.entries["output_height"].get())
            if output_w <=0 or output_h <=0: return
            image_to_process = None
            zoom_pan_mode = self.zoom_pan_timing_var.get()
            if zoom_pan_mode == "Process ROI (Pre-Effects)":
                image_to_process = self._get_image_roi_at_output_resolution(source_for_pipeline, output_w, output_h, step_num_for_multistep=0)
            else: 
                if source_for_pipeline.size != (output_w, output_h): image_to_process = source_for_pipeline.copy().resize((output_w, output_h), Image.Resampling.LANCZOS)
                else: image_to_process = source_for_pipeline.copy()
            if not image_to_process: return
            evolved_one_step = self._apply_evolution_pipeline_once(image_to_process, step_num_for_multistep=0) 
            self.current_evolving_image = evolved_one_step 
            self.display_image(self.current_evolving_image) 
            self.status_label.config(text="Preview updated.")
        except Exception as e: self.status_label.config(text=f"Preview Error: {e}"); traceback.print_exc()

    def _get_image_roi_at_output_resolution(self, source_image, target_w, target_h, step_num_for_multistep=0):
        if not source_image: return None
        src_w, src_h = source_image.size
        current_center_x_norm = self.view_center_x_norm.get(); current_center_y_norm = self.view_center_y_norm.get()
        if step_num_for_multistep > 0: 
            if self.anim_vars.get("pan_x_anim_enabled", tk.BooleanVar(value=False)).get():
                amp = self.anim_params.get("pan_x_anim_amp", tk.DoubleVar(value=0.1)).get(); period = self.anim_params.get("pan_x_anim_period", tk.IntVar(value=50)).get()
                if period > 0: current_center_x_norm = max(0.0, min(1.0, self.view_center_x_norm.get() + amp * math.sin(2 * math.pi * step_num_for_multistep / period)))
            if self.anim_vars.get("pan_y_anim_enabled", tk.BooleanVar(value=False)).get():
                amp = self.anim_params.get("pan_y_anim_amp", tk.DoubleVar(value=0.1)).get(); period = self.anim_params.get("pan_y_anim_period", tk.IntVar(value=60)).get()
                if period > 0: current_center_y_norm = max(0.0, min(1.0, self.view_center_y_norm.get() + amp * math.sin(2 * math.pi * step_num_for_multistep / period)))
        zoom = max(0.01, min(self.zoom_factor.get(), 100.0))
        center_x_abs = current_center_x_norm * src_w; center_y_abs = current_center_y_norm * src_h
        view_w_on_src = src_w/zoom; view_h_on_src = src_h/zoom
        x0=center_x_abs-view_w_on_src/2; y0=center_y_abs-view_h_on_src/2; x1=x0+view_w_on_src; y1=y0+view_h_on_src
        crop_box=(max(0,int(round(x0))), max(0,int(round(y0))), min(src_w,int(round(x1))), min(src_h,int(round(y1))))
        if crop_box[2]<=crop_box[0] or crop_box[3]<=crop_box[1]:
            return source_image.copy().resize((target_w,target_h), Image.Resampling.LANCZOS)
        return source_image.crop(crop_box).resize((target_w,target_h), Image.Resampling.LANCZOS)

    def _apply_symmetry(self, image_in):
        symmetry_type = self.symmetry_type_var.get();
        if symmetry_type == "None" or image_in is None: return image_in
        img_to_sym = image_in.copy(); w, h = img_to_sym.size; hw, hh = w // 2, h // 2
        try:
            if symmetry_type == "Horizontal (Left Master)":
                if hw <= 0: return img_to_sym 
                left = img_to_sym.crop((0, 0, hw, h)); right_f = left.transpose(Image.FLIP_LEFT_RIGHT); img_to_sym.paste(right_f, (w - hw, 0))
            elif symmetry_type == "Horizontal (Right Master)":
                if hw <= 0: return img_to_sym
                right = img_to_sym.crop((hw, 0, w, h)); left_f = right.transpose(Image.FLIP_LEFT_RIGHT); img_to_sym.paste(left_f, (0, 0))
            elif symmetry_type == "Vertical (Top Master)":
                if hh <= 0: return img_to_sym
                top = img_to_sym.crop((0, 0, w, hh)); bottom_f = top.transpose(Image.FLIP_TOP_BOTTOM); img_to_sym.paste(bottom_f, (0, h - hh))
            elif symmetry_type == "Vertical (Bottom Master)":
                if hh <= 0: return img_to_sym
                bottom = img_to_sym.crop((0, hh, w, h)); top_f = bottom.transpose(Image.FLIP_TOP_BOTTOM); img_to_sym.paste(top_f, (0, 0))
            elif symmetry_type == "4-Way Mirror (Top-Left Master)":
                if hw <= 0 or hh <= 0: return img_to_sym 
                new_img = Image.new('RGB', (w,h), color=(0,0,0)) 
                tl = img_to_sym.crop((0,0,hw,hh))
                tr = tl.transpose(Image.FLIP_LEFT_RIGHT)
                bl = tl.transpose(Image.FLIP_TOP_BOTTOM)
                br = tr.transpose(Image.FLIP_TOP_BOTTOM)
                new_img.paste(tl,(0,0)); new_img.paste(tr,(hw,0)); 
                new_img.paste(bl,(0,hh)); new_img.paste(br,(hw,hh))
                img_to_sym = new_img
            elif symmetry_type == "2-Fold Rotational (Average)": img_to_sym = Image.blend(img_to_sym, img_to_sym.rotate(180), alpha=0.5)
            elif symmetry_type.startswith("Kaleidoscope"):
                folds = 6 if "6-fold" in symmetry_type else 8
                angle_slice_rad = math.pi / folds 
                output_img = Image.new(image_in.mode, (w, h))
                target_pixels = output_img.load(); source_pixels = image_in.load()
                for yt_target in range(h):
                    for xt_target in range(w):
                        vx = xt_target - (w/2.0); vy = yt_target - (h/2.0)
                        angle = math.atan2(vy, vx); radius = math.sqrt(vx*vx + vy*vy)
                        angle = angle % (2 * angle_slice_rad)
                        if angle > angle_slice_rad: angle = (2 * angle_slice_rad) - angle
                        xs = (w/2.0) + radius * math.cos(angle); ys = (h/2.0) + radius * math.sin(angle)
                        xs_int = max(0, min(w - 1, int(round(xs)))); ys_int = max(0, min(h - 1, int(round(ys))))
                        target_pixels[xt_target, yt_target] = source_pixels[xs_int, ys_int]
                img_to_sym = output_img
            elif symmetry_type == "Diagonal Mirror (TL-BR Master)": img_to_sym = img_to_sym.transpose(Image.TRANSPOSE)
            elif symmetry_type == "Diagonal Mirror (TR-BL Master)": img_to_sym = img_to_sym.transpose(Image.TRANSVERSE)
        except Exception as e: print(f"Symmetry Error ('{symmetry_type}'): {e}"); traceback.print_exc(); return image_in 
        return img_to_sym

    def _apply_evolution_pipeline_once(self, image_in, step_num_for_multistep=0):
        current_img = image_in.copy(); output_w, output_h = current_img.size; active_params = {}
        for op_key_cfg, op_config in self.operations_config.items():
            if "params" in op_config:
                for param_key_cfg, p_config in op_config["params"].items():
                    var_key = p_config["var_key"]; base_val = self.op_params[var_key].get(); final_val = base_val
                    anim_enabled_key = f"{var_key}_anim_enabled"
                    if step_num_for_multistep > 0 and anim_enabled_key in self.anim_vars and self.anim_vars[anim_enabled_key].get():
                        amp_key = f"{var_key}_anim_amp"; period_key = f"{var_key}_anim_period"
                        amplitude = self.anim_params[amp_key].get(); period = self.anim_params[period_key].get()
                        if period > 0: final_val = base_val + amplitude * math.sin(2 * math.pi * step_num_for_multistep / period); final_val = max(p_config["min"], min(p_config["max"], final_val))
                    active_params[var_key] = int(final_val) if p_config.get("is_int") else final_val
        
        blend_alpha_val = self.op_params["blend_alpha_value"].get()

        if self.op_vars["blur_enabled"].get(): current_img = current_img.filter(ImageFilter.GaussianBlur(radius=active_params["blur_radius"]))
        if self.op_vars.get("unsharp_enabled").get(): current_img = current_img.filter(ImageFilter.UnsharpMask(radius=active_params["unsharp_radius"], percent=active_params["unsharp_percent"], threshold=active_params["unsharp_threshold"]))
        if self.op_vars.get("edge_blend_enabled").get(): edges = current_img.filter(ImageFilter.FIND_EDGES).convert("RGB"); current_img = Image.blend(current_img, edges, alpha=active_params["edge_blend_alpha"])
        if self.op_vars.get("pixelate_enabled").get():
            bs=max(2,active_params["pixelate_block_size"]); w_px,h_px=current_img.size 
            if w_px//bs>0 and h_px//bs>0: tmp=current_img.resize((w_px//bs,h_px//bs),Image.Resampling.NEAREST); current_img=tmp.resize((w_px,h_px),Image.Resampling.NEAREST)
        if self.op_vars.get("chanshift_enabled").get():
            r,g,b=current_img.split(); r_s=ImageChops.offset(r,active_params["chanshift_rx"],active_params["chanshift_ry"]); b_s=ImageChops.offset(b,active_params["chanshift_bx"],active_params["chanshift_by"]); current_img=Image.merge("RGB",(r_s,g,b_s))
        
        if self.op_vars.get("shear_enabled").get():
            sx = active_params.get("shear_x_factor", 0.0)
            w_shear, h_shear = current_img.size
            new_sheared_image = Image.new('RGB', (w_shear, h_shear), (0,0,0)) 
            source_pixels = current_img.load()
            target_pixels = new_sheared_image.load()
            for yt_target in range(h_shear):
                for xt_target in range(w_shear):
                    xs_source_float = xt_target - sx * yt_target 
                    ys_source_float = yt_target     
                    xs_source_wrapped = int(round(xs_source_float)) % w_shear
                    ys_source_wrapped = int(round(ys_source_float)) % h_shear
                    target_pixels[xt_target, yt_target] = source_pixels[xs_source_wrapped, ys_source_wrapped]
            current_img = new_sheared_image
        
        if self.op_vars.get("displace_enabled", tk.BooleanVar(value=False)).get() and self.displacement_map_image:
            dmap = self.displacement_map_image.resize(current_img.size, Image.Resampling.LANCZOS)
            dmap_channel_name = self.op_params["displace_map_channel"].get()
            map_band_idx = {'Luminance': 0, 'Red': 0, 'Green': 1, 'Blue': 2, 'Alpha':3}.get(dmap_channel_name, 0)
            if dmap_channel_name == "Luminance": dmap_values = dmap.convert("L")
            elif dmap_channel_name in ["Red", "Green", "Blue"]: dmap_values = dmap.split()[map_band_idx]
            elif dmap_channel_name == "Alpha" and dmap.mode == 'RGBA': dmap_values = dmap.split()[3]
            else: dmap_values = dmap.convert("L")
            dmap_pixels = dmap_values.load(); source_pixels = current_img.load()
            displaced_img = Image.new('RGB', current_img.size); displaced_pixels = displaced_img.load()
            x_scale = active_params.get("displace_x_scale", 10.0); y_scale = active_params.get("displace_y_scale", 10.0)
            for y_out in range(output_h):
                for x_out in range(output_w):
                    map_val = dmap_pixels[x_out, y_out]; 
                    if isinstance(map_val, tuple): map_val = map_val[0]
                    norm_map_val = (map_val - 128) / 128.0
                    dx = norm_map_val * x_scale; dy = norm_map_val * y_scale
                    src_x = max(0, min(output_w - 1, int(round(x_out + dx))))
                    src_y = max(0, min(output_h - 1, int(round(y_out + dy))))
                    displaced_pixels[x_out, y_out] = source_pixels[src_x, src_y]
            current_img = displaced_img
        
        if self.op_vars.get("tile_enabled", tk.BooleanVar(value=False)).get():
            tile_scale = active_params.get("tile_scale_factor", 0.5)
            if tile_scale > 0:
                w_tile_src, h_tile_src = current_img.size 
                tile_w = int(w_tile_src * tile_scale); tile_h = int(h_tile_src * tile_scale)
                if tile_w > 0 and tile_h > 0:
                    small_tile = current_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                    tiled_output = Image.new('RGB', (w_tile_src, h_tile_src)) 
                    for y_pos in range(0, h_tile_src, tile_h):
                        for x_pos in range(0, w_tile_src, tile_w):
                            tiled_output.paste(small_tile, (x_pos, y_pos))
                    current_img = tiled_output

        if self.op_vars["hue_enabled"].get():
            try: hsv=current_img.convert('HSV');h_ch,s_ch,v_ch=hsv.split();h_data=[(p+int(active_params["hue_amount"]*255))%256 for p in h_ch.getdata()];h_ch.putdata(h_data);current_img=Image.merge('HSV',(h_ch,s_ch,v_ch)).convert('RGB')
            except Exception as e:print(f"HueErr:{e}")
        if self.op_vars["rotate_enabled"].get():
            ow,oh=current_img.size; tile_cvs=Image.new('RGB',(ow*3,oh*3),(30,30,30));
            for itx in range(3):
                for ity in range(3): tile_cvs.paste(current_img,(itx*ow,ity*oh))
            angle_to_rotate = active_params["rotate_angle_value"] 
            rot_tile=tile_cvs.rotate(angle_to_rotate,Image.Resampling.BICUBIC,expand=False,fillcolor=(30,30,30)); current_img=rot_tile.crop((ow,oh,ow*2,oh*2))
        if self.op_vars["brightness_enabled"].get(): enh=ImageEnhance.Brightness(current_img); current_img=enh.enhance(active_params["brightness_factor"])
        if self.op_vars["contrast_enabled"].get(): enh=ImageEnhance.Contrast(current_img); current_img=enh.enhance(active_params["contrast_factor"])
        if self.op_vars["saturation_enabled"].get(): enh=ImageEnhance.Color(current_img); current_img=enh.enhance(active_params["saturation_factor"])
        if self.op_vars["blend_original_enabled"].get() and self.input_image_loaded:
            original_roi = self._get_image_roi_at_output_resolution(self.input_image_loaded, output_w, output_h, step_num_for_multistep)
            if original_roi: current_img=Image.blend(current_img,original_roi,alpha=blend_alpha_val)
        current_img = self._apply_symmetry(current_img) 
        return current_img

    def on_mouse_move_over_image(self, event):
        if self.panning_active: self.on_pan_drag(event); return
    def on_mouse_leave_image(self, event): pass  
    def on_pan_start(self, event):
        self._cancel_post_pan_animation() 
        if not self.input_image_loaded: return
        self.panning_active = True; self.pan_start_mouse_x = event.x; self.pan_start_mouse_y = event.y
        self.pan_start_view_cx = self.view_center_x_norm.get(); self.pan_start_view_cy = self.view_center_y_norm.get()
        self.image_display_label.config(cursor="fleur")
    def on_pan_drag(self, event):
        if not self.panning_active or not self.input_image_loaded: return
        dx_mouse = event.x - self.pan_start_mouse_x; dy_mouse = event.y - self.pan_start_mouse_y
        label_w = self.image_display_label.winfo_width(); label_h = self.image_display_label.winfo_height()
        if label_w <=1 or label_h <=1 : return
        zoom = self.zoom_factor.get();
        if zoom == 0: return 
        delta_cx_norm = (dx_mouse / label_w) / zoom; delta_cy_norm = (dy_mouse / label_h) / zoom
        new_cx = self.pan_start_view_cx - delta_cx_norm; new_cy = self.pan_start_view_cy - delta_cy_norm
        self.view_center_x_norm.set(max(0.0, min(1.0, new_cx))); self.view_center_y_norm.set(max(0.0, min(1.0, new_cy)))
        self.schedule_interactive_update() 
    def on_pan_end(self, event):
        if self.panning_active: 
            self.panning_active = False; self.image_display_label.config(cursor="")
            if self.post_pan_anim_enabled_var.get():
                self._start_post_pan_animation(event.x - self.pan_start_mouse_x, event.y - self.pan_start_mouse_y, input_is_pixel_delta=True)
            else:
                self.schedule_interactive_update() 

    def load_input_image(self): 
        self._cancel_post_pan_animation()
        fpath = filedialog.askopenfilename(title="Select Input Image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All", "*.*")], parent=self.root)
        if fpath:
            try:
                img = Image.open(fpath);
                if img.mode in ('RGBA', 'LA', 'P'): img = img.convert('RGB')
                self.input_image_loaded = img; self.input_image_filename_var.set(fpath.split('/')[-1])
                self.status_label.config(text=f"Loaded: {self.input_image_filename_var.get()}")
                self.current_evolving_image = None; self.reset_view() 
            except Exception as e: messagebox.showerror("Load Error", f"Failed: {e}", parent=self.root); self.status_label.config(text="Load error."); traceback.print_exc()

    def load_displacement_map_image(self): 
        fpath = filedialog.askopenfilename(title="Select Displacement Map Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff"), ("All files", "*.*")],parent=self.root)
        if fpath:
            try:
                img = Image.open(fpath)
                self.displacement_map_image = img 
                self.displacement_map_filename_var.set(fpath.split('/')[-1])
                self.status_label.config(text=f"DMap: {self.displacement_map_filename_var.get()}")
            except Exception as e:
                messagebox.showerror("Map Load Error", f"Failed to load DMap: {e}", parent=self.root)
                self.displacement_map_image = None; self.displacement_map_filename_var.set("No map loaded.")
                traceback.print_exc()
    
    def load_texture_for_reptile(self): 
        self._cancel_post_pan_animation()
        fpath = filedialog.askopenfilename(
            title="Select Texture Image for Rep-Tile",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff"), ("All files", "*.*")],
            parent=self.root
        )
        if fpath:
            try:
                img = Image.open(fpath)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB') 
                self.texture_image_for_reptile = img
                self.texture_image_filename_var.set(fpath.split('/')[-1])
                if hasattr(self, 'status_label_reptile'): 
                    self.status_label_reptile.config(text=f"Texture: {self.texture_image_filename_var.get()}")
            except Exception as e:
                messagebox.showerror("Texture Load Error", f"Failed to load texture: {e}", parent=self.root)
                self.texture_image_for_reptile = None
                self.texture_image_filename_var.set("No texture loaded.")
                traceback.print_exc()

    def generate_and_load_starter_shape(self): 
        self._cancel_post_pan_animation()
        shape_type = self.starter_shape_type_var.get()
        try:
            w = int(self.entries["output_width"].get()); h = int(self.entries["output_height"].get())
            if w <= 0 or h <= 0: messagebox.showerror("Error", "Output W/H must be positive.", parent=self.root); return
            img = Image.new('RGB', (w, h), color='black'); draw = ImageDraw.Draw(img)
            if shape_type == "Circle": radius = min(w,h)//3; x0,y0=w//2-radius,h//2-radius; x1,y1=w//2+radius,h//2+radius; draw.ellipse([(x0,y0),(x1,y1)],fill='white',outline='gray')
            elif shape_type == "Square": side=min(w,h)//2; x0,y0=w//2-side//2,h//2-side//2; x1,y1=w//2+side//2,h//2+side//2; draw.rectangle([(x0,y0),(x1,y1)],fill='white',outline='gray')
            elif shape_type == "Horizontal Lines":
                for i in range(0,h,max(1,h//20)): draw.line([(0,i),(w,i)],fill='white',width=1)
            elif shape_type == "Vertical Lines":
                for i in range(0,w,max(1,w//20)): draw.line([(i,0),(i,h)],fill='white',width=1)
            elif shape_type == "Linear Gradient (H)":
                for xg in range(w): val=int((xg/w)*255); draw.line([(xg,0),(xg,h)],fill=(val,val,val))
            elif shape_type == "Linear Gradient (V)":
                for yg in range(h): val=int((yg/h)*255); draw.line([(0,yg),(w,yg)],fill=(val,val,val))
            elif shape_type == "Radial Gradient":
                mr=math.sqrt((w/2)**2+(h/2)**2) if w>0 and h>0 else 1
                for yg in range(h):
                    for xg in range(w): dist=math.sqrt((xg-w/2)**2+(yg-h/2)**2); val=int(255*(1-min(1,dist/mr if mr>0 else 1))); draw.point((xg,yg),fill=(val,val,val))
            elif shape_type == "Noise (Grayscale)":
                px=img.load(); 
                for yp in range(h):
                    for xp in range(w): v=random.randint(0,255); px[xp,yp]=(v,v,v)
            self.input_image_loaded=img; self.input_image_filename_var.set(f"Generated {shape_type}"); self.current_evolving_image=None; self.reset_view(); self.status_label.config(text=f"Generated {shape_type} loaded.")
        except Exception as e: messagebox.showerror("Starter Shape Error",f"{e}",parent=self.root); traceback.print_exc()

    def zoom_in_view(self): self._cancel_post_pan_animation(); self.zoom_factor.set(min(self.zoom_factor.get() * self.zoom_increment, 20.0)); self.schedule_interactive_update()
    def zoom_out_view(self): self._cancel_post_pan_animation(); self.zoom_factor.set(max(self.zoom_factor.get() / self.zoom_increment, 0.05)); self.schedule_interactive_update()
    def reset_view(self): 
        self._cancel_post_pan_animation()
        self.zoom_factor.set(1.0); self.view_center_x_norm.set(0.5); self.view_center_y_norm.set(0.5); 
        self.schedule_interactive_update()
    def pan_view(self, dx_factor, dy_factor): 
        self._cancel_post_pan_animation()
        if not self.input_image_loaded: return
        zoom = self.zoom_factor.get();
        if zoom == 0: return
        _px= (self.pan_increment_base/zoom)*dx_factor;_py=(self.pan_increment_base/zoom)*dy_factor
        self.view_center_x_norm.set(max(0.0,min(1.0,self.view_center_x_norm.get()+_px))); self.view_center_y_norm.set(max(0.0,min(1.0,self.view_center_y_norm.get()+_py)))
        if self.post_pan_anim_enabled_var.get():
            self._start_post_pan_animation(dx_factor, dy_factor, input_is_pixel_delta=False) 
        else:
            self.schedule_interactive_update()

    def trigger_main_action(self): 
        self._cancel_post_pan_animation()
        mode = self.app_mode_var.get()
        if mode == "Image Evolver":
            self.trigger_multistep_evolution()
        elif mode == "Rep-Tile Patterner":
            self.trigger_reptile_generation()

    def trigger_multistep_evolution(self):
        if not self.input_image_loaded: messagebox.showerror("Error", "Load input image first for Evolver.", parent=self.root); return
        
        self.stop_evolution_requested = False 
        self.stop_button_evolver.config(state=tk.NORMAL) 

        try:
            num_steps = int(self.entries["steps"].get())
            output_w = int(self.entries["output_width"].get()); output_h = int(self.entries["output_height"].get())
            if num_steps <= 0 or output_w <= 0 or output_h <= 0: messagebox.showerror("Input Error", "Steps, W, H must be > 0.", parent=self.root); return
            self.status_label.config(text=f"Multi-step Evolving ({num_steps} steps)..."); self.root.update_idletasks()
            
            current_processing_img = None
            zoom_pan_mode = self.zoom_pan_timing_var.get()

            if zoom_pan_mode == "Process ROI (Pre-Effects)":
                base_for_multistep_start_roi = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
                current_processing_img = self._get_image_roi_at_output_resolution(base_for_multistep_start_roi, output_w, output_h, step_num_for_multistep=0)
            else: 
                source_for_full_processing = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
                if source_for_full_processing.size != (output_w, output_h): current_processing_img = source_for_full_processing.copy().resize((output_w, output_h), Image.Resampling.LANCZOS)
                else: current_processing_img = source_for_full_processing.copy()

            if not current_processing_img: self.status_label.config(text="Error: Could not prepare ROI/Image for evolution."); return
            
            frames_to_save = []
            save_frames_enabled = self.save_animation_frames_var.get()
            
            for step in range(num_steps):
                if self.stop_evolution_requested: self.status_label.config(text=f"Evolution stopped at step {step}."); break
                if zoom_pan_mode == "Process ROI (Pre-Effects)":
                    roi_content_for_this_step = self._get_image_roi_at_output_resolution(current_processing_img, output_w, output_h, step_num_for_multistep=step + 1)
                    if not roi_content_for_this_step: break
                    current_processing_img = self._apply_evolution_pipeline_once(roi_content_for_this_step, step_num_for_multistep=step + 1)
                else: 
                    current_processing_img = self._apply_evolution_pipeline_once(current_processing_img, step_num_for_multistep=step + 1)

                if save_frames_enabled: frames_to_save.append(current_processing_img.copy())
                if (step + 1) % 1 == 0 or step == num_steps - 1:
                    self.current_evolving_image = current_processing_img.copy()
                    self.display_image(self.current_evolving_image)
                    self.status_label.config(text=f"Multi-Step Evolution: {step + 1}/{num_steps}")
                    self.root.update_idletasks()
            if not self.stop_evolution_requested: self.status_label.config(text=f"Multi-step evolution complete ({num_steps} steps).")
            if save_frames_enabled and frames_to_save: self.save_animation_frame_sequence(frames_to_save)
        except ValueError as ve: messagebox.showerror("Input Error", f"{ve}", parent=self.root); self.status_label.config(text=f"Input Error: {ve}")
        except Exception as e: messagebox.showerror("Evo Error", f"{e}", parent=self.root); self.status_label.config(text=f"Evo Error: {e}"); traceback.print_exc()
        finally: 
            self.stop_button_evolver.config(state=tk.DISABLED) 
            self.stop_evolution_requested = False
    
    def trigger_reptile_generation(self): 
        self._cancel_post_pan_animation()
        self.status_label_reptile.config(text="Generating Rep-Tile...")
        self.root.update_idletasks()
        
        texture_source_for_tiles = None 
        if self.reptile_use_texture_var.get(): 
            if self.texture_image_for_reptile: 
                texture_source_for_tiles = self.texture_image_for_reptile
            elif self.current_evolving_image:
                texture_source_for_tiles = self.current_evolving_image
            elif self.input_image_loaded:
                texture_source_for_tiles = self.input_image_loaded
        
        try:
            width = self.reptile_output_width.get()
            height = self.reptile_output_height.get()
            depth = self.reptile_recursion_depth.get() 
            tile_type = self.reptile_type_var.get()
            bg_type = self.reptile_background_type_var.get()


            if width <= 0 or height <= 0:
                messagebox.showerror("Input Error", "Output dimensions must be positive.", parent=self.root)
                self.status_label_reptile.config(text="Error: Invalid dimensions.")
                return
            # Depth check is now conditional inside _draw_custom_ifs_recursive
            # and for L-Tromino/Sphinx it's handled by their respective functions.

            output_image_mode = 'RGB'
            bg_fill_color = 'black' 
            if bg_type == "Transparent (PNG only)":
                output_image_mode = 'RGBA'
                bg_fill_color = (0,0,0,0) 
            elif bg_type == "Solid Color (Light Gray)":
                bg_fill_color = (200, 200, 200)
            
            output_image = Image.new(output_image_mode, (width, height), bg_fill_color)

            if bg_type == "Tile with Texture": 
                bg_texture_to_tile = self.texture_image_for_reptile 
                if not bg_texture_to_tile: 
                     bg_texture_to_tile = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded

                if bg_texture_to_tile:
                    tex_w_bg, tex_h_bg = bg_texture_to_tile.size
                    if tex_w_bg > 0 and tex_h_bg > 0:
                        source_bg_pixels = bg_texture_to_tile.load()
                        output_bg_pixels = output_image.load() 
                        for i_bg in range(width): 
                            for j_bg in range(height): 
                                output_bg_pixels[i_bg,j_bg] = source_bg_pixels[i_bg % tex_w_bg, j_bg % tex_h_bg]
                    else: 
                        if hasattr(self, 'status_label_reptile'): self.status_label_reptile.config(text="BG Texture invalid, using dark gray.")
                        ImageDraw.Draw(output_image).rectangle([0,0,width,height], fill=(100,100,100)) 
                else: 
                    if hasattr(self, 'status_label_reptile'): self.status_label_reptile.config(text="BG Texture not found, using dark gray.")
                    ImageDraw.Draw(output_image).rectangle([0,0,width,height], fill=(100,100,100))
            
            if tile_type == "L-Tromino":
                self._draw_l_tromino_recursive(output_image, texture_source_for_tiles, 0, 0, width, height, depth, 0) 
            elif tile_type == "Ammann Rhombus":
                self._draw_ammann_rhombus_recursive(output_image, texture_source_for_tiles, 0, 0, width, height, depth, 0, is_thin=False)
            elif tile_type == "Ammann Thin Rhombus":
                self._draw_ammann_thin_rhombus_recursive(output_image, texture_source_for_tiles, 0, 0, width, height, depth, 0) # Call the new method
            elif tile_type == "Custom IFS Fractal":
                initial_ifs_transform = (width, 0, 0, 0, height, 0) 
                self._draw_custom_ifs_recursive(output_image, texture_source_for_tiles, initial_ifs_transform, depth, 0)


            self.current_reptile_image = output_image
            self.display_image(self.current_reptile_image)
            status_msg = f"{tile_type} generated."
            if tile_type != "Custom IFS Fractal": 
                status_msg += f" (Depth: {depth})"
            else: 
                status_msg += f" (Iterations: {self.ifs_iterations_var.get()}, Depth: {depth})"
            self.status_label_reptile.config(text=status_msg)


        except Exception as e:
            messagebox.showerror("Rep-Tile Error", f"Could not generate Rep-Tile: {e}", parent=self.root)
            self.status_label_reptile.config(text=f"Error: {e}")
            traceback.print_exc()

    def _draw_single_l_tromino(self, canvas_img, texture_img_full_source, x_bbox, y_bbox, w_bbox, h_bbox, orientation, color_if_no_texture):
        draw = ImageDraw.Draw(canvas_img)
        s_w, s_h = math.ceil(w_bbox / 2.0), math.ceil(h_bbox / 2.0)
        s_w_int = max(1, int(s_w))
        s_h_int = max(1, int(s_h))

        if w_bbox < 2 or h_bbox < 2: 
            final_fill_color = color_if_no_texture
            if texture_img_full_source and self.reptile_use_texture_var.get():
                try:
                    patch_w = max(1, int(w_bbox))
                    patch_h = max(1, int(h_bbox))
                    
                    current_texture_to_sample = texture_img_full_source
                    if self.reptile_texture_effect_var.get() == "Rotate & Tile (per Unit)":
                         angle = random.choice([0, 90, 180, 270])
                         if texture_img_full_source: 
                            current_texture_to_sample = texture_img_full_source.copy().rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)

                    patch = Image.new('RGB', (patch_w, patch_h))
                    patch_pixels = patch.load()
                    texture_pixels_loaded = current_texture_to_sample.load() 
                    tex_w_full, tex_h_full = current_texture_to_sample.size

                    for i in range(patch_w):
                        for j in range(patch_h):
                            tex_coord_x = (int(x_bbox) + i) % tex_w_full
                            tex_coord_y = (int(y_bbox) + j) % tex_h_full
                            patch_pixels[i,j] = texture_pixels_loaded[tex_coord_x, tex_coord_y]
                    canvas_img.paste(patch, (int(x_bbox), int(y_bbox)))
                    return 
                except Exception: 
                    final_fill_color = (255,0,255) 
            draw.rectangle([x_bbox, y_bbox, x_bbox + w_bbox -1, y_bbox + h_bbox -1], fill=final_fill_color)
            return

        squares_to_render_params = [] 
        if orientation == 0: 
            squares_to_render_params = [ (x_bbox, y_bbox, s_w_int, s_h_int), (x_bbox, y_bbox + s_h_int, s_w_int, s_h_int), (x_bbox + s_w_int, y_bbox + s_h_int, s_w_int, s_h_int) ]
        elif orientation == 1: 
            squares_to_render_params = [ (x_bbox, y_bbox, s_w_int, s_h_int), (x_bbox + s_w_int, y_bbox, s_w_int, s_h_int), (x_bbox, y_bbox + s_h_int, s_w_int, s_h_int) ]
        elif orientation == 2: 
            squares_to_render_params = [ (x_bbox + s_w_int, y_bbox, s_w_int, s_h_int), (x_bbox, y_bbox, s_w_int, s_h_int), (x_bbox + s_w_int, y_bbox + s_h_int, s_w_int, s_h_int) ]
        elif orientation == 3: 
            squares_to_render_params = [ (x_bbox + s_w_int, y_bbox, s_w_int, s_h_int), (x_bbox, y_bbox + s_h_int, s_w_int, s_h_int), (x_bbox + s_w_int, y_bbox + s_h_int, s_w_int, s_h_int) ]
        
        texture_effect = self.reptile_texture_effect_var.get()
        use_texture_on_tiles = texture_img_full_source and self.reptile_use_texture_var.get()

        texture_for_this_l_unit = texture_img_full_source 
        if use_texture_on_tiles and texture_effect == "Rotate & Tile (per Unit)":
            angle = random.choice([0, 90, 180, 270])
            if texture_img_full_source: 
                texture_for_this_l_unit = texture_img_full_source.copy().rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
        
        for sq_x_abs, sq_y_abs, sq_w, sq_h in squares_to_render_params:
            if sq_w < 1 or sq_h < 1: continue

            if use_texture_on_tiles and texture_for_this_l_unit:
                texture_to_process_for_square = texture_img_full_source if texture_effect == "Stretch to Unit" else texture_for_this_l_unit
                
                if not texture_to_process_for_square or not hasattr(texture_to_process_for_square, 'size') or not all(s > 0 for s in texture_to_process_for_square.size):
                    draw.rectangle([sq_x_abs, sq_y_abs, sq_x_abs + sq_w -1 , sq_y_abs + sq_h -1 ], fill=(0,0,0)) 
                    continue
                
                tex_w_full, tex_h_full = texture_to_process_for_square.size
                source_texture_pixels = texture_to_process_for_square.load() 
                
                if texture_effect == "Stretch to Unit":
                    if sq_w > 0 and sq_h > 0:
                        patch = texture_to_process_for_square.resize((sq_w, sq_h), Image.Resampling.LANCZOS)
                        canvas_img.paste(patch, (int(sq_x_abs), int(sq_y_abs)))
                else: 
                    patch = Image.new('RGB', (sq_w, sq_h)) 
                    patch_pixels = patch.load()
                    
                    for i in range(sq_w): 
                        for j in range(sq_h): 
                            global_x_on_canvas = int(sq_x_abs + i)
                            global_y_on_canvas = int(sq_y_abs + j)
                            
                            tex_x = global_x_on_canvas % tex_w_full
                            tex_y = global_y_on_canvas % tex_h_full
                            
                            try:
                                patch_pixels[i, j] = source_texture_pixels[tex_x, tex_y]
                            except IndexError: 
                                patch_pixels[i, j] = (255, 0, 255) 
                            except Exception: 
                                patch_pixels[i,j] = (0,255,255) 
                    canvas_img.paste(patch, (int(sq_x_abs), int(sq_y_abs)))
            else: 
                draw.rectangle([sq_x_abs, sq_y_abs, sq_x_abs + sq_w -1 , sq_y_abs + sq_h -1 ], fill=color_if_no_texture)


    def _draw_l_tromino_recursive(self, canvas_img, texture_img, x, y, w, h, depth, current_orientation):
        if depth < 0 or w < 1 or h < 1: return
        
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)] 

        if depth == 0:
            self._draw_single_l_tromino(canvas_img, texture_img, x, y, w, h, current_orientation, colors[current_orientation % 4])
            return

        w2, h2 = w / 2.0, h / 2.0
        if w2 < 1 or h2 < 1: 
            self._draw_single_l_tromino(canvas_img, texture_img, x, y, w, h, current_orientation, colors[current_orientation % 4])
            return
        
        self._draw_l_tromino_recursive(canvas_img, texture_img, x, y, w2, h2, depth - 1, 1)          
        self._draw_l_tromino_recursive(canvas_img, texture_img, x + w2, y, w2, h2, depth - 1, 0)      
        self._draw_l_tromino_recursive(canvas_img, texture_img, x, y + h2, w2, h2, depth - 1, 0)      
        self._draw_l_tromino_recursive(canvas_img, texture_img, x + w2, y + h2, w2, h2, depth - 1, 3)  

    def _get_ammann_rhombus_unit_polygon(self, side_length=1.0, thin=False):
        """Defines a unit Ammann rhombus centered at origin.
        If thin is False, it's the fat rhombus (72/108 deg).
        If thin is True, it's the thin rhombus (36/144 deg).
        Pointy ends are along the y-axis.
        """
        acute_angle_deg = 36.0 if thin else 72.0
        alpha_rad_half = (acute_angle_deg / 2.0) * math.pi / 180.0
        
        d1_h = side_length * math.cos(alpha_rad_half) 
        d2_h = side_length * math.sin(alpha_rad_half) 

        if thin: 
            return [
                (0, d1_h),      # Top acute vertex
                (d2_h, 0),      # Right obtuse vertex
                (0, -d1_h),     # Bottom acute vertex
                (-d2_h, 0)      # Left obtuse vertex
            ]
        else: 
             return [
                (0, d1_h),      # Top
                (d2_h, 0),      # Right
                (0, -d1_h),     # Bottom
                (-d2_h, 0)      # Left
            ]

    def _draw_single_ammann_rhombus_generic(self, canvas_img, texture_img_full_source, x_bbox, y_bbox, w_bbox, h_bbox, color_if_no_texture, is_thin_rhombus):
        """Draws a single Ammann rhombus (fat or thin), scaled to fit bbox, optionally textured."""
        
        unit_poly_pts = self._get_ammann_rhombus_unit_polygon(side_length=1.0, thin=is_thin_rhombus)
        
        acute_angle_deg = 36.0 if is_thin_rhombus else 72.0
        unit_h = 2 * math.cos((acute_angle_deg/2.0) * math.pi / 180.0) 
        unit_w = 2 * math.sin((acute_angle_deg/2.0) * math.pi / 180.0) 

        if unit_w == 0 or unit_h == 0: return

        scale_x = w_bbox / unit_w
        scale_y = h_bbox / unit_h
        scale = min(scale_x, scale_y)

        transformed_poly = []
        center_x = x_bbox + w_bbox / 2.0
        center_y = y_bbox + h_bbox / 2.0
        for p_unit in unit_poly_pts:
            tx = center_x + p_unit[0] * scale
            ty = center_y + p_unit[1] * scale 
            transformed_poly.append((int(round(tx)), int(round(ty))))

        if len(transformed_poly) < 3: return

        current_tile_image = Image.new('RGBA', (int(round(w_bbox)), int(round(h_bbox))), (0,0,0,0))
        draw_tile = ImageDraw.Draw(current_tile_image)
        mask_poly = [(p[0] - x_bbox, p[1] - y_bbox) for p in transformed_poly]
        
        if self.reptile_use_texture_var.get() and texture_img_full_source:
            try:
                base_texture = texture_img_full_source
                if not isinstance(base_texture, Image.Image):
                     raise ValueError("Texture source is not a valid PIL Image.")
                if base_texture.mode != 'RGBA': base_texture = base_texture.convert('RGBA')
                else: base_texture = base_texture.copy()

                texture_effect = self.reptile_texture_effect_var.get()
                tex_prepared = None
                
                if texture_effect == "Rotate & Tile (per Unit)":
                    angle = random.choice([0, 90, 180, 270])
                    tex_rotated = base_texture.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
                    tex_prepared = tex_rotated.resize((int(round(w_bbox)), int(round(h_bbox))), Image.Resampling.LANCZOS)
                elif texture_effect == "Stretch to Unit":
                    tex_prepared = base_texture.resize((int(round(w_bbox)), int(round(h_bbox))), Image.Resampling.LANCZOS)
                else: # "Tile (Global Coords)"
                    tex_prepared = Image.new('RGBA', (int(round(w_bbox)), int(round(h_bbox))))
                    tex_w_src, tex_h_src = base_texture.size
                    src_pixels = base_texture.load()
                    patch_pixels = tex_prepared.load()
                    for i_tile in range(int(round(w_bbox))):
                        for j_tile in range(int(round(h_bbox))):
                            sample_x = (int(x_bbox) + i_tile) % tex_w_src
                            sample_y = (int(y_bbox) + j_tile) % tex_h_src
                            try: patch_pixels[i_tile, j_tile] = src_pixels[sample_x, sample_y]
                            except IndexError: patch_pixels[i_tile, j_tile] = (0,0,0,255)
                
                if tex_prepared.mode != 'RGBA': tex_prepared = tex_prepared.convert('RGBA')
                
                rhombus_mask_for_tile = Image.new('L', (int(round(w_bbox)), int(round(h_bbox))), 0)
                ImageDraw.Draw(rhombus_mask_for_tile).polygon(mask_poly, fill=255, outline=255)
                current_tile_image.paste(tex_prepared, (0,0), mask=rhombus_mask_for_tile)

            except Exception as e_tex:
                print(f"Error texturing Ammann rhombus: {e_tex}")
                traceback.print_exc()
                draw_tile.polygon(mask_poly, fill=color_if_no_texture + (255,), outline=None)
        else:
            draw_tile.polygon(mask_poly, fill=color_if_no_texture + (255,), outline=None)
        
        canvas_img.paste(current_tile_image, (int(round(x_bbox)), int(round(y_bbox))), mask=current_tile_image)

    def _draw_ammann_rhombus_recursive(self, canvas_img, texture_img, x, y, w, h, depth, color_idx_offset, is_thin=False):
        if depth < 0 or w < 1 or h < 1: return
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (128,0,128), (0,128,128)] 

        if depth == 0:
            self._draw_single_ammann_rhombus_generic(canvas_img, texture_img, x, y, w, h, colors[color_idx_offset % len(colors)], is_thin)
            return

        w2, h2 = w / 2.0, h / 2.0
        if w2 < 1 or h2 < 1: 
            self._draw_single_ammann_rhombus_generic(canvas_img, texture_img, x, y, w, h, colors[color_idx_offset % len(colors)], is_thin)
            return
        
        self._draw_ammann_rhombus_recursive(canvas_img, texture_img, x, y, w2, h2, depth - 1, color_idx_offset + 0, is_thin)          
        self._draw_ammann_rhombus_recursive(canvas_img, texture_img, x + w2, y, w2, h2, depth - 1, color_idx_offset + 1, is_thin)      
        self._draw_ammann_rhombus_recursive(canvas_img, texture_img, x, y + h2, w2, h2, depth - 1, color_idx_offset + 2, is_thin)      
        self._draw_ammann_rhombus_recursive(canvas_img, texture_img, x + w2, y + h2, w2, h2, depth - 1, color_idx_offset + 3, is_thin)  

    def _draw_ammann_thin_rhombus_recursive(self, canvas_img, texture_img, x, y, w, h, depth, color_idx_offset):
        """ Wrapper to call _draw_ammann_rhombus_recursive with is_thin=True """
        self._draw_ammann_rhombus_recursive(canvas_img, texture_img, x, y, w, h, depth, color_idx_offset, is_thin=True)


    def _draw_custom_ifs_recursive(self, main_canvas_img, texture_source_for_tiles, current_placement_transform, depth, color_offset_idx):
        if depth < 0:
            return

        unit_canvas_size = self.ifs_unit_canvas_size_var.get()
        if unit_canvas_size <= 0: unit_canvas_size = 256

        if depth == 0:
            ifs_unit_mask = Image.new('L', (unit_canvas_size, unit_canvas_size), 0)
            draw_mask = ImageDraw.Draw(ifs_unit_mask)
            
            num_points_ifs = self.ifs_iterations_var.get()
            ifs_render_scale = self.ifs_scale_var.get()
            ifs_render_offset_x = self.ifs_offset_x_var.get()
            ifs_render_offset_y = self.ifs_offset_y_var.get()
            
            x_ifs, y_ifs = 0.0, 0.0
            burn_in_ifs = 100
            
            transforms_for_choice_ifs = []
            weights_for_choice_ifs = []
            for i in range(len(self.ifs_transform_vars)):
                params = self.ifs_transform_vars[i]
                pillow_tf = (params['a'].get(), params['b'].get(), params['c'].get(),
                             params['d'].get(), params['e'].get(), params['f'].get())
                transforms_for_choice_ifs.append({'matrix': pillow_tf, 'color_idx': params['color_idx'].get()})
                weights_for_choice_ifs.append(params['weight'].get())

            if not transforms_for_choice_ifs or not weights_for_choice_ifs or sum(weights_for_choice_ifs) <= 0:
                return 

            for i in range(num_points_ifs):
                try:
                    chosen_transform_data_ifs = random.choices(transforms_for_choice_ifs, weights=weights_for_choice_ifs, k=1)[0]
                except ValueError: continue
                
                transform_matrix_ifs = chosen_transform_data_ifs['matrix']
                x_new_ifs, y_new_ifs = transform_point((x_ifs, y_ifs), transform_matrix_ifs)
                x_ifs, y_ifs = x_new_ifs, y_new_ifs

                if i > burn_in_ifs:
                    plot_x = int(x_ifs * ifs_render_scale + ifs_render_offset_x)
                    plot_y = int(unit_canvas_size - (y_ifs * ifs_render_scale + ifs_render_offset_y))
                    if 0 <= plot_x < unit_canvas_size and 0 <= plot_y < unit_canvas_size:
                        draw_mask.point((plot_x, plot_y), fill=255)
            
            ifs_tile_content = Image.new('RGBA', (unit_canvas_size, unit_canvas_size), (0,0,0,0))

            if self.reptile_use_texture_var.get() and texture_source_for_tiles:
                try:
                    base_texture = texture_source_for_tiles
                    if not isinstance(base_texture, Image.Image): 
                         raise ValueError("Texture source is not a valid PIL Image.")
                    if base_texture.mode != 'RGBA': base_texture = base_texture.convert('RGBA')
                    else: base_texture = base_texture.copy() 
                    
                    texture_effect = self.reptile_texture_effect_var.get()
                    tex_prepared = None

                    if texture_effect == "Rotate & Tile (per Unit)":
                        angle = random.choice([0, 90, 180, 270])
                        tex_rotated = base_texture.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC)
                        tex_prepared = tex_rotated.resize((unit_canvas_size, unit_canvas_size), Image.Resampling.LANCZOS)
                    elif texture_effect == "Stretch to Unit":
                        tex_prepared = base_texture.resize((unit_canvas_size, unit_canvas_size), Image.Resampling.LANCZOS)
                    else: # "Tile (Global Coords)"
                        tex_prepared = Image.new('RGBA', (unit_canvas_size, unit_canvas_size))
                        tex_w_src, tex_h_src = base_texture.size
                        src_pixels = base_texture.load()
                        patch_pixels = tex_prepared.load()
                        
                        final_tile_x_on_canvas, final_tile_y_on_canvas = transform_point((0,0), current_placement_transform)
                        
                        for i_tile in range(unit_canvas_size):
                            for j_tile in range(unit_canvas_size):
                                sample_x = (int(final_tile_x_on_canvas) + i_tile) % tex_w_src
                                sample_y = (int(final_tile_y_on_canvas) + j_tile) % tex_h_src
                                try:
                                    patch_pixels[i_tile, j_tile] = src_pixels[sample_x, sample_y]
                                except IndexError: 
                                    patch_pixels[i_tile, j_tile] = (0,0,0,255)


                    if tex_prepared.mode != 'RGBA': tex_prepared = tex_prepared.convert('RGBA')
                    ifs_tile_content.paste(tex_prepared, (0,0), mask=ifs_unit_mask)
                except Exception as e_tex_unit:
                    print(f"Error texturing IFS unit: {e_tex_unit} - {type(e_tex_unit)}")
                    traceback.print_exc()
                    if hasattr(self, 'status_label_reptile'): self.status_label_reptile.config(text="IFS Texturing Error. Check console.")
                    ImageDraw.Draw(ifs_tile_content).bitmap((0,0), ifs_unit_mask, fill=(255,0,255,255))

            else: # No texture
                point_color_str = self.ifs_point_color_var.get().lower()
                final_color = self.ifs_color_palette[color_offset_idx % len(self.ifs_color_palette)]
                if point_color_str != "palette":
                    try: final_color = ImageColor.getrgb(point_color_str)
                    except ValueError: pass 
                if len(final_color) == 3: final_color += (255,)
                ImageDraw.Draw(ifs_tile_content).bitmap((0,0), ifs_unit_mask, fill=final_color)

            tile_rotation = self.ifs_tile_rotation_var.get()
            if tile_rotation != 0:
                ifs_tile_content = ifs_tile_content.rotate(tile_rotation, resample=Image.Resampling.BICUBIC, expand=False)

            unit_square_for_placement = [(0,0), (1,0), (1,1), (0,1)] 
            target_quad_on_canvas_corners = transform_polygon(unit_square_for_placement, current_placement_transform)
            
            min_xt_global = min(p[0] for p in target_quad_on_canvas_corners)
            max_xt_global = max(p[0] for p in target_quad_on_canvas_corners)
            min_yt_global = min(p[1] for p in target_quad_on_canvas_corners)
            max_yt_global = max(p[1] for p in target_quad_on_canvas_corners)
            
            final_display_width = int(round(max_xt_global - min_xt_global))
            final_display_height = int(round(max_yt_global - min_yt_global))

            if final_display_width > 0 and final_display_height > 0:
                try:
                    pasted_tile = ifs_tile_content.resize((final_display_width, final_display_height), Image.Resampling.LANCZOS)
                    
                    if main_canvas_img.mode == 'RGB' and pasted_tile.mode == 'RGBA':
                        bg_slice = main_canvas_img.crop((int(min_xt_global), int(min_yt_global), 
                                                         int(round(min_xt_global)) + final_display_width, 
                                                         int(round(min_yt_global)) + final_display_height))
                        if bg_slice.mode != 'RGB': bg_slice = bg_slice.convert('RGB')
                        
                        temp_composite_tile = bg_slice.copy()
                        temp_composite_tile.paste(pasted_tile,(0,0), mask=pasted_tile.getchannel('A'))
                        main_canvas_img.paste(temp_composite_tile, (int(min_xt_global), int(min_yt_global)))
                    else:
                        main_canvas_img.paste(pasted_tile, (int(min_xt_global), int(min_yt_global)), mask=pasted_tile.getchannel('A') if pasted_tile.mode=='RGBA' else None)

                except Exception as e_paste:
                    print(f"Error pasting transformed IFS tile: {e_paste}")
                    traceback.print_exc()
            return 

        # Recursive Step
        meta_s = 0.5 
        meta_transforms = [
            (meta_s, 0, 0,      0, meta_s, 0     ),
            (meta_s, 0, meta_s, 0, meta_s, 0     ),
            (meta_s, 0, 0,      0, meta_s, meta_s),
            (meta_s, 0, meta_s, 0, meta_s, meta_s)
        ]

        for i, meta_tf_local in enumerate(meta_transforms):
            new_placement_transform = compose_affine(current_placement_transform, meta_tf_local) 
            self._draw_custom_ifs_recursive(main_canvas_img, texture_source_for_tiles, new_placement_transform, depth - 1, color_offset_idx + i)


    def request_stop_evolution(self): 
        self.stop_evolution_requested = True
        self.status_label.config(text="Stop requested for multi-step evolution...")


    def save_animation_frame_sequence(self, frames):
        if not frames: return
        base_fname_sugg = self.entries["default_filename"].get(); base_n, _ = os.path.splitext(base_fname_sugg)
        if not base_n: base_n = "evolved_anim"
        save_dir = filedialog.askdirectory(title="Select Folder for Animation Frames", parent=self.root)
        if not save_dir: self.status_label.config(text="Frame saving cancelled."); return
        frames_subdir = os.path.join(save_dir, f"{base_n}_frames_{random.randint(100,999)}")
        try:
            if not os.path.exists(frames_subdir): os.makedirs(frames_subdir)
            num_digits = len(str(len(frames)))
            for i, frame_img in enumerate(frames):
                frame_fname = f"frame_{i+1:0{num_digits}d}.png"; frame_path = os.path.join(frames_subdir, frame_fname)
                frame_img.save(frame_path)
            messagebox.showinfo("Frames Saved", f"{len(frames)} frames saved in:\n{frames_subdir}", parent=self.root)
            self.status_label.config(text=f"Frames saved to {frames_subdir}")
        except Exception as e: messagebox.showerror("Frame Save Error", f"{e}", parent=self.root); self.status_label.config(text=f"Error saving frames: {e}"); traceback.print_exc()

    def display_image(self, pil_img_to_display):
        if not pil_img_to_display:
            self.image_display_label.config(image='')
            self.photo_image = None
            return
        try:
            image_for_thumbnail = pil_img_to_display
            zoom_pan_mode = self.zoom_pan_timing_var.get()
            app_mode = self.app_mode_var.get()

            if app_mode == "Image Evolver" and zoom_pan_mode == "View Full Image (Post-Effects)":
                src_w, src_h = image_for_thumbnail.size
                zoom = max(0.01, min(self.zoom_factor.get(), 100.0))
                center_x_abs = self.view_center_x_norm.get() * src_w
                center_y_abs = self.view_center_y_norm.get() * src_h
                view_w_on_src = src_w / zoom; view_h_on_src = src_h / zoom
                x0 = center_x_abs - view_w_on_src / 2; y0 = center_y_abs - view_h_on_src / 2
                x1 = x0 + view_w_on_src; y1 = y0 + view_h_on_src
                crop_box = (max(0, int(round(x0))), max(0, int(round(y0))),
                            min(src_w, int(round(x1))), min(src_h, int(round(y1))))
                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    image_for_thumbnail = image_for_thumbnail.crop(crop_box)
            
            mdw = self.target_display_width if self.target_display_width > 0 else 300
            mdh = self.target_display_height if self.target_display_height > 0 else 300
            img_copy = image_for_thumbnail.copy(); img_copy.thumbnail((mdw, mdh), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img_copy); self.image_display_label.config(image=self.photo_image)
        except Exception as e: 
            current_status_label = self.status_label if self.app_mode_var.get() == "Image Evolver" else self.status_label_reptile
            current_status_label.config(text=f"Display Error: {e}"); 
            traceback.print_exc()

    def save_image_as(self):
        image_to_save = None
        status_label_to_use = self.status_label 
        default_fn = "output.png"

        if self.app_mode_var.get() == "Image Evolver":
            image_to_save = self.current_evolving_image
            default_fn = self.entries.get("default_filename", tk.StringVar(value="evolved_art.png")).get()
        elif self.app_mode_var.get() == "Rep-Tile Patterner":
            image_to_save = self.current_reptile_image
            default_fn = "reptile_pattern.png" 
            status_label_to_use = self.status_label_reptile


        if not image_to_save: messagebox.showwarning("No Image", "Generate an image first.", parent=self.root); return
        
        try:
            base_s, orig_e = os.path.splitext(default_fn)
            if not orig_e: orig_e = ".png"
            rand_s = f"{random.randint(0,9999):04d}"; init_file = f"{base_s}_{rand_s}{orig_e}"
            fpath = filedialog.asksaveasfilename(initialfile=init_file, defaultextension=orig_e, filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp"),("All","*.*")], parent=self.root)
            if fpath: 
                image_to_save.save(fpath); 
                status_label_to_use.config(text=f"Saved: {fpath}"); 
                messagebox.showinfo("Success", f"Saved to:\n{fpath}", parent=self.root)
            else: status_label_to_use.config(text="Save cancelled.")
        except Exception as e: messagebox.showerror("Save Error", f"{e}", parent=self.root); status_label_to_use.config(text=f"Save Error: {e}"); traceback.print_exc()

    def save_reptile_image(self):
        """Saves the current rep-tile image."""
        if not self.current_reptile_image:
            messagebox.showwarning("No Image", "Generate a Rep-Tile image first.", parent=self.root)
            return
        
        default_fn = f"{self.reptile_type_var.get().lower().replace(' ', '_')}_pattern.png"
        try:
            base_s, orig_e = os.path.splitext(default_fn)
            if not orig_e: orig_e = ".png"
            rand_s = f"{random.randint(0,9999):04d}"
            init_file = f"{base_s}_{rand_s}{orig_e}"
            
            fpath = filedialog.asksaveasfilename(
                initialfile=init_file, 
                defaultextension=orig_e,
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("All files", "*.*")],
                parent=self.root
            )
            if fpath:
                self.current_reptile_image.save(fpath)
                self.status_label_reptile.config(text=f"Saved: {fpath}")
                messagebox.showinfo("Success", f"Rep-Tile image saved to:\n{fpath}", parent=self.root)
            else:
                self.status_label_reptile.config(text="Save cancelled.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save Rep-Tile image: {e}", parent=self.root)
            self.status_label_reptile.config(text=f"Save Error: {e}")
            traceback.print_exc()

    def send_reptile_to_evolver(self):
        """Sends the current rep-tile image to the Image Evolver mode."""
        if not self.current_reptile_image:
            messagebox.showwarning("No Image", "Generate a Rep-Tile image first to send.", parent=self.root)
            return

        try:
            self.input_image_loaded = self.current_reptile_image.copy()
            self.input_image_filename_var.set(f"From Rep-Tiler ({self.reptile_type_var.get()})")
            self.current_evolving_image = None # Clear any previous evolution
            
            self.app_mode_var.set("Image Evolver")
            self.on_app_mode_change() # This will switch tabs and update display
            self.reset_view() # Reset zoom/pan for the new image in evolver
            
            if hasattr(self, 'status_label'):
                 self.status_label.config(text=f"Image from Rep-Tiler loaded. Output W/H set to {self.input_image_loaded.width}x{self.input_image_loaded.height}.")
            if hasattr(self, 'entries'): # Update evolver's output size to match
                self.entries["output_width"].set(str(self.input_image_loaded.width))
                self.entries["output_height"].set(str(self.input_image_loaded.height))

        except Exception as e:
            messagebox.showerror("Send Error", f"Could not send image to Evolver: {e}", parent=self.root)
            traceback.print_exc()


    def reset_image_to_original(self): 
        self._cancel_post_pan_animation()
        if self.app_mode_var.get() == "Image Evolver":
            if not self.input_image_loaded: messagebox.showwarning("No Image", "No image loaded for Evolver.", parent=self.root); return
            try:self.current_evolving_image=None;self.reset_view();self.status_label.config(text="Image & view reset.")
            except Exception as e:messagebox.showerror("Reset Error",f"{e}",parent=self.root);self.status_label.config(text=f"Reset Error:{e}");traceback.print_exc()
        elif self.app_mode_var.get() == "Rep-Tile Patterner":
            self.current_reptile_image = None
            self.display_image(None) 
            self.status_label_reptile.config(text="Rep-Tile canvas cleared.")


    def on_hold_evolve_press(self,event):
        if self.app_mode_var.get() != "Image Evolver": return
        self._cancel_post_pan_animation()
        if not self.input_image_loaded:messagebox.showerror("Error","Load image first.",parent=self.root);return
        self.hold_evolve_active=True;self.status_label.config(text="Continuously evolving...");self.continuous_evolve_step()
    def on_hold_evolve_release(self,event):
        self.hold_evolve_active=False
        if self.hold_evolve_after_id:self.root.after_cancel(self.hold_evolve_after_id);self.hold_evolve_after_id=None
        if hasattr(self, 'status_label') and self.status_label.cget("text").startswith("Continuously"):self.status_label.config(text="Continuous evolution stopped.")
    def continuous_evolve_step(self):
        if not self.hold_evolve_active or not self.input_image_loaded:self.hold_evolve_active=False;return
        source_for_pipeline = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
        if not source_for_pipeline: self.hold_evolve_active=False;return
        try:
            ow=int(self.entries["output_width"].get());oh=int(self.entries["output_height"].get())
            if ow<=0 or oh<=0:self.hold_evolve_active=False;return
            image_to_process = None
            zoom_pan_mode = self.zoom_pan_timing_var.get()
            if zoom_pan_mode == "Process ROI (Pre-Effects)":
                image_to_process = self._get_image_roi_at_output_resolution(source_for_pipeline, ow, oh, step_num_for_multistep=0)
            else: 
                if source_for_pipeline.size != (ow, oh): image_to_process = source_for_pipeline.copy().resize((ow,oh), Image.Resampling.LANCZOS)
                else: image_to_process = source_for_pipeline.copy()
            if not image_to_process:self.hold_evolve_active=False;return
            evolved_step=self._apply_evolution_pipeline_once(image_to_process,step_num_for_multistep=0) 
            self.current_evolving_image=evolved_step;self.display_image(self.current_evolving_image)
            if self.hold_evolve_active:self.hold_evolve_after_id=self.root.after(self.hold_evolve_delay,self.continuous_evolve_step)
        except Exception as e:self.status_label.config(text=f"Cont.Evo Err:{e}");self.hold_evolve_active=False;traceback.print_exc()

    def _start_post_pan_animation(self, last_pan_dx, last_pan_dy, input_is_pixel_delta=True):
        if not self.post_pan_anim_enabled_var.get() or not self.input_image_loaded: return
        self._cancel_post_pan_animation() 
        if input_is_pixel_delta: 
            self.post_pan_anim_dx_factor_dir = -1 if last_pan_dx > 1 else (1 if last_pan_dx < -1 else 0) 
            self.post_pan_anim_dy_factor_dir = -1 if last_pan_dy > 1 else (1 if last_pan_dy < -1 else 0)
        else: 
            self.post_pan_anim_dx_factor_dir = last_pan_dx
            self.post_pan_anim_dy_factor_dir = last_pan_dy
        if self.post_pan_anim_dx_factor_dir == 0 and self.post_pan_anim_dy_factor_dir == 0: return
        self.is_post_pan_anim_running = True
        self.post_pan_anim_current_step = 0
        if hasattr(self, 'status_label'): self.status_label.config(text="Post-pan drift...")
        self._run_post_pan_animation_step()

    def _run_post_pan_animation_step(self):
        if not self.is_post_pan_anim_running or not self.input_image_loaded: self.is_post_pan_anim_running = False; return
        max_steps = self.post_pan_drift_steps_var.get()
        if self.post_pan_anim_current_step >= max_steps:
            self.is_post_pan_anim_running = False; 
            if hasattr(self, 'status_label'): self.status_label.config(text="Post-pan drift finished."); 
            return
        drift_step_size_norm = self.post_pan_drift_amount_var.get()
        effective_pan_x_norm = self.post_pan_anim_dx_factor_dir * drift_step_size_norm
        effective_pan_y_norm = self.post_pan_anim_dy_factor_dir * drift_step_size_norm
        current_cx = self.view_center_x_norm.get(); current_cy = self.view_center_y_norm.get()
        new_center_x = current_cx + effective_pan_x_norm; new_center_y = current_cy + effective_pan_y_norm
        self.view_center_x_norm.set(max(0.0, min(1.0, new_center_x))); self.view_center_y_norm.set(max(0.0, min(1.0, new_center_y)))
        self.schedule_interactive_update() 
        self.post_pan_anim_current_step += 1
        delay_ms = self.post_pan_drift_delay_var.get()
        if delay_ms < 10: delay_ms = 10 
        self.post_pan_after_id = self.root.after(delay_ms, self._run_post_pan_animation_step)

    def _cancel_post_pan_animation(self):
        if self.post_pan_after_id: self.root.after_cancel(self.post_pan_after_id); self.post_pan_after_id = None
        self.is_post_pan_anim_running = False

# 4. SCRIPT EXECUTION
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ImageEvolverApp(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"Tkinter TclError: {e}")
        print("This application requires a graphical display environment to run.")
    except Exception as e:
        print(f"An unexpected error occurred at startup: {e}")
        traceback.print_exc()
