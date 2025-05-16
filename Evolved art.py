# 1. IMPORTS
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance, ImageChops
import colorsys 
import math
import traceback
import os 
import random 

# 2. HELPER FUNCTIONS (None needed at global scope for now)

# 3. MAIN APPLICATION CLASS DEFINITION
class ImageEvolverApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Image Evolver - Combobox Fix v2.13.6") 
        self.current_evolving_image = None
        self.photo_image = None
        self.input_image_loaded = None
        self.input_image_filename_var = tk.StringVar(value="No image loaded.")

        self.displacement_map_image = None 
        self.displacement_map_filename_var = tk.StringVar(value="No map loaded.")

        self.target_display_width = 500
        self.target_display_height = 500

        self.mouse_x_norm = tk.DoubleVar(value=0.5) 
        self.mouse_y_norm = tk.DoubleVar(value=0.5) 
        self.mouse_control_enabled = tk.BooleanVar(value=False)
        self.mouse_x_param_control = tk.StringVar(value="None")
        self.mouse_y_param_control = tk.StringVar(value="None")
        
        self.param_names_for_mouse = ["None"]
        self.after_id_preview = None 
        self.interactive_update_delay = 75 

        self.symmetry_type_var = tk.StringVar(value="None")
        self.symmetry_options = [
            "None", "Horizontal (Left Master)", "Horizontal (Right Master)",
            "Vertical (Top Master)", "Vertical (Bottom Master)",
            "4-Way Mirror (Top-Left Master)", "2-Fold Rotational (Average)"
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

        self.entries = {}
        self.op_vars = {} 
        self.op_params = {} 
        self.anim_vars = {} 
        self.anim_params = {}
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)

        canvas = tk.Canvas(main_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.controls_scrollable_frame = ttk.Frame(canvas, padding="10")
        self.controls_scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"), width=e.width))
        canvas.create_window((0, 0), window=self.controls_scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nswe")
        scrollbar.grid(row=0, column=1, sticky="ns")
        main_frame.columnconfigure(0, weight=0, minsize=460) 
        main_frame.rowconfigure(0, weight=1)
        
        csr = 0 

        self.load_image_button = ttk.Button(self.controls_scrollable_frame, text="Load Input Image", command=self.load_input_image)
        self.load_image_button.grid(row=csr, column=0, columnspan=4, sticky="we", padx=5, pady=5); csr += 1
        self.loaded_image_label = ttk.Label(self.controls_scrollable_frame, textvariable=self.input_image_filename_var, wraplength=380)
        self.loaded_image_label.grid(row=csr, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2); csr += 1

        ttk.Label(self.controls_scrollable_frame, text="Evolution Steps:").grid(row=csr, column=0, sticky=tk.W, padx=5, pady=3)
        self.entries["steps"] = tk.StringVar(value="50") 
        ttk.Entry(self.controls_scrollable_frame, textvariable=self.entries["steps"], width=10).grid(row=csr, column=1, sticky="we", padx=5, pady=3)
        ttk.Label(self.controls_scrollable_frame, text="ROI Proc. Width:").grid(row=csr, column=2, sticky=tk.W, padx=5, pady=3)
        self.entries["output_width"] = tk.StringVar(value="512")
        ttk.Entry(self.controls_scrollable_frame, textvariable=self.entries["output_width"], width=10).grid(row=csr, column=3, sticky="we", padx=5, pady=3); csr += 1
        ttk.Label(self.controls_scrollable_frame, text="ROI Proc. Height:").grid(row=csr, column=0, sticky=tk.W, padx=5, pady=3)
        self.entries["output_height"] = tk.StringVar(value="512")
        ttk.Entry(self.controls_scrollable_frame, textvariable=self.entries["output_height"], width=10).grid(row=csr, column=1, sticky="we", padx=5, pady=3); csr += 1

        op_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="Evolution Operations Pipeline", padding="10")
        op_frame.grid(row=csr, column=0, columnspan=4, sticky="new", padx=5, pady=10); csr +=1
        op_cr = 0
        self.operations_config = {
            "blur": {"var_key": "blur_enabled", "label": "Blur", "params": {"radius": {"var_key": "blur_radius", "default": 0.0, "min": 0.0, "max": 5.0, "label": "Radius:", "anim_config": {"amp_default": 0.5, "period_default": 40, "amp_min":0, "amp_max":2, "period_min":10, "period_max":100}}}},
            "unsharp_mask": {"var_key": "unsharp_enabled", "label": "Unsharp Mask", "params": {"radius": {"var_key": "unsharp_radius", "default": 2, "min": 0, "max": 10, "label": "Rad:", "is_int": True},"percent": {"var_key": "unsharp_percent", "default": 150, "min": 50, "max": 300, "label": "%:", "is_int": True},"threshold": {"var_key": "unsharp_threshold", "default": 3, "min": 0, "max": 10, "label": "Thr:", "is_int": True}}},
            "edge_blend": {"var_key": "edge_blend_enabled", "label": "Edge Blend", "params": {"alpha": {"var_key": "edge_blend_alpha", "default": 0.1, "min": 0.0, "max": 1.0, "label": "Alpha:"}}},
            "pixelate": {"var_key": "pixelate_enabled", "label": "Pixelate", "params": {"block_size": {"var_key": "pixelate_block_size", "default": 8, "min": 2, "max": 64, "label": "Block:", "is_int": True}}},
            "channel_shift": {"var_key": "chanshift_enabled", "label": "Channel Shift", "params": {"r_x": {"var_key": "chanshift_rx", "default": 0, "min": -10, "max": 10, "label": "R X:", "is_int": True}, "r_y": {"var_key": "chanshift_ry", "default": 0, "min": -10, "max": 10, "label": "R Y:", "is_int": True},"b_x": {"var_key": "chanshift_bx", "default": 0, "min": -10, "max": 10, "label": "B X:", "is_int": True}, "b_y": {"var_key": "chanshift_by", "default": 0, "min": -10, "max": 10, "label": "B Y:", "is_int": True}}},
            "shear": {"var_key": "shear_enabled", "label": "Shear", "params": {"x_factor": {"var_key": "shear_x_factor", "default": 0.0, "min": -0.3, "max": 0.3, "label": "X Fact:"}}},
            "displacement_map": {"var_key": "displace_enabled", "label": "Displace Map", "params": {
                "x_scale": {"var_key": "displace_x_scale", "default": 10.0, "min": -50.0, "max": 50.0, "label": "X Scale:"},
                "y_scale": {"var_key": "displace_y_scale", "default": 10.0, "min": -50.0, "max": 50.0, "label": "Y Scale:"},
            }},
            "hue_shift": {"var_key": "hue_enabled", "label": "Hue Shift", "params": {"amount": {"var_key": "hue_amount", "default": 0.0, "min": -0.05, "max": 0.05, "label": "Amt/Stp:", "anim_config": {"amp_default": 0.02, "period_default": 50, "amp_min":0, "amp_max":0.1, "period_min":10, "period_max":100}}}},
            "rotate": {"var_key": "rotate_enabled", "label": "Rotate (Tiled)", "params": {"angle": {"var_key": "rotate_angle_value", "default": 0.0, "min": -45.0, "max": 45.0, "label": "Ang/Stp(°):", "anim_config": {"amp_default": 15.0, "period_default": 60, "amp_min":0, "amp_max":45, "period_min":10, "period_max":200}}}},
            "brightness": {"var_key": "brightness_enabled", "label": "Brightness", "params": {"factor": {"var_key": "brightness_factor", "default": 1.0, "min": 0.7, "max": 1.3, "label": "Factor:", "anim_config": {"amp_default": 0.1, "period_default": 30, "amp_min":0, "amp_max":0.3, "period_min":10, "period_max":100}}}},
            "contrast": {"var_key": "contrast_enabled", "label": "Contrast", "params": {"factor": {"var_key": "contrast_factor", "default": 1.0, "min": 0.7, "max": 1.3, "label": "Factor:"}}},
            "saturation": {"var_key": "saturation_enabled", "label": "Saturation", "params": {"factor": {"var_key": "saturation_factor", "default": 1.0, "min": 0.0, "max": 2.0, "label": "Factor:"}}}
        }

        for op_key_cfg, config in self.operations_config.items():
            self.op_vars[config["var_key"]] = tk.BooleanVar(value=False);
            op_row_frame = ttk.Frame(op_frame); op_row_frame.grid(row=op_cr, column=0, columnspan=4, sticky=tk.W, pady=1)
            ttk.Checkbutton(op_row_frame, text=config["label"], variable=self.op_vars[config["var_key"]], command=self.schedule_interactive_update).pack(side=tk.LEFT, padx=(0,5))
            
            param_controls_frame = ttk.Frame(op_row_frame); param_controls_frame.pack(side=tk.LEFT)
            param_gui_row = 0

            if op_key_cfg == "displacement_map": 
                ttk.Button(param_controls_frame, text="Load DMap", command=self.load_displacement_map_image, width=10).grid(row=param_gui_row, column=0, padx=(5,2))
                ttk.Label(param_controls_frame, textvariable=self.displacement_map_filename_var, width=15, wraplength=100).grid(row=param_gui_row, column=1, columnspan=2, sticky=tk.W, padx=2)
                param_gui_row +=1
                
                self.op_params["displace_map_channel"] = tk.StringVar(value="Luminance")
                displace_map_channels = ["Luminance", "Red", "Green", "Blue", "Alpha"]
                ttk.Label(param_controls_frame, text="Map Chan:").grid(row=param_gui_row, column=0, sticky=tk.E, padx=(5,0))
                # CORRECTED: Added bind to the Combobox
                dmap_channel_combo = ttk.Combobox(param_controls_frame, textvariable=self.op_params["displace_map_channel"], values=displace_map_channels, state="readonly", width=10)
                dmap_channel_combo.grid(row=param_gui_row, column=1, sticky=tk.W)
                dmap_channel_combo.bind("<<ComboboxSelected>>", self.schedule_interactive_update)
                param_gui_row +=1


            for param_key_cfg, p_config in config["params"].items():
                param_var_key = p_config["var_key"]; var_type = tk.IntVar if p_config.get("is_int") else tk.DoubleVar
                self.op_params[param_var_key] = var_type(value=p_config["default"])
                param_display_name = config["label"] + " " + p_config["label"].replace(":", "")
                if param_display_name not in self.param_names_for_mouse: self.param_names_for_mouse.append(param_display_name)
                
                current_col_param = 0
                if op_key_cfg == "displacement_map" and param_gui_row > 1 : 
                    current_col_param = 0 
                
                ttk.Label(param_controls_frame, text=p_config["label"]).grid(row=param_gui_row, column=current_col_param, sticky=tk.E, padx=(5,0)); current_col_param+=1
                scale_length = 50 
                ttk.Scale(param_controls_frame, variable=self.op_params[param_var_key], from_=p_config["min"], to=p_config["max"], orient=tk.HORIZONTAL, length=scale_length, command=self.schedule_interactive_update).grid(row=param_gui_row, column=current_col_param, sticky=tk.W, padx=(0,2)); current_col_param+=1
                ttk.Label(param_controls_frame, textvariable=self.op_params[param_var_key], width=5).grid(row=param_gui_row, column=current_col_param, sticky=tk.W, padx=(0,5)); current_col_param+=1
                
                if "anim_config" in p_config:
                    anim_var_key = f"{param_var_key}_anim_enabled"; self.anim_vars[anim_var_key] = tk.BooleanVar(value=False)
                    anim_amp_key = f"{param_var_key}_anim_amp"; self.anim_params[anim_amp_key] = tk.DoubleVar(value=p_config["anim_config"]["amp_default"])
                    anim_period_key = f"{param_var_key}_anim_period"; self.anim_params[anim_period_key] = tk.IntVar(value=p_config["anim_config"]["period_default"])
                    ttk.Checkbutton(param_controls_frame, text="A", variable=self.anim_vars[anim_var_key], width=2, command=self.schedule_interactive_update).grid(row=param_gui_row, column=current_col_param, sticky=tk.W, padx=(10,0)); current_col_param+=1
                    ttk.Entry(param_controls_frame, textvariable=self.anim_params[anim_amp_key], width=4).grid(row=param_gui_row, column=current_col_param, padx=(0,1)); current_col_param+=1
                    ttk.Entry(param_controls_frame, textvariable=self.anim_params[anim_period_key], width=4).grid(row=param_gui_row, column=current_col_param, padx=(0,1)); current_col_param+=1
                param_gui_row +=1
            op_cr += 1
        
        symmetry_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="Symmetry (Applied Last)", padding="10"); symmetry_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr +=1
        ttk.Label(symmetry_frame, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.symmetry_combo = ttk.Combobox(symmetry_frame, textvariable=self.symmetry_type_var, values=self.symmetry_options, state="readonly", width=25); self.symmetry_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=5, pady=2); self.symmetry_combo.bind("<<ComboboxSelected>>", self.schedule_interactive_update)
        
        view_controls_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="View Controls (Defines ROI for Processing)", padding="10"); view_controls_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr +=1
        vc_row = 0
        zoom_btn_subframe = ttk.Frame(view_controls_frame); zoom_btn_subframe.grid(row=vc_row, column=0, columnspan=4, pady=2); vc_row+=1
        ttk.Button(zoom_btn_subframe, text="Zoom In (+)", command=self.zoom_in_view).pack(side=tk.LEFT, padx=5); ttk.Button(zoom_btn_subframe, text="Zoom Out (-)", command=self.zoom_out_view).pack(side=tk.LEFT, padx=5); ttk.Button(zoom_btn_subframe, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        pan_btn_subframe_top = ttk.Frame(view_controls_frame); pan_btn_subframe_top.grid(row=vc_row, column=0, columnspan=4, pady=1); vc_row+=1; ttk.Button(pan_btn_subframe_top, text="Pan Up (↑)", command=lambda: self.pan_view(0, -1)).pack()
        pan_btn_subframe_mid = ttk.Frame(view_controls_frame); pan_btn_subframe_mid.grid(row=vc_row, column=0, columnspan=4, pady=1); vc_row+=1; ttk.Button(pan_btn_subframe_mid, text="Pan Left (←)", command=lambda: self.pan_view(-1, 0)).pack(side=tk.LEFT, padx=40); ttk.Button(pan_btn_subframe_mid, text="Pan Right (→)", command=lambda: self.pan_view(1, 0)).pack(side=tk.RIGHT, padx=40)
        pan_btn_subframe_bot = ttk.Frame(view_controls_frame); pan_btn_subframe_bot.grid(row=vc_row, column=0, columnspan=4, pady=1); vc_row+=1; ttk.Button(pan_btn_subframe_bot, text="Pan Down (↓)", command=lambda: self.pan_view(0, 1)).pack()
        self.anim_vars["pan_x_anim_enabled"] = tk.BooleanVar(value=False); self.anim_params["pan_x_anim_amp"] = tk.DoubleVar(value=0.1); self.anim_params["pan_x_anim_period"] = tk.IntVar(value=50)
        pan_x_lfo_frame = ttk.Frame(view_controls_frame); pan_x_lfo_frame.grid(row=vc_row, column=0, columnspan=4, pady=2, sticky=tk.W); vc_row+=1; ttk.Checkbutton(pan_x_lfo_frame, text="Animate Pan X", variable=self.anim_vars["pan_x_anim_enabled"]).pack(side=tk.LEFT); ttk.Label(pan_x_lfo_frame, text="Amp:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_x_lfo_frame, textvariable=self.anim_params["pan_x_anim_amp"], width=5).pack(side=tk.LEFT); ttk.Label(pan_x_lfo_frame, text="Per:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_x_lfo_frame, textvariable=self.anim_params["pan_x_anim_period"], width=5).pack(side=tk.LEFT)
        self.anim_vars["pan_y_anim_enabled"] = tk.BooleanVar(value=False); self.anim_params["pan_y_anim_amp"] = tk.DoubleVar(value=0.1); self.anim_params["pan_y_anim_period"] = tk.IntVar(value=60)
        pan_y_lfo_frame = ttk.Frame(view_controls_frame); pan_y_lfo_frame.grid(row=vc_row, column=0, columnspan=4, pady=2, sticky=tk.W); vc_row+=1; ttk.Checkbutton(pan_y_lfo_frame, text="Animate Pan Y", variable=self.anim_vars["pan_y_anim_enabled"]).pack(side=tk.LEFT); ttk.Label(pan_y_lfo_frame, text="Amp:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_y_lfo_frame, textvariable=self.anim_params["pan_y_anim_amp"], width=5).pack(side=tk.LEFT); ttk.Label(pan_y_lfo_frame, text="Per:").pack(side=tk.LEFT, padx=(5,0)); ttk.Entry(pan_y_lfo_frame, textvariable=self.anim_params["pan_y_anim_period"], width=5).pack(side=tk.LEFT)
        
        post_pan_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="Post-Pan Animation", padding="10"); post_pan_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr += 1; ttk.Checkbutton(post_pan_frame, text="Enable Drift", variable=self.post_pan_anim_enabled_var).grid(row=0, column=0, sticky=tk.W); ttk.Label(post_pan_frame, text="Steps:").grid(row=0, column=1, sticky=tk.E, padx=(10,0)); ttk.Entry(post_pan_frame, textvariable=self.post_pan_drift_steps_var, width=5).grid(row=0, column=2, sticky=tk.W); ttk.Label(post_pan_frame, text="Delay(ms):").grid(row=1, column=1, sticky=tk.E, padx=(10,0)); ttk.Entry(post_pan_frame, textvariable=self.post_pan_drift_delay_var, width=5).grid(row=1, column=2, sticky=tk.W); ttk.Label(post_pan_frame, text="Amount:").grid(row=0, column=3, sticky=tk.E, padx=(10,0)); ttk.Scale(post_pan_frame, variable=self.post_pan_drift_amount_var, from_=0.001, to=0.05, orient=tk.HORIZONTAL, length=80).grid(row=0, column=4, sticky=tk.W); ttk.Label(post_pan_frame, textvariable=self.post_pan_drift_amount_var, width=6).grid(row=0, column=5, sticky=tk.W)
        
        feedback_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="Feedback Options", padding="10"); feedback_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr +=1; self.op_vars["blend_original_enabled"] = tk.BooleanVar(value=False); self.op_params["blend_alpha_value"] = tk.DoubleVar(value=0.01); blend_alpha_display_name = "Blend Original Alpha"; self.param_names_for_mouse.append(blend_alpha_display_name); ttk.Checkbutton(feedback_frame, text="Blend with Original", variable=self.op_vars["blend_original_enabled"], command=self.schedule_interactive_update).grid(row=0, column=0, sticky=tk.W); ttk.Label(feedback_frame, text="Alpha:").grid(row=0, column=1, sticky=tk.E, padx=2); ttk.Scale(feedback_frame, variable=self.op_params["blend_alpha_value"], from_=0.0, to=0.2, orient=tk.HORIZONTAL, length=100, command=self.schedule_interactive_update).grid(row=0, column=2, sticky=tk.W); ttk.Label(feedback_frame,textvariable=self.op_params["blend_alpha_value"], width=4).grid(row=0, column=3, sticky=tk.W)
        
        mouse_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="Mouse Interaction", padding="10"); mouse_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr += 1; ttk.Checkbutton(mouse_frame, text="Enable Mouse Control", variable=self.mouse_control_enabled, command=self.schedule_interactive_update).grid(row=0, column=0, columnspan=2, sticky=tk.W); ttk.Label(mouse_frame, text="Mouse X controls:").grid(row=1, column=0, sticky=tk.W); self.mouse_x_combo = ttk.Combobox(mouse_frame, textvariable=self.mouse_x_param_control, values=self.param_names_for_mouse, state="readonly", width=22, postcommand=self.update_mouse_param_options); self.mouse_x_combo.grid(row=1, column=1, sticky="ew"); self.mouse_x_combo.bind("<<ComboboxSelected>>", self.schedule_interactive_update); ttk.Label(mouse_frame, text="Mouse Y controls:").grid(row=2, column=0, sticky=tk.W); self.mouse_y_combo = ttk.Combobox(mouse_frame, textvariable=self.mouse_y_param_control, values=self.param_names_for_mouse, state="readonly", width=22, postcommand=self.update_mouse_param_options); self.mouse_y_combo.grid(row=2, column=1, sticky="ew"); self.mouse_y_combo.bind("<<ComboboxSelected>>", self.schedule_interactive_update)
        
        output_options_frame = ttk.LabelFrame(self.controls_scrollable_frame, text="Output Options", padding="10"); output_options_frame.grid(row=csr, column=0, columnspan=4, sticky="ew", padx=5, pady=10); csr += 1; ttk.Label(output_options_frame, text="Default Filename:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3); self.entries["default_filename"] = tk.StringVar(value="evolved_art.png"); ttk.Entry(output_options_frame, textvariable=self.entries["default_filename"], width=30).grid(row=0, column=1, columnspan=3, sticky="we", padx=5, pady=3); ttk.Checkbutton(output_options_frame, text="Save All Frames (Multi-Step)", variable=self.save_animation_frames_var).grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=3)
        
        buttons_frame = ttk.Frame(self.controls_scrollable_frame); buttons_frame.grid(row=csr, column=0, columnspan=4, pady=15, sticky="ew"); csr +=1; buttons_frame.columnconfigure(0, weight=1); buttons_frame.columnconfigure(1, weight=1); buttons_frame.columnconfigure(2, weight=1); buttons_frame.columnconfigure(3, weight=1); self.evolve_button = ttk.Button(buttons_frame, text="Multi-Step Evolve", command=self.trigger_multistep_evolution); self.evolve_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew"); self.hold_evolve_button = ttk.Button(buttons_frame, text="Hold to Evolve"); self.hold_evolve_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew"); self.hold_evolve_button.bind("<ButtonPress-1>", self.on_hold_evolve_press); self.hold_evolve_button.bind("<ButtonRelease-1>", self.on_hold_evolve_release); self.reset_button = ttk.Button(buttons_frame, text="Reset Image", command=self.reset_image_to_original); self.reset_button.grid(row=0, column=2, padx=2, pady=2, sticky="ew"); self.save_button = ttk.Button(buttons_frame, text="Save Image", command=self.save_image_as); self.save_button.grid(row=0, column=3, padx=2, pady=2, sticky="ew")
        
        self.status_label = ttk.Label(self.controls_scrollable_frame, text="Load an image. Adjust parameters for live preview.", wraplength=380); self.status_label.grid(row=csr, column=0, columnspan=4, pady=5, sticky=tk.W); csr+=1
        
        self.image_display_label = ttk.Label(main_frame, relief="sunken", anchor="center", background="#2B2B2B")
        self.image_display_label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        main_frame.columnconfigure(2, weight=1); main_frame.rowconfigure(0, weight=1)
        
        self.image_display_label.bind("<Motion>", self.on_mouse_move_over_image)
        self.image_display_label.bind("<Leave>", self.on_mouse_leave_image)
        self.image_display_label.bind("<ButtonPress-1>", self.on_pan_start)
        self.image_display_label.bind("<B1-Motion>", self.on_pan_drag)
        self.image_display_label.bind("<ButtonRelease-1>", self.on_pan_end)

        self.root.after(200, self._capture_initial_display_size)

    def _capture_initial_display_size(self):
        self.root.update_idletasks();label_w=self.image_display_label.winfo_width();label_h=self.image_display_label.winfo_height()
        if label_w > 50 and label_h > 50: self.target_display_width=label_w-10;self.target_display_height=label_h-10

    def update_mouse_param_options(self):
        self.mouse_x_combo['values']=self.param_names_for_mouse
        self.mouse_y_combo['values']=self.param_names_for_mouse

    def schedule_interactive_update(self, event=None): # DEFINED
        if self.is_post_pan_anim_running: return 
        if not self.input_image_loaded: return
        if self.after_id_preview: self.root.after_cancel(self.after_id_preview)
        self.after_id_preview = self.root.after(self.interactive_update_delay, self._perform_interactive_update) 

    def _perform_interactive_update(self): # DEFINED
        if not self.input_image_loaded: return
        base_for_roi = self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
        if not base_for_roi : return 
        try:
            output_w = int(self.entries["output_width"].get()); output_h = int(self.entries["output_height"].get())
            if output_w <=0 or output_h <=0: return
            image_to_process = self._get_image_roi_at_output_resolution(base_for_roi, output_w, output_h, step_num_for_multistep=0)
            if not image_to_process: return
            evolved_one_step = self._apply_evolution_pipeline_once(image_to_process, step_num_for_multistep=0) 
            self.current_evolving_image = evolved_one_step 
            self.display_image(self.current_evolving_image)
            self.status_label.config(text="Preview updated.")
        except Exception as e: self.status_label.config(text=f"Preview Error: {e}"); traceback.print_exc()

    def _get_image_roi_at_output_resolution(self, source_image, target_w, target_h, step_num_for_multistep=0):
        if not source_image: return None
        src_w, src_h = source_image.size
        current_center_x_norm = self.view_center_x_norm.get()
        current_center_y_norm = self.view_center_y_norm.get()
        if step_num_for_multistep > 0: 
            if self.anim_vars.get("pan_x_anim_enabled", tk.BooleanVar(value=False)).get():
                amp = self.anim_params.get("pan_x_anim_amp", tk.DoubleVar(value=0.1)).get()
                period = self.anim_params.get("pan_x_anim_period", tk.IntVar(value=50)).get()
                if period > 0: current_center_x_norm = max(0.0, min(1.0, self.view_center_x_norm.get() + amp * math.sin(2 * math.pi * step_num_for_multistep / period)))
            if self.anim_vars.get("pan_y_anim_enabled", tk.BooleanVar(value=False)).get():
                amp = self.anim_params.get("pan_y_anim_amp", tk.DoubleVar(value=0.1)).get()
                period = self.anim_params.get("pan_y_anim_period", tk.IntVar(value=60)).get()
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
        except Exception as e: print(f"Symmetry Error ('{symmetry_type}'): {e}"); traceback.print_exc(); return image_in 
        return img_to_sym

    def _apply_evolution_pipeline_once(self, image_in, step_num_for_multistep=0):
        current_img = image_in.copy(); output_w, output_h = current_img.size; active_params = {}
        for op_key_cfg, op_config in self.operations_config.items():
            for param_key_cfg, p_config in op_config["params"].items():
                var_key = p_config["var_key"]; base_val = self.op_params[var_key].get(); final_val = base_val
                anim_enabled_key = f"{var_key}_anim_enabled"
                if step_num_for_multistep > 0 and anim_enabled_key in self.anim_vars and self.anim_vars[anim_enabled_key].get():
                    amp_key = f"{var_key}_anim_amp"; period_key = f"{var_key}_anim_period"
                    amplitude = self.anim_params[amp_key].get(); period = self.anim_params[period_key].get()
                    if period > 0: final_val = base_val + amplitude * math.sin(2 * math.pi * step_num_for_multistep / period); final_val = max(p_config["min"], min(p_config["max"], final_val))
                if self.mouse_control_enabled.get():
                    disp_name = op_config["label"]+" "+p_config["label"].replace(":",""); m_x,m_y=self.mouse_x_norm.get(),self.mouse_y_norm.get(); p_min,p_max=p_config["min"],p_config["max"]
                    if self.mouse_x_param_control.get()==disp_name: final_val = p_min+(p_max-p_min)*m_x
                    elif self.mouse_y_param_control.get()==disp_name: final_val = p_min+(p_max-p_min)*m_y
                active_params[var_key] = int(final_val) if p_config.get("is_int") else final_val
        
        blend_alpha_val = self.op_params["blend_alpha_value"].get()
        if self.mouse_control_enabled.get():
            b_min,b_max=0.0,0.2
            if self.mouse_x_param_control.get()=="Blend Original Alpha": blend_alpha_val=b_min+(b_max-b_min)*self.mouse_x_norm.get()
            elif self.mouse_y_param_control.get()=="Blend Original Alpha": blend_alpha_val=b_min+(b_max-b_min)*self.mouse_y_norm.get()

        if self.op_vars["blur_enabled"].get(): current_img = current_img.filter(ImageFilter.GaussianBlur(radius=active_params["blur_radius"]))
        if self.op_vars.get("unsharp_enabled").get(): current_img = current_img.filter(ImageFilter.UnsharpMask(radius=active_params["unsharp_radius"], percent=active_params["unsharp_percent"], threshold=active_params["unsharp_threshold"]))
        if self.op_vars.get("edge_blend_enabled").get(): edges = current_img.filter(ImageFilter.FIND_EDGES).convert("RGB"); current_img = Image.blend(current_img, edges, alpha=active_params["edge_blend_alpha"])
        if self.op_vars.get("pixelate_enabled").get():
            bs=max(2,active_params["pixelate_block_size"]); w,h=current_img.size
            if w//bs>0 and h//bs>0: tmp=current_img.resize((w//bs,h//bs),Image.Resampling.NEAREST); current_img=tmp.resize((w,h),Image.Resampling.NEAREST)
        if self.op_vars.get("chanshift_enabled").get():
            r,g,b=current_img.split(); r_s=ImageChops.offset(r,active_params["chanshift_rx"],active_params["chanshift_ry"]); b_s=ImageChops.offset(b,active_params["chanshift_bx"],active_params["chanshift_by"]); current_img=Image.merge("RGB",(r_s,g,b_s))
        if self.op_vars.get("shear_enabled").get(): sx=active_params.get("shear_x_factor",0.0); matrix=(1,sx,0,0,1,0); current_img=current_img.transform(current_img.size,Image.AFFINE,matrix,Image.Resampling.BICUBIC,fillcolor=(50,50,50))
        
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
        current_img = self._apply_symmetry(current_img) # Apply symmetry last
        return current_img

    def on_mouse_move_over_image(self, event):
        if self.panning_active: self.on_pan_drag(event); return
        if not (self.image_display_label.winfo_width() > 1 and self.image_display_label.winfo_height() > 1): return
        x = max(0.0, min(1.0, event.x / self.image_display_label.winfo_width())); y = max(0.0, min(1.0, event.y / self.image_display_label.winfo_height()))
        self.mouse_x_norm.set(x); self.mouse_y_norm.set(y)
        if self.mouse_control_enabled.get() and (self.mouse_x_param_control.get() != "None" or self.mouse_y_param_control.get() != "None"):
            self.schedule_interactive_update()
    def on_mouse_leave_image(self, event): pass  
    def on_pan_start(self, event):
        self._cancel_post_pan_animation() 
        if not self.input_image_loaded: return
        if self.mouse_control_enabled.get() and (self.mouse_x_param_control.get() != "None" or self.mouse_y_param_control.get() != "None"): return
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

    def load_input_image(self): # DEFINED
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

    def load_displacement_map_image(self): # DEFINED
        fpath = filedialog.askopenfilename(
            title="Select Displacement Map Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff"), ("All files", "*.*")],
            parent=self.root
        )
        if fpath:
            try:
                img = Image.open(fpath)
                self.displacement_map_image = img # Keep original mode for flexibility in displacement logic
                self.displacement_map_filename_var.set(fpath.split('/')[-1])
                self.status_label.config(text=f"DMap: {self.displacement_map_filename_var.get()}")
                self.schedule_interactive_update() 
            except Exception as e:
                messagebox.showerror("Map Load Error", f"Failed to load DMap: {e}", parent=self.root)
                self.displacement_map_image = None; self.displacement_map_filename_var.set("No map loaded.")
                traceback.print_exc()

    def zoom_in_view(self): self._cancel_post_pan_animation(); self.zoom_factor.set(min(self.zoom_factor.get() * self.zoom_increment, 20.0)); self.schedule_interactive_update()
    def zoom_out_view(self): self._cancel_post_pan_animation(); self.zoom_factor.set(max(self.zoom_factor.get() / self.zoom_increment, 0.05)); self.schedule_interactive_update()
    def reset_view(self): 
        self._cancel_post_pan_animation()
        self.zoom_factor.set(1.0); self.view_center_x_norm.set(0.5); self.view_center_y_norm.set(0.5); 
        self.schedule_interactive_update()
    def pan_view(self, dx_factor, dy_factor): # For button pan
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

    def trigger_multistep_evolution(self):
        self._cancel_post_pan_animation()
        if not self.input_image_loaded: messagebox.showerror("Error", "Load input image first.", parent=self.root); return
        try:
            num_steps = int(self.entries["steps"].get())
            output_w = int(self.entries["output_width"].get()); output_h = int(self.entries["output_height"].get())
            if num_steps <= 0 or output_w <= 0 or output_h <= 0: messagebox.showerror("Input Error", "Steps, W, H must be > 0.", parent=self.root); return
            self.status_label.config(text=f"Multi-step Evolving ({num_steps} steps)..."); self.root.update_idletasks()
            current_processing_img = self._get_image_roi_at_output_resolution(self.current_evolving_image if self.current_evolving_image else self.input_image_loaded, output_w, output_h, step_num_for_multistep=0)
            if not current_processing_img: self.status_label.config(text="Error: Could not prepare ROI."); return
            frames_to_save = []
            save_frames_enabled = self.save_animation_frames_var.get()
            for step in range(num_steps):
                base_for_this_step_roi = current_processing_img 
                roi_content_for_this_step = self._get_image_roi_at_output_resolution(base_for_this_step_roi, output_w, output_h, step_num_for_multistep=step + 1)
                if not roi_content_for_this_step: break
                current_processing_img = self._apply_evolution_pipeline_once(roi_content_for_this_step, step_num_for_multistep=step + 1)
                if save_frames_enabled: frames_to_save.append(current_processing_img.copy())
                if (step + 1) % 1 == 0 or step == num_steps - 1:
                    self.current_evolving_image = current_processing_img.copy()
                    self.display_image(self.current_evolving_image)
                    self.status_label.config(text=f"Multi-Step Evolution: {step + 1}/{num_steps}")
                    self.root.update_idletasks()
            self.status_label.config(text=f"Multi-step evolution complete ({num_steps} steps).")
            if save_frames_enabled and frames_to_save: self.save_animation_frame_sequence(frames_to_save)
        except ValueError as ve: messagebox.showerror("Input Error", f"{ve}", parent=self.root); self.status_label.config(text=f"Input Error: {ve}")
        except Exception as e: messagebox.showerror("Evo Error", f"{e}", parent=self.root); self.status_label.config(text=f"Evo Error: {e}"); traceback.print_exc()

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
        if not pil_img_to_display: self.image_display_label.config(image=''); self.photo_image = None; return
        try:
            mdw = self.target_display_width if self.target_display_width > 0 else 300
            mdh = self.target_display_height if self.target_display_height > 0 else 300
            img_copy = pil_img_to_display.copy(); img_copy.thumbnail((mdw, mdh), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img_copy); self.image_display_label.config(image=self.photo_image)
        except Exception as e: self.status_label.config(text=f"Display Error: {e}"); traceback.print_exc()

    def save_image_as(self):
        if not self.current_evolving_image: messagebox.showwarning("No Image", "Evolve an image first.", parent=self.root); return
        try:
            sugg_fname = self.entries["default_filename"].get(); base_s, orig_e = os.path.splitext(sugg_fname)
            if not orig_e: orig_e = ".png"
            rand_s = f"{random.randint(0,9999):04d}"; init_file = f"{base_s}_{rand_s}{orig_e}"
            fpath = filedialog.asksaveasfilename(initialfile=init_file, defaultextension=orig_e, filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("BMP","*.bmp"),("All","*.*")], parent=self.root)
            if fpath: self.current_evolving_image.save(fpath); self.status_label.config(text=f"Saved: {fpath}"); messagebox.showinfo("Success", f"Saved to:\n{fpath}", parent=self.root)
            else: self.status_label.config(text="Save cancelled.")
        except Exception as e: messagebox.showerror("Save Error", f"{e}", parent=self.root); self.status_label.config(text=f"Save Error: {e}"); traceback.print_exc()

    def reset_image_to_original(self):
        self._cancel_post_pan_animation()
        if not self.input_image_loaded: messagebox.showwarning("No Image", "No image loaded.", parent=self.root); return
        try:self.current_evolving_image=None;self.reset_view();self.status_label.config(text="Image & view reset.")
        except Exception as e:messagebox.showerror("Reset Error",f"{e}",parent=self.root);self.status_label.config(text=f"Reset Error:{e}");traceback.print_exc()

    def on_hold_evolve_press(self,event):
        self._cancel_post_pan_animation()
        if not self.input_image_loaded:messagebox.showerror("Error","Load image first.",parent=self.root);return
        self.hold_evolve_active=True;self.status_label.config(text="Continuously evolving...");self.continuous_evolve_step()
    def on_hold_evolve_release(self,event):
        self.hold_evolve_active=False
        if self.hold_evolve_after_id:self.root.after_cancel(self.hold_evolve_after_id);self.hold_evolve_after_id=None
        if self.status_label.cget("text").startswith("Continuously"):self.status_label.config(text="Continuous evolution stopped.")
    def continuous_evolve_step(self):
        if not self.hold_evolve_active or not self.input_image_loaded:self.hold_evolve_active=False;return
        base_img=self.current_evolving_image if self.current_evolving_image else self.input_image_loaded
        if not base_img:self.hold_evolve_active=False;return
        try:
            ow=int(self.entries["output_width"].get());oh=int(self.entries["output_height"].get())
            if ow<=0 or oh<=0:self.hold_evolve_active=False;return
            img_to_proc=self._get_image_roi_at_output_resolution(base_img,ow,oh, step_num_for_multistep=0) 
            if not img_to_proc:self.hold_evolve_active=False;return
            evolved_step=self._apply_evolution_pipeline_once(img_to_proc,step_num_for_multistep=0) 
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
        self.status_label.config(text="Post-pan drift...")
        self._run_post_pan_animation_step()

    def _run_post_pan_animation_step(self):
        if not self.is_post_pan_anim_running or not self.input_image_loaded: self.is_post_pan_anim_running = False; return
        max_steps = self.post_pan_drift_steps_var.get()
        if self.post_pan_anim_current_step >= max_steps:
            self.is_post_pan_anim_running = False; self.status_label.config(text="Post-pan drift finished."); 
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
