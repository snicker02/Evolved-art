import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from kornia.filters import gaussian_blur2d
from kornia.color import hsv_to_rgb, rgb_to_hsv
import kornia.geometry.transform as K
import numpy as np
import math
import random
import noise
from scipy.spatial import cKDTree
from scipy.signal import convolve2d

def pil_to_tensor(pil_image, device):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    numpy_array = np.array(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(numpy_array).permute(2, 0, 1).unsqueeze(0).to(device)

class Effect:
    """Base class for all effects."""
    name = "Base Effect"
    
    def __init__(self, app):
        self.app = app
        self.device = app.gpu_evolver.device if app.gpu_evolver else 'cpu'

    def apply(self, tensor, params):
        """Applies the effect. Must be overridden by subclasses."""
        raise NotImplementedError

class BlurEffect(Effect):
    name = "Blur"
    def apply(self, tensor, params):
        radius = params.get("blur_radius", 0.0)
        if radius > 0:
            kernel_size = int(radius * 2) * 2 + 1
            return gaussian_blur2d(tensor, kernel_size=(kernel_size, kernel_size), sigma=(radius, radius))
        return tensor

class UnsharpMaskEffect(Effect):
    name = "Unsharp Mask"
    def apply(self, tensor, params):
        radius = params.get("unsharp_radius", 2)
        kernel_size = int(radius) * 2 + 1
        amount = params.get("unsharp_percent", 150) / 100.0
        blurred_tensor = gaussian_blur2d(tensor, kernel_size=(kernel_size, kernel_size), sigma=(radius, radius))
        high_pass = tensor - blurred_tensor
        sharpened_tensor = tensor + high_pass * amount
        return sharpened_tensor

class SharpenEffect(Effect):
    name = "Sharpen (Convolution)"
    def __init__(self, app):
        super().__init__(app)
        kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=torch.float32)
        self.kernel = kernel.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)

    def apply(self, tensor, params):
        strength = params.get("sharpen_strength", 0.5)
        if strength == 0: return tensor
        sharpened_tensor = F.conv2d(tensor, self.kernel, padding='same', groups=3)
        return torch.lerp(tensor, sharpened_tensor, strength)

class PixelateEffect(Effect):
    name = "Pixelate"
    def apply(self, tensor, params):
        B, C, H, W = tensor.shape
        block_size = int(params.get("pixelate_block_size", 8))
        if block_size > 1:
            small = F.interpolate(tensor, scale_factor=1/block_size, mode='nearest')
            return F.interpolate(small, size=(H, W), mode='nearest')
        return tensor

class SwirlEffect(Effect):
    name = "Swirl/Whirl"
    def apply(self, tensor, params):
        strength = params.get("swirl_strength", 0.0)
        if strength == 0: return tensor
        B, C, H, W = tensor.shape
        center_x = 0.5; center_y = 0.5
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        dx = grid_x - (2 * center_x - 1); dy = grid_y - (2 * center_y - 1)
        radius = torch.sqrt(dx*dx + dy*dy)
        angle = torch.atan2(dy, dx)
        swirl_angle = strength * radius
        new_angle = angle + swirl_angle
        new_x = radius * torch.cos(new_angle) + (2 * center_x - 1)
        new_y = radius * torch.sin(new_angle) + (2 * center_y - 1)
        sample_grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        return F.grid_sample(tensor, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)

class ChromaticAberrationEffect(Effect):
    name = "Chromatic Aberration"
    def apply(self, tensor, params):
        red_offset = params.get("ca_red_offset", 0.0); red_angle_deg = params.get("ca_red_angle", 0.0)
        blue_offset = params.get("ca_blue_offset", 0.0); blue_angle_deg = params.get("ca_blue_angle", 0.0)
        if red_offset == 0 and blue_offset == 0: return tensor
        B, C, H, W = tensor.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        if red_offset != 0:
            red_angle_rad = math.radians(red_angle_deg)
            offset_x_r = (red_offset / (W / 2)) * math.cos(red_angle_rad); offset_y_r = (red_offset / (H / 2)) * math.sin(red_angle_rad)
            grid_r = base_grid.clone(); grid_r[..., 0] += offset_x_r; grid_r[..., 1] += offset_y_r
            r_channel = F.grid_sample(tensor[:, 0:1, :, :], grid_r, mode='bilinear', padding_mode='border', align_corners=False)
        else: r_channel = tensor[:, 0:1, :, :]
        g_channel = tensor[:, 1:2, :, :]
        if blue_offset != 0:
            blue_angle_rad = math.radians(blue_angle_deg)
            offset_x_b = (blue_offset / (W / 2)) * math.cos(blue_angle_rad); offset_y_b = (blue_offset / (H / 2)) * math.sin(blue_angle_rad)
            grid_b = base_grid.clone(); grid_b[..., 0] += offset_x_b; grid_b[..., 1] += offset_y_b
            b_channel = F.grid_sample(tensor[:, 2:3, :, :], grid_b, mode='bilinear', padding_mode='border', align_corners=False)
        else: b_channel = tensor[:, 2:3, :, :]
        return torch.cat([r_channel, g_channel, b_channel], dim=1)

class PerspectiveTransformEffect(Effect):
    name = "Perspective Transform"
    def apply(self, tensor, params):
        tl_x = params.get("pt_tl_x", 0.0); tl_y = params.get("pt_tl_y", 0.0)
        tr_x = params.get("pt_tr_x", 1.0); tr_y = params.get("pt_tr_y", 0.0)
        br_x = params.get("pt_br_x", 1.0); br_y = params.get("pt_br_y", 1.0)
        bl_x = params.get("pt_bl_x", 0.0); bl_y = params.get("pt_bl_y", 1.0)
        tiled = params.get("pt_tiled", True)
        B, C, H, W = tensor.shape
        src_points = torch.tensor([[[tl_x*W, tl_y*H], [tr_x*W, tr_y*H], [br_x*W, br_y*H], [bl_x*W, bl_y*H]]], device=self.device)
        dst_points = torch.tensor([[[0., 0.], [W - 1., 0.], [W - 1., H - 1.], [0., H - 1.]]], device=self.device)
        if torch.equal(src_points, dst_points): return tensor
        try:
            perspective_transform = K.get_perspective_transform(src_points, dst_points)
            padding_mode = 'border' if not tiled else 'zeros'
            warped_tensor = K.warp_perspective(tensor, perspective_transform, (H, W), align_corners=False, padding_mode=padding_mode)
            if tiled:
                grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
                base_grid = torch.stack((grid_x, grid_y), 2)
                flat_grid = base_grid.view(-1, 2)
                homo_grid = torch.cat([flat_grid, torch.ones(H*W, 1, device=self.device)], dim=1)
                inv_transform = torch.inverse(perspective_transform.squeeze(0))
                transformed_grid = torch.matmul(homo_grid, inv_transform.t())
                transformed_grid = transformed_grid[:, :2] / transformed_grid[:, 2:]
                wrapped_grid = (transformed_grid.view(1, H, W, 2) + 1) % 2 - 1
                return F.grid_sample(tensor, wrapped_grid, mode='bilinear', padding_mode='border', align_corners=False)
            return warped_tensor
        except Exception: return tensor

class ChannelShiftEffect(Effect):
    name = "Channel Shift"
    def apply(self, tensor, params):
        rx = int(params.get("chanshift_rx", 0)); ry = int(params.get("chanshift_ry", 0))
        bx = int(params.get("chanshift_bx", 0)); by = int(params.get("chanshift_by", 0))
        r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
        r_rolled = torch.roll(r, shifts=(ry, rx), dims=(2, 3))
        b_rolled = torch.roll(b, shifts=(by, bx), dims=(2, 3))
        return torch.cat([r_rolled, g, b_rolled], dim=1)

class DisplaceMapEffect(Effect):
    name = "Displacement Map"
    def apply(self, tensor, params):
        dmap_pil = self.app.displacement_map_image
        if not dmap_pil: return tensor
        B, C, H, W = tensor.shape
        dmap_tensor = pil_to_tensor(dmap_pil, self.device)
        dmap_resized = F.interpolate(dmap_tensor, size=(H, W), mode='bilinear', align_corners=False)
        dmap_lum = torch.mean(dmap_resized, dim=1, keepdim=True)
        norm_map = (dmap_lum - 0.5) * 2.0
        x_scale = params.get("displace_x_scale", 10.0) / (W / 2)
        y_scale = params.get("displace_y_scale", 10.0) / (H / 2)
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)
        offset_grid = base_grid.clone()
        displacement = norm_map.squeeze(1)
        offset_grid[:, :, :, 0] += displacement * x_scale
        offset_grid[:, :, :, 1] += displacement * y_scale
        return F.grid_sample(tensor, offset_grid, mode='bilinear', padding_mode='border', align_corners=False)

class WaveDisplaceEffect(Effect):
    name = "Wave Displacement"
    def _generate_wave(self, coords, wave_type, freq):
        t = coords * freq * 0.5
        if wave_type == "Sine":
            return torch.sin(t * torch.pi)
        elif wave_type == "Triangle":
            return 2 * torch.abs(torch.fmod(t, 2) - 1) - 1
        elif wave_type == "Sawtooth":
            return torch.fmod(t, 2) - 1
        return torch.zeros_like(coords)
    def apply(self, tensor, params):
        amp_x = params.get("wave_amp_x", 0.0); freq_x = params.get("wave_freq_x", 4.0)
        amp_y = params.get("wave_amp_y", 0.0); freq_y = params.get("wave_freq_y", 4.0)
        wave_type_x = params.get("wave_type_x", "Sine")
        wave_type_y = params.get("wave_type_y", "Sine")
        if amp_x == 0 and amp_y == 0: return tensor
        B, C, H, W = tensor.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        offset_x = (amp_x / W) * self._generate_wave(grid_y, wave_type_x, freq_x)
        offset_y = (amp_y / H) * self._generate_wave(grid_x, wave_type_y, freq_y)
        grid = torch.stack((grid_x + offset_x, grid_y + offset_y), dim=-1).unsqueeze(0)
        return F.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)

class PixelJitterEffect(Effect):
    name = "Pixel Jitter"
    def apply(self, tensor, params):
        strength = params.get("jitter_strength", 0.0)
        if strength == 0: return tensor
        B, C, H, W = tensor.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)
        noise = (torch.rand_like(base_grid) - 0.5) * 2 * (strength / W)
        jitter_grid = base_grid + noise
        return F.grid_sample(tensor, jitter_grid, mode='bilinear', padding_mode='border', align_corners=False)

class SphericalDistortEffect(Effect):
    name = "Spherical Distortion"
    def apply(self, tensor, params):
        strength = params.get("sphere_strength", 1.0)
        zoom = params.get("sphere_zoom", 1.0)
        if strength == 1.0 and zoom == 1.0: return tensor
        B, C, H, W = tensor.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        radius = torch.sqrt(grid_x**2 + grid_y**2)
        theta = torch.atan2(grid_y, grid_x)
        radius = torch.where(radius == 0, torch.tensor(1e-6, device=self.device), radius)
        new_radius = radius ** (1.0 / strength)
        new_x = (1 / zoom) * new_radius * torch.cos(theta)
        new_y = (1 / zoom) * new_radius * torch.sin(theta)
        new_x = torch.clamp(new_x, -1.0, 1.0)
        new_y = torch.clamp(new_y, -1.0, 1.0)
        sample_grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        return F.grid_sample(tensor, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)

class TurbulenceEffect(Effect):
    name = "Turbulence"
    def apply(self, tensor, params):
        strength = params.get("turbulence_strength", 10.0)
        scale = params.get("turbulence_scale", 50.0)
        octaves = params.get("turbulence_octaves", 4)
        if strength == 0 or scale == 0: return tensor
        
        B, C, H, W = tensor.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        x_coords = np.arange(W); y_coords = np.arange(H)
        xx, yy = np.meshgrid(x_coords, y_coords)

        vec_pnoise2 = np.vectorize(noise.pnoise2)

        noise_x_np = vec_pnoise2(xx / scale, yy / scale, octaves=octaves, persistence=0.5, lacunarity=2.0, base=0)
        noise_y_np = vec_pnoise2(xx / scale, yy / scale, octaves=octaves, persistence=0.5, lacunarity=2.0, base=100)
        
        noise_x = torch.from_numpy(noise_x_np).float().to(self.device)
        noise_y = torch.from_numpy(noise_y_np).float().to(self.device)

        offset_x = (strength / W) * noise_x
        offset_y = (strength / H) * noise_y
        
        distort_grid = base_grid.clone()
        distort_grid[..., 0] += offset_x
        distort_grid[..., 1] += offset_y
        
        return F.grid_sample(tensor, distort_grid, mode='bilinear', padding_mode='border', align_corners=False)

class VoronoiNoiseEffect(Effect):
    name = "Voronoi Noise"
    def apply(self, tensor, params):
        num_points = params.get("voronoi_points", 50)
        strength = params.get("voronoi_strength", 10.0)
        if strength == 0: return tensor

        B, C, H, W = tensor.shape
        points = np.random.rand(num_points, 2) * np.array([W, H])
        x = np.arange(W); y = np.arange(H)
        xx, yy = np.meshgrid(x, y)
        pixel_coords = np.c_[xx.ravel(), yy.ravel()]
        
        tree = cKDTree(points)
        distances, indices = tree.query(pixel_coords)
        
        dist_map = distances.reshape(H, W)
        norm_dist_map = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min())
        
        displacement = torch.from_numpy(norm_dist_map).float().to(self.device) * (strength / W)
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        distort_grid = base_grid.clone()
        distort_grid[..., 0] += displacement
        distort_grid[..., 1] += displacement
        
        return F.grid_sample(tensor, distort_grid, mode='bilinear', padding_mode='border', align_corners=False)

class JuliaDistortEffect(Effect):
    name = "Julia Distortion"
    def apply(self, tensor, params):
        c_real = params.get("julia_c_real", -0.8)
        c_imag = params.get("julia_c_imag", 0.156)
        iterations = params.get("julia_iterations", 20)
        strength = params.get("julia_strength", 10.0)

        if strength == 0: return tensor

        B, C, H, W = tensor.shape
        
        y, x = torch.meshgrid(torch.linspace(-1.5, 1.5, H, device=self.device), torch.linspace(-1.5, 1.5, W, device=self.device), indexing='ij')
        z = x + y * 1j
        c = torch.complex(torch.tensor(c_real), torch.tensor(c_imag)).to(self.device)
        
        iterations_map = torch.zeros_like(x)
        for i in range(iterations):
            mask = torch.abs(z) < 2.0
            z[mask] = z[mask]**2 + c
            iterations_map[mask] += 1
            
        norm_map = (iterations_map / iterations) * 2.0 - 1.0
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        offset_grid = base_grid.clone()
        offset_grid[..., 0] += (strength / W) * norm_map
        offset_grid[..., 1] += (strength / H) * norm_map
        
        return F.grid_sample(tensor, offset_grid, mode='bilinear', padding_mode='border', align_corners=False)

class MandelbrotDistortEffect(Effect):
    name = "Mandelbrot Distortion"
    def apply(self, tensor, params):
        iterations = params.get("mandel_iterations", 20)
        strength = params.get("mandel_strength", 10.0)

        if strength == 0: return tensor

        B, C, H, W = tensor.shape
        
        y, x = torch.meshgrid(torch.linspace(-1.5, 1.5, H, device=self.device), torch.linspace(-2.0, 1.0, W, device=self.device), indexing='ij')
        c = x + y * 1j
        z = torch.zeros_like(c)
        
        iterations_map = torch.zeros_like(x)
        for i in range(iterations):
            mask = torch.abs(z) < 2.0
            z[mask] = z[mask]**2 + c[mask]
            iterations_map[mask] += 1
            
        norm_map = (iterations_map / iterations) * 2.0 - 1.0
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        offset_grid = base_grid.clone()
        offset_grid[..., 0] += (strength / W) * norm_map
        offset_grid[..., 1] += (strength / H) * norm_map
        
        return F.grid_sample(tensor, offset_grid, mode='bilinear', padding_mode='border', align_corners=False)

class BurningShipDistortEffect(Effect):
    name = "Burning Ship Distortion"
    def apply(self, tensor, params):
        iterations = params.get("ship_iterations", 20)
        strength = params.get("ship_strength", 10.0)

        if strength == 0: return tensor

        B, C, H, W = tensor.shape
        
        y, x = torch.meshgrid(torch.linspace(-2.0, 1.0, H, device=self.device), torch.linspace(-2.0, 1.0, W, device=self.device), indexing='ij')
        c = x + y * 1j
        z = torch.zeros_like(c)
        
        iterations_map = torch.zeros_like(x)
        for i in range(iterations):
            mask = torch.abs(z) < 2.0
            z_abs = torch.complex(torch.abs(z.real), torch.abs(z.imag))
            z[mask] = (z_abs[mask]**2) + c[mask]
            iterations_map[mask] += 1
            
        norm_map = (iterations_map / iterations) * 2.0 - 1.0
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        offset_grid = base_grid.clone()
        offset_grid[..., 0] += (strength / W) * norm_map
        offset_grid[..., 1] += (strength / H) * norm_map
        
        return F.grid_sample(tensor, offset_grid, mode='bilinear', padding_mode='border', align_corners=False)

class ReactionDiffusionEffect(Effect):
    name = "Reaction-Diffusion"
    def __init__(self, app):
        super().__init__(app)
        self.laplacian_kernel = np.array([[0.05, 0.2, 0.05],
                                          [0.2, -1, 0.2],
                                          [0.05, 0.2, 0.05]])
    
    def apply(self, tensor, params):
        B, C, H, W = tensor.shape
        
        if not hasattr(self.app, 'rd_grids') or self.app.rd_grids['U'].shape != (H, W):
            U = np.ones((H, W))
            V = np.zeros((H, W))
            seed_size = 10
            cx, cy = W // 2, H // 2
            U[cy-seed_size:cy+seed_size, cx-seed_size:cx+seed_size] = 0.5
            V[cy-seed_size:cy+seed_size, cx-seed_size:cx+seed_size] = 0.25
            self.app.rd_grids = {'U': U, 'V': V}
        else:
            U = self.app.rd_grids['U']
            V = self.app.rd_grids['V']

        Du = params.get("rd_Du", 1.0)
        Dv = params.get("rd_Dv", 0.5)
        feed = params.get("rd_feed", 0.055)
        kill = params.get("rd_kill", 0.062)
        
        lap_U = convolve2d(U, self.laplacian_kernel, mode='same', boundary='wrap')
        lap_V = convolve2d(V, self.laplacian_kernel, mode='same', boundary='wrap')

        uvv = U * V * V
        U_new = U + (Du * lap_U - uvv + feed * (1 - U))
        V_new = V + (Dv * lap_V + uvv - (feed + kill) * V)

        self.app.rd_grids['U'] = np.clip(U_new, 0, 1)
        self.app.rd_grids['V'] = np.clip(V_new, 0, 1)
        
        v_tensor = torch.from_numpy(self.app.rd_grids['V']).float().to(self.device).unsqueeze(0).unsqueeze(0)
        norm_map = (v_tensor - 0.5) * 2.0

        strength = params.get("rd_strength", 10.0)
        x_scale = strength / (W / 2)
        y_scale = strength / (H / 2)
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)
        offset_grid = base_grid.clone()
        displacement = norm_map.squeeze(1)
        offset_grid[:, :, :, 0] += displacement * x_scale
        offset_grid[:, :, :, 1] += displacement * y_scale
        
        return F.grid_sample(tensor, offset_grid, mode='bilinear', padding_mode='border', align_corners=False)

class ColorControlsEffect(Effect):
    name = "Color Controls"
    def apply(self, tensor, params):
        tensor = TF.adjust_brightness(tensor, params.get("brightness_factor", 1.0))
        tensor = TF.adjust_contrast(tensor, params.get("contrast_factor", 1.0))
        tensor = TF.adjust_saturation(tensor, params.get("saturation_factor", 1.0))
        hue_amount = params.get("hue_amount", 0.0)
        if hue_amount != 0:
            hsv_tensor = rgb_to_hsv(tensor)
            hue_shift_amount = hue_amount * (2 * torch.pi)
            hsv_tensor[:, 0, :, :] = (hsv_tensor[:, 0, :, :] + hue_shift_amount) % (2 * torch.pi)
            tensor = hsv_to_rgb(hsv_tensor)
        return tensor

class SimpleTileEffect(Effect):
    name = "Simple Tile"
    def apply(self, tensor, params):
        B, C, H, W = tensor.shape
        scale = params.get("tile_scale_factor", 0.5)
        if scale > 0 and W > 0 and H > 0:
            tile_w, tile_h = int(W * scale), int(H * scale)
            if tile_w > 0 and tile_h > 0:
                small_tile = F.interpolate(tensor, size=(tile_h, tile_w), mode='bilinear', align_corners=False)
                repeats_h = int(np.ceil(H / tile_h)); repeats_w = int(np.ceil(W / tile_w))
                tiled = small_tile.repeat(1, 1, repeats_h, repeats_w)
                return tiled[:, :, :H, :W]
        return tensor

class ShearEffect(Effect):
    name = "Shear (Tiled)"
    def apply(self, tensor, params):
        sx = params.get("shear_x_factor", 0.0)
        if sx == 0: return tensor
        B, C, H, W = tensor.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)
        flow_grid = base_grid.clone()
        flow_grid[:, :, :, 0] = base_grid[:, :, :, 0] - sx * base_grid[:, :, :, 1]
        wrapped_grid = (flow_grid + 1) % 2 - 1
        return F.grid_sample(tensor, wrapped_grid, mode='bilinear', padding_mode='border', align_corners=False)

class TiledRotateEffect(Effect):
    name = "Rotate (Tiled)"
    def apply(self, tensor, params):
        angle = params.get("rotate_angle_value", 0.0)
        if angle == 0: return tensor
        angle_rad = math.radians(-angle) 
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        affine_matrix = torch.tensor([[cos_a, -sin_a, 0], [sin_a,  cos_a, 0]], device=self.device, dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(affine_matrix, tensor.size(), align_corners=False)
        wrapped_grid = (grid + 1) % 2 - 1
        return F.grid_sample(tensor, wrapped_grid, mode='bilinear', padding_mode='border', align_corners=False)

class SymmetryEffect(Effect):
    name = "Symmetry"
    def apply(self, tensor, params):
        symmetry_type = params.get("symmetry_type", "None")
        B, C, H, W = tensor.shape
        if symmetry_type == "None": return tensor
        elif symmetry_type == "Horizontal (Left Master)":
            left = tensor[:, :, :, :W // 2]; right_flipped = torch.flip(left, dims=[3]); return torch.cat([left, right_flipped], dim=3)
        elif symmetry_type == "Horizontal (Right Master)":
            right = tensor[:, :, :, W // 2:]; left_flipped = torch.flip(right, dims=[3]); return torch.cat([left_flipped, right], dim=3)
        elif symmetry_type == "Vertical (Top Master)":
            top = tensor[:, :, :H // 2, :]; bottom_flipped = torch.flip(top, dims=[2]); return torch.cat([top, bottom_flipped], dim=2)
        elif symmetry_type == "Vertical (Bottom Master)":
            bottom = tensor[:, :, H // 2:, :]; top_flipped = torch.flip(bottom, dims=[2]); return torch.cat([top_flipped, bottom], dim=2)
        elif symmetry_type == "4-Way Mirror (Top-Left Master)":
            tl = tensor[:, :, :H // 2, :W // 2]; tr = torch.flip(tl, dims=[3]); bl = torch.flip(tl, dims=[2]); br = torch.flip(tl, dims=[2, 3])
            top_half = torch.cat([tl, tr], dim=3); bottom_half = torch.cat([bl, br], dim=3); return torch.cat([top_half, bottom_half], dim=2)
        elif symmetry_type == "2-Fold Rotational (Average)":
            rotated = torch.rot90(tensor, k=2, dims=[2, 3]); return (tensor + rotated) / 2.0
        elif symmetry_type.startswith("Kaleidoscope"):
            folds = 6 if "6-fold" in symmetry_type else 8
            angle_slice = math.pi / folds
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=self.device), torch.linspace(-1, 1, W, device=self.device), indexing='ij')
            angle = torch.atan2(grid_y, grid_x); radius = torch.sqrt(grid_x**2 + grid_y**2)
            wrapped_angle = angle % (2 * angle_slice)
            reflected_angle = torch.where(wrapped_angle > angle_slice, (2 * angle_slice) - wrapped_angle, wrapped_angle)
            new_x = radius * torch.cos(reflected_angle); new_y = radius * torch.sin(reflected_angle)
            sample_grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
            return F.grid_sample(tensor, sample_grid, mode='bilinear', padding_mode='border', align_corners=False)
        elif symmetry_type == "Diagonal Mirror (TL-BR Master)": return tensor.transpose(2, 3)
        elif symmetry_type == "Diagonal Mirror (TR-BL Master)": return torch.flip(tensor.transpose(2, 3), dims=[2, 3])
        return tensor

class RandomEffectsBlock(Effect):
    name = "Random Effects Block"
    def apply(self, tensor, params):
        num_to_pick = params.get("rand_num_to_pick", 1)
        randomize_params = params.get("rand_randomize_params", True)
        if num_to_pick == 0: return tensor
        randomizable_effects = [k for k in AVAILABLE_EFFECTS.keys() if k not in ["symmetry", "random_block", "shape"]]
        if not randomizable_effects: return tensor
        chosen_effect_keys = random.sample(randomizable_effects, k=min(num_to_pick, len(randomizable_effects)))
        temp_params = params.copy()
        for key in chosen_effect_keys:
            effect_class = AVAILABLE_EFFECTS[key]
            if randomize_params:
                config = self.app.effects_config.get(key, {})
                for param_name, p_config in config.get("params", {}).items():
                    if "min" in p_config and "max" in p_config:
                        var_key = p_config["var_key"]
                        min_val, max_val = p_config["min"], p_config["max"]
                        if p_config.get("is_int"):
                            temp_params[var_key] = random.randint(min_val, max_val)
                        else:
                            temp_params[var_key] = random.uniform(min_val, max_val)
            effect_instance = effect_class(self.app)
            tensor = effect_instance.apply(tensor, temp_params)
        return tensor

def hex_to_rgb_tensor(hex_str):
    """Helper function to parse a hex color string to a PyTorch tensor."""
    hex_str = hex_str.lstrip('#')
    return torch.tensor([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=torch.float32) / 255.0

class ShapeEffect(Effect):
    name = "Shape"

    def apply(self, tensor, params):
        b, c, h, w = tensor.shape
        device = tensor.device

        shape_type = params.get("shape_type", "Circle")
        center_x_norm = params.get("shape_x", 0.5)
        center_y_norm = params.get("shape_y", 0.5)
        size_norm = params.get("shape_size", 0.2)
        color_hex = params.get("shape_color", "#ffffff")
        blend_alpha = params.get("shape_blend", 0.5)
        has_outline = params.get("shape_outline", True)
        inner_effect_name = params.get("shape_inner_effect", "None")
        inner_strength = params.get("shape_inner_strength", 1.0)
        
        center_x = center_x_norm * w
        center_y = center_y_norm * h
        radius = (size_norm * min(w, h)) / 2

        y_coords = torch.arange(h, device=device).float()
        x_coords = torch.arange(w, device=device).float()
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        if shape_type == "Circle":
            fill_mask = ((xx - center_x)**2 + (yy - center_y)**2) < radius**2
        elif shape_type == "Rectangle":
            fill_mask = (torch.abs(xx - center_x) < radius) & (torch.abs(yy - center_y) < radius)
        elif shape_type == "Diamond":
            fill_mask = (torch.abs(xx - center_x) + torch.abs(yy - center_y)) < radius
        elif shape_type == "Triangle":
            p1 = (center_x, center_y - radius)
            p2 = (center_x + radius, center_y + radius)
            p3 = (center_x - radius, center_y + radius)
            d1 = (xx - p3[0]) * (p1[1] - p3[1]) - (p1[0] - p3[0]) * (yy - p3[1])
            d2 = (xx - p1[0]) * (p2[1] - p1[1]) - (p2[0] - p1[0]) * (yy - p1[1])
            d3 = (xx - p2[0]) * (p3[1] - p2[1]) - (p3[0] - p2[0]) * (yy - p2[1])
            fill_mask = ~((d1 < 0) | (d2 < 0) | (d3 < 0))
        else:
            fill_mask = torch.zeros((h, w), dtype=torch.bool, device=device)

        tensor_for_compositing = tensor
        if inner_effect_name != "None":
            inner_effect_key = next((k for k, v in AVAILABLE_EFFECTS.items() if v.name == inner_effect_name), None)
            if inner_effect_key:
                temp_params = params.copy()
                inner_config = self.app.effects_config.get(inner_effect_key, {})
                for p_name, p_config in inner_config.get("params", {}).items():
                    if p_config.get("animatable"):
                        var_key = p_config["var_key"]
                        base_val = temp_params.get(var_key, p_config["default"])
                        temp_params[var_key] = base_val * inner_strength
                        break
                
                inner_effect_instance = AVAILABLE_EFFECTS[inner_effect_key](self.app)
                warped_tensor = inner_effect_instance.apply(tensor.clone(), temp_params)
                tensor_for_compositing = torch.where(fill_mask.unsqueeze(0).unsqueeze(0), warped_tensor, tensor)
        
        shape_layer = tensor_for_compositing.clone()
        color_tensor = hex_to_rgb_tensor(color_hex).to(device).view(1, 3, 1, 1)
        
        if has_outline:
            outline_width = 2
            outline_radius = radius + outline_width
            outline_color_tensor = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)

            if shape_type == "Circle":
                outline_mask = ((xx - center_x)**2 + (yy - center_y)**2) < outline_radius**2
            elif shape_type == "Rectangle":
                outline_mask = (torch.abs(xx - center_x) < outline_radius) & (torch.abs(yy - center_y) < outline_radius)
            elif shape_type == "Diamond":
                outline_mask = (torch.abs(xx - center_x) + torch.abs(yy - center_y)) < outline_radius
            else: 
                outline_mask = fill_mask

            border_mask = outline_mask & ~fill_mask
            shape_layer = torch.where(border_mask.unsqueeze(0).unsqueeze(0), outline_color_tensor, shape_layer)
        
        shape_layer = torch.where(fill_mask.unsqueeze(0).unsqueeze(0), color_tensor, shape_layer)
        
        blended_tensor = torch.lerp(tensor_for_compositing, shape_layer, blend_alpha)
        
        return blended_tensor

AVAILABLE_EFFECTS = {
    "blur": BlurEffect,
    "unsharp_mask": UnsharpMaskEffect,
    "sharpen": SharpenEffect,
    "pixelate": PixelateEffect,
    "swirl": SwirlEffect,
    "chromatic_aberration": ChromaticAberrationEffect,
    "perspective_transform": PerspectiveTransformEffect,
    "channel_shift": ChannelShiftEffect,
    "displacement_map": DisplaceMapEffect,
    "wave_displace": WaveDisplaceEffect,
    "pixel_jitter": PixelJitterEffect,
    "spherical_distort": SphericalDistortEffect,
    "turbulence": TurbulenceEffect,
    "voronoi_noise": VoronoiNoiseEffect,
    "julia_distort": JuliaDistortEffect,
    "mandelbrot_distort": MandelbrotDistortEffect,
    "burning_ship_distort": BurningShipDistortEffect,
    "reaction_diffusion": ReactionDiffusionEffect,
    "color": ColorControlsEffect,
    "simple_tile": SimpleTileEffect,
    "shear": ShearEffect,
    "rotate": TiledRotateEffect,
    "symmetry": SymmetryEffect,
    "random_block": RandomEffectsBlock,
    "shape": ShapeEffect,
}