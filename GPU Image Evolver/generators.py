from PIL import Image
import numpy as np
import noise
import math

def hex_to_rgb(hex_color):
    """Converts a hex color string to an (r, g, b) tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6: return (0, 0, 0)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def parse_color_string(color_str):
    """Parses a comma-separated hex string into a list of RGB tuples."""
    return [np.array(hex_to_rgb(c.strip())) for c in color_str.split(',') if c.strip()]

class Generator:
    """Base class for all image generators."""
    name = "Base Generator"
    params = {}

    def generate(self, width, height, params):
        """Generates and returns a PIL Image. Must be overridden."""
        raise NotImplementedError

class LinearGradient(Generator):
    name = "Linear Gradient"
    params = {
        'colors': {'type': 'multicolor', 'label': 'Colors', 'default': '#FF0000,#0000FF'},
        'angle': {'type': 'slider', 'label': 'Angle', 'min': 0, 'max': 360, 'default': 0},
    }

    def generate(self, width, height, p):
        colors = parse_color_string(p['colors'])
        if not colors: return Image.new('RGB', (width, height))
        
        stops = np.linspace(0, 1, len(colors))
        angle_rad = np.deg2rad(p['angle'])
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        
        rot_x = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
        t = (rot_x - rot_x.min()) / (rot_x.max() - rot_x.min())
        
        r = np.interp(t, stops, [c[0] for c in colors])
        g = np.interp(t, stops, [c[1] for c in colors])
        b = np.interp(t, stops, [c[2] for c in colors])
        
        img_array = np.stack([r, g, b], axis=-1)
        return Image.fromarray(img_array.astype(np.uint8), 'RGB')

class RadialGradient(Generator):
    name = "Radial Gradient"
    params = {
        'colors': {'type': 'multicolor', 'label': 'Colors', 'default': '#FFFFFF,#000000'},
        'center_x': {'type': 'slider', 'label': 'Center X', 'min': -1.0, 'max': 1.0, 'default': 0.0},
        'center_y': {'type': 'slider', 'label': 'Center Y', 'min': -1.0, 'max': 1.0, 'default': 0.0},
    }

    def generate(self, width, height, p):
        colors = parse_color_string(p['colors'])
        if not colors: return Image.new('RGB', (width, height))

        stops = np.linspace(0, 1, len(colors))
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)

        distance = np.sqrt((xx - p['center_x'])**2 + (yy - p['center_y'])**2)
        t = distance / distance.max()

        r = np.interp(t, stops, [c[0] for c in colors])
        g = np.interp(t, stops, [c[1] for c in colors])
        b = np.interp(t, stops, [c[2] for c in colors])

        img_array = np.stack([r, g, b], axis=-1)
        return Image.fromarray(img_array.astype(np.uint8), 'RGB')

class StripesGenerator(Generator):
    name = "Stripes"
    params = {
        'colors': {'type': 'multicolor', 'label': 'Colors', 'default': '#FFFFFF,#000000'},
        'num_stripes': {'type': 'slider_int', 'label': 'Count', 'min': 2, 'max': 50, 'default': 10},
        'angle': {'type': 'slider', 'label': 'Angle', 'min': 0, 'max': 360, 'default': 45},
    }

    def generate(self, width, height, p):
        colors = parse_color_string(p['colors'])
        if not colors: return Image.new('RGB', (width, height))
        num_colors = len(colors)

        angle_rad = np.deg2rad(p['angle'])
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # Center coordinates
        cx, cy = width / 2, height / 2
        
        # Rotate grid and scale
        t = (xx - cx) * math.cos(angle_rad) + (yy - cy) * math.sin(angle_rad)
        t_scaled = (t - t.min()) / (t.max() - t.min()) * p['num_stripes']
        
        # Get color index for each pixel
        color_indices = np.floor(t_scaled).astype(int) % num_colors
        
        # Create image from indices and color list
        img_array = np.array(colors)[color_indices]
        return Image.fromarray(img_array.astype(np.uint8), 'RGB')

class PerlinNoise(Generator):
    name = "Perlin Noise"
    params = {
        'scale': {'type': 'slider', 'label': 'Scale', 'min': 10.0, 'max': 200.0, 'default': 50.0},
        'octaves': {'type': 'slider_int', 'label': 'Octaves', 'min': 1, 'max': 8, 'default': 4},
        'persistence': {'type': 'slider', 'label': 'Persistence', 'min': 0.1, 'max': 1.0, 'default': 0.5},
        'lacunarity': {'type': 'slider', 'label': 'Lacunarity', 'min': 1.0, 'max': 4.0, 'default': 2.0},
        'colors': {'type': 'multicolor', 'label': 'Colors', 'default': '#222222,#EEEEEE'},
    }

    def generate(self, width, height, p):
        colors = parse_color_string(p['colors'])
        if not colors: return Image.new('RGB', (width, height))
        
        stops = np.linspace(0, 1, len(colors))
        arr = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                arr[y][x] = noise.pnoise2(x / p['scale'], y / p['scale'],
                                          octaves=p['octaves'], persistence=p['persistence'],
                                          lacunarity=p['lacunarity'], base=0)
        
        t = (arr - arr.min()) / (arr.max() - arr.min())

        r = np.interp(t, stops, [c[0] for c in colors])
        g = np.interp(t, stops, [c[1] for c in colors])
        b = np.interp(t, stops, [c[2] for c in colors])
        
        img_array = np.stack([r, g, b], axis=-1)
        return Image.fromarray(img_array.astype(np.uint8), 'RGB')

# The registry of all available generators
AVAILABLE_GENERATORS = {
    "linear_gradient": LinearGradient,
    "radial_gradient": RadialGradient,
    "stripes": StripesGenerator,
    "perlin_noise": PerlinNoise,
}