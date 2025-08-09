import os
import xml.etree.ElementTree as ET
from tkinter import messagebox
import re # Import the regular expression module

def _parse_map_file(filepath):
    """
    Parses a binary .map file (like those from JWildfire/Apophysis).
    Handles both legacy (768 bytes) and modern formats with headers
    by reading the last 768 bytes of the file.
    """
    try:
        with open(filepath, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            if file_size >= 768:
                f.seek(-768, os.SEEK_END)
            else:
                f.seek(0)
            data = f.read()

        colors = []
        for i in range(0, len(data), 3):
            if i + 2 < len(data):
                r, g, b = data[i], data[i+1], data[i+2]
                colors.append((r, g, b))
        return colors
    except Exception as e:
        messagebox.showerror("MAP Parse Error", f"Could not parse .map file:\n{e}")
        return []

def _parse_ugr_xml_file(filepath):
    """Parses an XML-based .ugr file."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            xml_content = f.read()
        
        start_index = xml_content.find('<')
        if start_index == -1:
            # This indicates it's likely not an XML file, so we return None to try the next parser.
            return None
        
        clean_xml = xml_content[start_index:]
        root = ET.fromstring(clean_xml)
        
        points = root.findall('.//point')
        if not points:
            points = root.findall('entry')

        colors = []
        for point in points:
            color_int = int(point.get('color'))
            blue = color_int & 255
            green = (color_int >> 8) & 255
            red = (color_int >> 16) & 255
            colors.append((red, green, blue))
        return colors
    except ET.ParseError:
        # This is also expected if it's not an XML file, so we let the next parser try.
        return None 
    except Exception as e:
        # For any other unexpected error, show the message box.
        messagebox.showerror("UGR XML Parse Error", f"Could not parse .ugr file as XML:\n{e}")
        return []

def _parse_ugr_text_file(filepath):
    """Parses a text-based (non-XML) .ugr file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use regex to find all lines with 'color=NUMBER'
        color_matches = re.findall(r'color=(\d+)', content)
        if not color_matches:
            raise ValueError("No 'color=' entries found in the file.")
            
        colors = []
        for color_str in color_matches:
            color_int = int(color_str)
            blue = color_int & 255
            green = (color_int >> 8) & 255
            red = (color_int >> 16) & 255
            colors.append((red, green, blue))
        return colors
    except Exception as e:
        messagebox.showerror("UGR Text Parse Error", f"Could not parse .ugr file as text:\n{e}")
        return []


def rgb_to_hex(rgb_tuple):
    """Converts an (r, g, b) tuple to a #RRGGBB hex string."""
    return f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"

def load_gradient_from_file(filepath):
    """
    Loads a gradient from a file, automatically detecting the format.
    Returns a comma-separated hex string.
    """
    if not os.path.exists(filepath):
        return ""

    _, extension = os.path.splitext(filepath)
    extension = extension.lower()

    rgb_colors = []
    if extension == '.map':
        rgb_colors = _parse_map_file(filepath)
    elif extension == '.ugr':
        # First, try parsing as XML. This is the more common format.
        rgb_colors = _parse_ugr_xml_file(filepath)
        # If it returns None, it means it wasn't XML, so try the text parser.
        if rgb_colors is None:
            rgb_colors = _parse_ugr_text_file(filepath)
    else:
        messagebox.showwarning("Unsupported Format", f"File format '{extension}' is not supported.")
        return ""

    if not rgb_colors:
        return ""

    hex_string = ",".join([rgb_to_hex(c) for c in rgb_colors])
    return hex_string

