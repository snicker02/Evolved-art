Getting Started with the Modular Image Evolver 🎨
Welcome! This program is a creative tool for making evolving, abstract art. The main idea is to start with an image and then apply a series of effects to it over and over, creating a feedback loop that results in unique and unexpected visuals.

1. Setup: Downloading and Running
First, you'll need to download the project files from GitHub.

Download the Project: Go to the project's GitHub page. Click the green < > Code button and select "Download ZIP".

Unzip the Files: Find the downloaded ZIP file on your computer and extract its contents into a new folder.

Check Your Files: Inside the folder, you should have four essential Python files:

image_evolver.py (the main application)

effects.py

generators.py

gradient_loader.py

Run the Program: Open your terminal or command prompt, navigate to the folder where you extracted the files (e.g., cd path/to/ImageEvolver), and run the following command:

Bash

python image_evolver.py
The application window should now open.

2. Installation
Before you can use all the features, you need to install a few libraries. Run these commands in your terminal.

Core Requirements (CPU Mode)
These are required for the program to function.

Bash

pip install Pillow numpy noise scipy
Optional GPU Acceleration
Install these only if you have a compatible NVIDIA GPU.

Install PyTorch: Get the correct command for your system from the official PyTorch website. A common command is:

Bash

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
Install Kornia: After PyTorch is installed, run:

Bash

pip install kornia
3. Getting Your First Image (The Canvas)
You can't do anything without an image to work on. You have two choices:

A) Generate an Image (Recommended for fun!):

Look at the "Image Input & Generation" section.

In the listbox, click on a generator like "Perlin Noise" or "Gradient".

Adjust any options that appear, then click the Generate New Image button.

B) Load Your Own Image:

Click the Load Image File button.

Choose any image from your computer.

4. The Effects Pipeline
This is the core of the program. Think of it like a factory assembly line for your image.

Available Effects: The list on the left shows all the effects you can use.

Active Pipeline: The list on the right is your "assembly line." Effects here will be applied to your image from top to bottom. The order matters!

The Buttons:

Add >>: Adds a selected effect to your pipeline.

<< Remove: Removes a selected effect from your pipeline.

Move Up / Move Down: Change the order of your active effects.

5. Making Changes and Seeing Results
Once you have at least one effect in your Active Pipeline, you can start evolving.

Adjust Parameters: Click on an effect in your Active Pipeline (e.g., "Swirl"). Its sliders and options will appear in the "Effect Parameters" box below.

Use the Action Buttons:

Preview Step: Applies your pipeline one time.

Hold to Evolve: Applies the effects over and over, showing the feedback loop in action without changing the parameters on the sliders.

Multi-Step Evolve: Runs the evolution for a set number of Steps. This is the button to use for creating animations, as it will make your parameters change over time if animation is enabled.

✨ A Fun First Project
Let's make something!

Generate a Canvas: Select "Perlin Noise" and click Generate New Image.

Add a Warp: From the "Available Effects" list, select "Swirl" and click Add >>.

Adjust the Swirl: Click on "Swirl" in your pipeline. Drag the Strength slider to the right and click Preview Step a few times.

Add Some Color: Now, select "Color Controls" and Add >> it to your pipeline below Swirl.

Adjust the Color: Click on "Color Controls" and play with the Saturation and Hue Shift sliders.

Evolve! Click and hold the Hold to Evolve button and watch your image transform!

Experiment with Feedback: Find the "Final Operations" section and lower the Feedback slider to 0.98. Now when you evolve, the effect will be more ghostly and painterly.

🚀 Using the Movable Shape Effect
This feature lets you apply effects to just one part of the image.

Add the "Shape" effect to your pipeline.

To move it, simply right-click anywhere on the image preview! The shape will instantly jump to where you clicked.

To create a "portal" effect, select another effect like "Turbulence" from the Inner Effect dropdown. Now, that effect will only happen inside your shape! Use the Inner Strength slider to control its intensity.

🖼️ Saving Your Work
Save a Still Image: When you have an image you like, just click the Save Image button.

Save an Animation: Before clicking Multi-Step Evolve, check the box that says "Save Animation Frames". After the evolution is finished, it will ask you where to save the individual frames.

Happy evolving!
![image](https://github.com/user-attachments/assets/974123e2-7039-4be2-a0db-5f2d1779f68f)
