import sys
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import re
from scipy.stats import linregress, zscore
import pandas as pd
import tkinter as tk
from tkinter import ttk
import threading
from tkinter import filedialog
import glob

# Self-made classes
from Tilt_Correction_ClassBased import RotationApp
from CannyEdgeApp import CannyEdgeDetectionApp
from ImageThresholdApp import ImageThresholdApp
from CrackGraphApp import CrackGraphApp

# Initial values for elastic foundation model
beam_thickness_val = 3
beam_modulus_val = 180000
beam_width_val = 25.0
adhesive_modulus_val = 2000
adhesive_thickness_val = 0.01
displacement_val = 2.7

threshold_value = 100
comparison_choice = 'G_CC_mod'
scale_factor = 0

# fatigue test frequency
frequency = 5

# Global variables for image cropping
drawing = False  # True when mouse is pressed
start_x, start_y = -1, -1  # Starting coordinates
end_x, end_y = -1, -1  # Ending coordinates
cropped_region = None  # Store cropped image

# initialize strings for different filepaths
image_filepath = ""  # path to images
instron_filepath = ""  # path to .trends file
outputpath = ""  # path were evaluated files are saved
evaluation_filepath = ""  # path to the evaluation file for post-processing

# dictionary storage container for tkinter GUI elements
elements = {}
cycles = []

# default values for Hartman-Schijve fits
min_crack_prop = 0.2
D_HS = 0
A_HS = 1000
n_HS = 0
G_thresh = 0

# default thresholds for edge detection app
threshold1_global = 100
threshold2_global = 200

"""Compliance calibration and modified compliance calibration parameters"""

# modified CC at 23°C
#int_CC_mod = -2.33
#n_CC_mod = 630.9367

# EVA 60°C
n_CC = 2.7925
int_CC = 6.222

# EVA modified 60°C
int_CC_mod = -1.58934
n_CC_mod = 566.142

# Al/DP8810 60°C allometric fit
#n_CC = 2.78252
#int_CC = 6.31138

"""End of Compliance calibration parameters"""


def load_asset(path):
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    assets = os.path.join(base, "assets")
    return os.path.join(assets, path)


class JAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("900x578")
        self.configure(bg="#b5c1bf")
        self.title("J-Analyzer")
        self.resizable(False, False)

        self.canvas = tk.Canvas(
            self,
            bg="#b5c1bf",
            width=900,
            height=578,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # Dictionary to store GUI elements
        self.elements = {}

        self.create_shapes()

        self.load_assets()
        self.create_widgets()

    def load_assets(self):
        self.image_1 = tk.PhotoImage(file=load_asset("PostPro.png"))
        self.image_2 = tk.PhotoImage(file=load_asset("Files.png"))
        self.button_1_image = tk.PhotoImage(file=load_asset("1.png"))
        self.button_2_image = tk.PhotoImage(file=load_asset("2.png"))
        self.button_3_image = tk.PhotoImage(file=load_asset("3.png"))
        self.button_4_image = tk.PhotoImage(file=load_asset("4.png"))
        self.image_3 = tk.PhotoImage(file=load_asset("Settings.png"))
        self.button_5_image = tk.PhotoImage(file=load_asset("5.png"))
        self.button_6_image = tk.PhotoImage(file=load_asset("6.png"))
        self.button_7_image = tk.PhotoImage(file=load_asset("7.png"))
        self.button_8_image = tk.PhotoImage(file=load_asset("8.png"))
        self.button_9_image = tk.PhotoImage(file=load_asset("9.png"))
        self.button_10_image = tk.PhotoImage(file=load_asset("10.png"))
        self.button_11_image = tk.PhotoImage(file=load_asset("11.png"))
        self.button_12_image = tk.PhotoImage(file=load_asset("12.png"))
        self.button_13_image = tk.PhotoImage(file=load_asset("13.png"))

    def create_widgets(self):
        global elements
        self.canvas.create_image(213, 369, image=self.image_1)
        self.canvas.create_image(213, 55, image=self.image_2)
        self.canvas.create_image(652, 55, image=self.image_3)

        self.create_buttons()
        self.create_textboxes()
        self.create_labels()
        self.create_shapes()
        elements = self.elements

    def create_buttons(self):
        button_1 = tk.Button(image=self.button_1_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("1"))
        button_1.place(x=271, y=97, width=62, height=50)

        button_2 = tk.Button(image=self.button_2_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("2"))
        button_2.place(x=270, y=163, width=64, height=54)

        button_3 = tk.Button(image=self.button_3_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("3"))
        button_3.place(x=269, y=232, width=65, height=54)

        button_4 = tk.Button(image=self.button_4_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("4"))
        button_4.place(x=271, y=407, width=62, height=50)

        button_5 = tk.Button(image=self.button_5_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("5"))
        button_5.place(x=453, y=101, width=287, height=33)

        button_6 = tk.Button(image=self.button_6_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("6"))
        button_6.place(x=453, y=151, width=287, height=33)

        button_7 = tk.Button(image=self.button_7_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("7"))
        button_7.place(x=453, y=200, width=287, height=33)

        button_8 = tk.Button(image=self.button_8_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("8"))
        button_8.place(x=441, y=392, width=198, height=33)

        button_9 = tk.Button(image=self.button_9_image, relief="flat", borderwidth=0, highlightthickness=0,
                             command=lambda: print("9"))
        button_9.place(x=656, y=441, width=198, height=33)

        button_10 = tk.Button(image=self.button_10_image, relief="flat", borderwidth=0, highlightthickness=0,
                              command=lambda: print("10"))
        button_10.place(x=441, y=441, width=198, height=33)

        button_11 = tk.Button(image=self.button_11_image, relief="flat", borderwidth=0, highlightthickness=0,
                              command=lambda: print("11"))
        button_11.place(x=446, y=491, width=198, height=66)

        button_12 = tk.Button(image=self.button_12_image, relief="flat", borderwidth=0, highlightthickness=0,
                              command=lambda: print("12"))
        button_12.place(x=661, y=491, width=198, height=66)

        button_13 = tk.Button(image=self.button_13_image, relief="flat", borderwidth=0, highlightthickness=0,
                              command=lambda: print("13"))
        button_13.place(x=656, y=392, width=198, height=33)

        self.elements["button_1"] = button_1
        self.elements["button_2"] = button_2
        self.elements["button_3"] = button_3
        self.elements["button_4"] = button_4
        self.elements["button_5"] = button_5
        self.elements["button_6"] = button_6
        self.elements["button_7"] = button_7
        self.elements["button_8"] = button_8
        self.elements["button_9"] = button_9
        self.elements["button_10"] = button_10
        self.elements["button_11"] = button_11
        self.elements["button_12"] = button_12
        self.elements["button_13"] = button_13
        # Add more buttons as needed...

    def create_textboxes(self):
        entry_1 = TkForge_Entry(self, placeholder="1", bg="#d9d9d9", fg="#ffffff")
        entry_1.place(x=293, y=476, width=84, height=31)
        entry_2 = TkForge_Entry(self, placeholder="6", bg="#d9d9d9", fg="#ffffff")
        entry_2.place(x=207, y=299, width=170, height=31)
        entry_3 = TkForge_Entry(self, placeholder="3", bg="#d9d9d9", fg="#ffffff")
        entry_3.place(x=691, y=254, width=112, height=32)
        entry_4 = TkForge_Entry(self, placeholder="4", bg="#d9d9d9", fg="#ffffff")
        entry_4.place(x=691, y=300, width=112, height=31)
        entry_5 = TkForge_Entry(self, placeholder="5", bg="#d9d9d9", fg="#ffffff")
        entry_5.place(x=691, y=345, width=112, height=31)
        entry_6 = TkForge_Entry(self, placeholder="2", bg="#d9d9d9", fg="#ffffff")
        entry_6.place(x=293, y=521, width=84, height=31)
        self.elements["entry_1"] = entry_1
        self.elements["entry_2"] = entry_2
        self.elements["entry_3"] = entry_3
        self.elements["entry_4"] = entry_4
        self.elements["entry_5"] = entry_5
        self.elements["entry_6"] = entry_6

    def set_entry(self, entry_key, entry):
        if entry_key in self.elements:
            self.elements[entry_key].set_value(entry)

    def get_entry(self, entry_key):
        if entry_key in self.elements:
            return self.elements[entry_key].get()

    def create_labels(self):
        self.canvas.create_text(35, 110, anchor="nw", text="To Images", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(35, 176, anchor="nw", text="To Fatigue Data", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(35, 244, anchor="nw", text="To Output Directory", fill="#000000",
                                font=("Arial", 23 * -1))
        self.canvas.create_text(35, 419, anchor="nw", text="Evaluated File", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(35, 478, anchor="nw", text="SERR column name", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(43, 300, anchor="nw", text="Filename", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(450, 260, anchor="nw", text="Image Number", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(450, 303, anchor="nw", text="Cut Width", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(450, 346, anchor="nw", text="Cut Height", fill="#000000", font=("Arial", 23 * -1))
        self.canvas.create_text(35, 521, anchor="nw", text="Crack column name", fill="#000000", font=("Arial", 23 * -1))
        # Add more labels as needed...

    def create_shapes(self):
        # fill color green: #37ff00, red: #ff7972
        oval_1 = self.canvas.create_oval(361, 111, 384, 134, fill="#ff7972", outline="")
        self.elements["oval_1"] = oval_1
        oval_2 = self.canvas.create_oval(777, 107, 800, 130, fill="#ff7972", outline="")
        self.elements["oval_2"] = oval_2
        oval_3 = self.canvas.create_oval(361, 181, 384, 204, fill="#ff7972", outline="")
        self.elements["oval_3"] = oval_3
        oval_4 = self.canvas.create_oval(361, 253, 384, 276, fill="#ff7972", outline="")
        self.elements["oval_4"] = oval_4
        oval_5 = self.canvas.create_oval(361, 424, 384, 447, fill="#ff7972", outline="")
        self.elements["oval_5"] = oval_5
        oval_6 = self.canvas.create_oval(777, 156, 800, 179, fill="#ff7972", outline="")
        self.elements["oval_6"] = oval_6
        oval_7 = self.canvas.create_oval(777, 205, 800, 228, fill="#ff7972", outline="")
        self.elements["oval_7"] = oval_7

    def change_button_command(self, button_key, new_command):
        # Dynamically change the command of the button
        if button_key in self.elements:
            self.elements[button_key].config(command=new_command)

    def change_oval_color(self, oval_key, color):
        if oval_key in self.elements:
            oval = self.elements[oval_key]
            self.canvas.itemconfig(oval, fill=color)


class TkForge_Entry(tk.Entry):
    def __init__(self, master=None, placeholder="Enter text", placeholder_fg='grey', **kwargs):
        super().__init__(master, **kwargs)
        self.p, self.p_fg, self.fg = placeholder, placeholder_fg, self.cget("fg")
        self.putp()
        self.bind("<FocusIn>", self.toggle)
        self.bind("<FocusOut>", self.toggle)

    def putp(self):
        self.delete(0, tk.END)
        self.insert(0, self.p)
        self.config(fg=self.p_fg)
        self.p_a = True

    def toggle(self, event):
        if self.p_a:
            self.delete(0, tk.END)
            self.config(fg=self.fg)
            self.p_a = False
        elif not self.get():
            self.putp()

    def get(self):
        return '' if self.p_a else super().get()

    def is_placeholder(self, b):
        self.p_a = b
        self.config(fg=self.p_fg if b == True else self.fg)

    def get_placeholder(self):
        return self.p

    # New method to update the placeholder text
    def update_placeholder(self, new_placeholder):
        self.p = new_placeholder
        self.putp()  # This will reapply the updated placeholder text

    # Method to set an actual user input value (not just the placeholder)
    def set_value(self, value):
        """Sets the value of the entry directly, overriding the placeholder."""
        self.delete(0, tk.END)  # Clear any current content (including placeholder)
        self.insert(0, value)  # Insert the predefined value
        self.config(fg="black")  # Ensure the text is in the correct color (user input)
        self.p_a = False  # Indicate that this is not a placeholder anymore


def natural_sort_key(s):
    """A natural sorting key for strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def launch_rotation_tool(root):
    global image_filepath, start_y, end_y, threshold_value
    input_folder = image_filepath
    if not os.path.exists(input_folder):
        raise FileNotFoundError
    y_cut = (start_y, end_y)
    threshold = threshold_value

    def on_rotation_complete(output_path):
        global image_filepath
        image_filepath = output_path
        print(f"\n✅ Rotation complete. New images saved in:\n{output_path}")
        root.change_oval_color(oval_key="oval_6", color="#37ff00")
        # GUI updates should happen on the main thread, but Tkinter is usually okay with this.

    def run_rotation():
        RotationApp(root, input_folder, y_cut, threshold, completion_callback=on_rotation_complete)

    # Start rotation in a new thread
    threading.Thread(target=run_rotation, daemon=True).start()


def get_non_black_points(image):
    """
    Return the coordinates of points that are not black (value != 0) in a 2D grayscale image.

    Parameters:
    image (numpy.ndarray): A 2D array representing a grayscale image.

    Returns:
    list of tuple: A list of coordinates (row, column) for points that are not black.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Use numpy's vectorized non-zero lookup
    coords = np.transpose(np.nonzero(image))  # shape: (N, 2)
    return [tuple(coord) for coord in coords]


# creates cycle numbers according to the picture timing settings
def create_cycle_numbers():
    # adjust this section according to your picture timing settings
    freq = 5
    t1 = 15.2  # take picture every x seconds
    t1_lim = 600  # until t1_lim is reached.
    t2 = 60.2
    t2_lim = 3600
    t3 = 600.2
    t3_lim = 1 * 10 ** 6

    runtime = 0
    cur_cycle = 1
    cycles = []

    while runtime < t3_lim:
        if (runtime < t1_lim) and (runtime < t2_lim) and (runtime < t3_lim):
            if runtime == 0:
                cycles.append(cur_cycle)
                cur_cycle = t1 * freq + 1
                cycles.append(cur_cycle)
                runtime += t1
            else:
                cur_cycle += t1 * freq
                cycles.append(cur_cycle)
                runtime += t1

        if (runtime > t1_lim) and (runtime < t2_lim) and (runtime < t3_lim):
            cur_cycle += t2 * freq
            cycles.append(cur_cycle)
            runtime += t2

        if (runtime > t1_lim) and (runtime > t2_lim) and (runtime < t3_lim):
            cur_cycle += t3 * freq
            cycles.append(cur_cycle)
            runtime += t3
#    for index, value in enumerate(cycles):
#        print(f"{index}:{value}")
    return cycles


# couples pictures which are taken time-controlled with machine data, which is extracted every 1, 10, 100 or 1000 cycles
# uses interpolation if cycle numbers do not match
# calculated image cycles in "create_cycle_numbers" are matched with force data
def couple_force_and_cycles(path, cycles):
    global displacement_val
    # reads in the data from the instron file, adjust this section according to your machine data
    df = pd.read_csv(path, encoding='latin1', delimiter=",")
    machine_cycles = df['Total Cycles'].values.tolist()
    machine_forces = df['Load(Linear:Kraft):Maximum (N)'].values.tolist()
    machine_forces_min = df['Load(Linear:Kraft):Minimum (N)'].values.tolist()
    displacement_max = np.array(df["Displacement(Linear:Digitale Position):Maximum (mm)"].values.tolist())
    displacement_min = np.array(df["Displacement(Linear:Digitale Position):Minimum (mm)"].values.tolist())
    try:
        machine_G = np.array(df["StrainEnergy:User-Defined (J/m²)"].values.tolist())
        machine_a = np.array(df["CrackLength:User-Defined (mm)"].values.tolist())
    except KeyError:
        machine_G = machine_cycles
        machine_a = machine_cycles

    displacement_val = np.average(displacement_max[20:]-displacement_min[20:])
    print(f"The displacement in the test was: {displacement_val}")

    cyc_force = []

    for cycle in cycles:
        if cycle == 1:
            tup = (cycles[0], machine_forces[0], machine_forces_min[0], machine_G[0])
            cyc_force.append(tup)

        else:
            for count, value in enumerate(machine_cycles):
                if (machine_cycles[count - 1] < cycle) and (cycle <= machine_cycles[count]):
                    data_force = [[machine_cycles[count - 1], machine_forces[count - 1]],
                                  [machine_cycles[count], machine_forces[count]]]
                    data_force_min = [[machine_cycles[count - 1], machine_forces_min[count - 1]],
                                      [machine_cycles[count], machine_forces_min[count]]]
                    tup = (cycle, interpolation(data_force, cycle), interpolation(data_force_min, cycle),
                           machine_G[count], machine_a[count], displacement_max[count], displacement_min[count])
                    cyc_force.append(tup)
                    break
    return cyc_force


# function used for linear interpolation
def interpolation(d, x):
    output = d[0][1] + (x - d[0][0]) * ((d[1][1] - d[0][1]) / (d[1][0] - d[0][0]))
    return output


# Function to perform linear regression using scipy
def fit_linear_regression(points, z_thresh=2.0):
    x_values = np.array([point[1] for point in points])
    y_values = np.array([point[0] for point in points])

    try:
        # Initial regression
        slope, intercept, r_value, _, _ = linregress(x_values, y_values)

        # Predicted values
        y_pred = slope * x_values + intercept

        # Residuals
        residuals = y_values - y_pred

        # Z-scores of residuals
        z_scores = np.abs(zscore(residuals))

        # Keep only points within threshold
        mask = z_scores < z_thresh
        x_filtered = x_values[mask]
        y_filtered = y_values[mask]

        # Final regression with filtered data
        slope, intercept, r_value, _, _ = linregress(x_filtered, y_filtered)

    except ValueError:
        slope, intercept, r_value = 0, 0, 0
    return slope, intercept, r_value ** 2


def cut_image(image, x_tuple, y_tuple):
    x_start, x_end = x_tuple
    y_start, y_end = y_tuple

    cut_image = image[y_start:y_end, x_start:x_end]
    return cut_image


def average_white_pixel_y(img):
    """
    Calculate the average Y-coordinate of all white pixels in a binarized image.

    Parameters:
        img (numpy.ndarray): A binarized image (single channel, 0 and 255 values).

    Returns:
        float: The average Y-coordinate of white pixels. Returns None if no white pixels are found.
    """
    # Ensure the image is binary
    if len(img.shape) != 2:
        raise ValueError("Input image must be a single-channel binary image.")

    # Find coordinates of all white pixels (255)
    white_pixels = np.column_stack(np.where(img == 255))

    if white_pixels.size == 0:
        return None  # No white pixels found

    # Y-coordinates are in the first column (row indices)
    avg_y = np.mean(white_pixels[:, 0])
    return avg_y



# main function for image processing and extracting beam curvature
def process_images_in_folder(cyc_force_data):
    # get input variables from the entry fields
    global image_filepath
    global outputpath, comparison_choice
    global threshold1_global, threshold2_global
    global start_y, start_x, end_y, end_x, displacement_val

    folder_path = image_filepath
    filepath = outputpath
    filename = elements["entry_2"].get()
    x_cut_range = elements["entry_4"].get()
    x_cut_range = x_cut_range.split(',')
    x_cut_range = tuple(int(val.strip()) for val in x_cut_range)
    y_cut_range = elements["entry_5"].get()
    y_cut_range = y_cut_range.split(',')
    y_cut_range = tuple(int(val.strip()) for val in y_cut_range)

    if x_cut_range[0] != -1 and y_cut_range[0] != -1:
        x_cut_range = (start_x, end_x)
        y_cut_range = (start_y, end_y)

    if os.path.exists(folder_path):
        image_files = sorted(os.listdir(folder_path), key=natural_sort_key)
    else:
        raise AttributeError('could not load image')
    filepath = os.path.join(filepath, filename)

    popup = tk.Toplevel()
    popup.title("Processing...")
    popup.geometry("300x100")
    popup.resizable(False, False)

    label = ttk.Label(popup, text="Images are being processed...")
    label.pack(pady=10)

    progress = ttk.Progressbar(popup, orient="horizontal", length=250, mode="determinate")
    progress.pack(pady=5)
    progress["maximum"] = len(image_files)
    progress["value"] = 0

    # Update the UI to show the popup immediately
    popup.update()

    i = 0
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply threshold to extract black rectangles
            _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            current_edges = cv2.Canny(image, threshold1_global, threshold2_global)

            current_edges = cut_image(current_edges, x_cut_range, y_cut_range)
            extract_points = get_non_black_points(current_edges)
            # Split points into lower and upper halves based on the y-coordinate
            white_pixel_middle = average_white_pixel_y(cut_image(current_edges, x_cut_range, y_cut_range))
            lower_slope_points = [point for point in extract_points if point[0] >= white_pixel_middle]
            upper_slope_points = [point for point in extract_points if point[0] < white_pixel_middle]

            # Fit linear regression to lower and upper halves
            slope_lower, intercept_lower, r_squared_lower = fit_linear_regression(lower_slope_points)
            slope_upper, intercept_upper, r_squared_upper = fit_linear_regression(upper_slope_points)

            # Check R² difference
            r_diff = abs(r_squared_lower - r_squared_upper)

            if r_diff <= 0.05:
                # R² values are similar — use both slopes
                slope_add = abs(slope_upper) + abs(slope_lower)
            else:
                # Use only the slope with the higher R²
                if r_squared_upper > r_squared_lower:
                    slope_add = 2 * abs(slope_upper)
                else:
                    slope_add = 2 * abs(slope_lower)

            # calculates J_max and delta J
            j_integral_max = slope_add * cyc_force_data[i][1] / beam_width_val * 1000

            # calculate compliance calibration LEFM data
            compliance = displacement_val / cyc_force_data[i][1]
            crack_length_CC = (compliance / (10 ** (-int_CC))) ** (1 / n_CC)
            SERR_CC = cyc_force_data[i][1] * displacement_val * n_CC / (2 * crack_length_CC * beam_width_val) * 1000

            # calculate modified compliance calibration LEFM data
            crack_length_mod_CC = beam_thickness_val * (int_CC_mod + n_CC_mod * (compliance / 1000) ** (1 / 3))
            SERR_CC_mod = 3 * cyc_force_data[i][1] ** 2 * (compliance / 1000) ** (2 / 3) / (
                    2 * n_CC_mod * beam_thickness_val * beam_width_val * 10 ** (-6))

            header = "image;cycle;max. force;delta max;delta min;slope lower;slope upper;slopes added;" \
                     "J-max;Gmax-CC;a-CC;G-CC-mod;a-CC-mod;R-sq up;R-sq low"

            data = f"{i};{cyc_force_data[i][0]};{cyc_force_data[i][1]:.2f};{cyc_force_data[i][5]:2f};" \
                   f"{cyc_force_data[i][6]:2f};{round(slope_lower, 5)};" \
                   f"{round(slope_upper, 5)};{round(slope_add, 5)};{j_integral_max:.2f};" \
                   f"{SERR_CC:.2f};{crack_length_CC:.2f};{SERR_CC_mod:.2f};{crack_length_mod_CC:.2f};" \
                   f"{r_squared_upper:.2f};{r_squared_lower:.2f}"

            # calculations for the elastic foundation model (Erdman, 2000)
            #  calculates crack length based on current J-value and LEFM
            crack_with_jmax = 3*(cyc_force_data[i][5]-cyc_force_data[i][6])/(2*slope_add)
            # moment of inertia
            Inertia = (beam_thickness_val ** 3) * beam_width_val / 12
            # foundation modulus
            k = (adhesive_modulus_val * beam_width_val) / adhesive_thickness_val
            beta = (k / (4 * beam_modulus_val * Inertia)) ** (1 / 4)
            # crack length a
            a = ((((beta ** 8) * (3 * compliance * k - 4 * beta)) ** (1 / 3)) - 2 * beta ** 3) / (2 * beta ** 4)
            # G value based on simple beam theory
            G_efm = 3 * cyc_force_data[i][1] * displacement_val / (2 * beam_width_val * a) * 1000
            G_efm = np.array(G_efm)
            j_integral_max = np.array(j_integral_max)
            # percentage difference between modified CC and J integral value
            try:
                if comparison_choice.get() == "G_CC":
                    perc_diff = (cyc_force_data[i][3] / j_integral_max - 1) * 100
                    #perc_diff = (cyc_force_data[i][3] / j_integral_max - 1) * 100 # for G-controlled tests
                elif comparison_choice.get() == "G_CC_mod":
                    perc_diff = (cyc_force_data[i][3] / j_integral_max - 1) * 100
                else:
                    perc_diff = (cyc_force_data[i][3] / j_integral_max - 1) * 100
            except Exception as e:
                perc_diff = (SERR_CC / j_integral_max - 1) * 100

            perc_diff = perc_diff.tolist()
            header = header + f";Compliance;a-EFM;a-CWJ;G-EFM;Machine-G;Machine-a;%diff;displacement: {displacement_val:.2f}; " \
                              f"x-cut: {x_cut_range}\n"
            data = data + f';{compliance:.4f};{a:.2f};{crack_with_jmax:.2f};{G_efm:.2f};{cyc_force_data[i][3]};' \
                          f'{cyc_force_data[i][4]};{perc_diff:.2f}\n'

            # write data to csv file
            with open(filepath, mode='a') as csv_file:
                if i == 0:
                    csv_file.write(header)
                csv_file.write(data)

            # Update progress bar
            i += 1
            progress["value"] = i + 1
            popup.update_idletasks()

    # Close the popup
    popup.destroy()


# starts a new thread for image processing so the GUI stays responsive
def process_images_in_folder_threaded(cyc_force_data):
    # Create a new thread to run the function
    thread = threading.Thread(target=process_images_in_folder, args=(cyc_force_data,))
    # Start the thread
    thread.start()


def show_picture():
    # get input variables from the entry fields
    global image_filepath, comparison_choice
    global start_y, start_x, end_y, end_x

    folder_path = image_filepath
    image_number_show = int(elements["entry_3"].get())
    x_cut_range = elements["entry_4"].get()
    x_cut_range = x_cut_range.split(',')
    x_cut_range = tuple(int(val.strip()) for val in x_cut_range)
    y_cut_range = elements["entry_5"].get()
    y_cut_range = y_cut_range.split(',')
    y_cut_range = tuple(int(val.strip()) for val in y_cut_range)
    image_files = sorted(os.listdir(folder_path), key=natural_sort_key)
    image_file = image_files[image_number_show]

    if x_cut_range[0] != -1 and y_cut_range[0] != -1:
        x_cut_range = (start_x, end_x)
        y_cut_range = (start_y, end_y)

    if os.path.exists(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
    else:
        raise AttributeError('could not load image')

    # Read the image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply threshold to extract black rectangles
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    current_edges = cv2.Canny(image, threshold1_global, threshold2_global)

    current_edges = cut_image(current_edges, x_cut_range, y_cut_range)

    extract_points = get_non_black_points(current_edges)

    # Split points into lower and upper halves based on the y-coordinate
    white_pixel_middle = average_white_pixel_y(cut_image(current_edges, x_cut_range, y_cut_range))
    lower_slope_points = [point for point in extract_points if point[0] >= white_pixel_middle]  # y >= height/2
    upper_slope_points = [point for point in extract_points if point[0] < white_pixel_middle]  # y < height/2

    # Fit linear regression to lower and upper halves, lower and upper changed as pixel zero point is located
    # at upper left
    slope_lower, intercept_lower, r_squared_lower = fit_linear_regression(lower_slope_points)
    slope_upper, intercept_upper, r_squared_upper = fit_linear_regression(upper_slope_points)

    lower_slope_points_sorted = sorted(lower_slope_points, key=lambda x: x[1])
    upper_slope_points_sorted = sorted(lower_slope_points, key=lambda x: x[1])

    # Plot linear fits on the graph
    x_lower = np.array([points[1] for points in lower_slope_points_sorted])
    y_lower_fit = slope_lower * x_lower + intercept_lower
    x_upper = np.array([points[1] for points in upper_slope_points_sorted])
    y_upper_fit = slope_upper * x_upper + intercept_upper
    plt.imshow(current_edges, cmap="gray")
    plt.plot(x_upper, y_upper_fit, color='blue', label=f'Upper Fit: Rsq: {r_squared_upper:.2f}')
    plt.plot(x_lower, y_lower_fit, color='red', label=f'Lower Fit: Rsq: {r_squared_lower:.2f}')
    plt.title("Image with Valid Contours and Linear Fits")
    plt.axis('off')
    plt.legend()
    plt.show()


def write_params(params):
    global A_HS, D_HS, G_thresh, n_HS, min_crack_prop
    A_HS = params[0]
    D_HS = params[1]
    G_thresh = params[2]
    n_HS = params[3]
    min_crack_prop = params[4]
    print(f"A:{A_HS}, D:{D_HS}, Gth: {G_thresh}, n: {n_HS}, Min Prop: {min_crack_prop}")


def thresh_callback(param):
    global threshold_value
    print(param)
    threshold_value = int(param)


#callback function to auto adapt the image filepath within the image rotation tool
def image_path_callback(path):
    global image_filepath
    image_filepath = path
    print(f"New Image Filepath: {image_filepath}")
    image_filepath = path


#  creates the subwindow when "Additional Parameters" button is clicked, saves params by clicking save button
def open_subwindow(root, back_col, label_col, button_col):
    global beam_thickness_val, beam_modulus_val, beam_width_val, adhesive_modulus_val, adhesive_thickness_val
    global displacement_val
    global int_CC_mod, n_CC_mod, comparison_choice

    init_beam_thickness_val = tk.StringVar(value=beam_thickness_val)
    init_beam_modulus_val = tk.StringVar(value=beam_modulus_val)
    init_beam_width_val = tk.StringVar(value=beam_width_val)
    init_adhesive_modulus_val = tk.StringVar(value=adhesive_modulus_val)
    init_adhesive_thickness_val = tk.StringVar(value=adhesive_thickness_val)
    init_displacement_val = tk.StringVar(value=displacement_val)
    init_n_CC_val = tk.StringVar(value=n_CC)
    init_intersect_CC_val = tk.StringVar(value=int_CC)
    init_n_CC_mod = tk.StringVar(value=n_CC_mod)
    init_int_CC_mod = tk.StringVar(value=int_CC_mod)

    comparison_choice = tk.StringVar(value="G_CC")

    def save_settings():
        global beam_thickness_val, beam_modulus_val, beam_width_val, adhesive_modulus_val, adhesive_thickness_val
        global n_CC, int_CC
        global int_CC_mod, n_CC_mod
        global displacement_val, comparison_choice

        beam_thickness_val = float(beam_thickness_entry.get())
        beam_modulus_val = int(beam_modulus_entry.get())
        beam_width_val = float(beam_width_entry.get())
        adhesive_modulus_val = int(adhesive_modulus_entry.get())
        adhesive_thickness_val = float(adhesive_thickness_entry.get())
        displacement_val = float(displacement_entry.get())
        n_CC = float(n_CC_entry.get())
        int_CC = float(intersect_CC_entry.get())
        n_CC_mod = float(n_CC_mod_entry.get())
        int_CC_mod = float(intersect_CC_mod_entry.get())
        selected_comparison = comparison_choice.get()

        print("Saved settings:")
        print(f"Beam thickness: {beam_thickness_val}")
        print(f"Beam modulus: {beam_modulus_val}")
        print(f"Beam width: {beam_width_val}")
        print(f"Adhesive modulus: {adhesive_modulus_val}")
        print(f"Adhesive thickness: {adhesive_thickness_val}")
        print(f"Displacement: {displacement_val}")
        print(f"Slope of compliance calibration: {n_CC}")
        print(f"Intersect of compliance calibration: {int_CC}")
        print(f"Slope of modified compliance calibration: {n_CC_mod}")
        print(f"Intersect of modified compliance calibration: {int_CC_mod}")
        print(f"Comparison choice: {selected_comparison}")

        sub_window.destroy()

    sub_window = tk.Toplevel(root)
    sub_window.title("Additional parameters")
    sub_window.configure(background=back_col)

    tk.Label(sub_window, text="Beam thickness:", bg=label_col).grid(row=0, column=0, sticky='w')
    beam_thickness_entry = tk.Entry(sub_window, textvariable=init_beam_thickness_val)
    beam_thickness_entry.grid(row=0, column=1)

    tk.Label(sub_window, text="Beam modulus:", bg=label_col).grid(row=1, column=0, sticky='w')
    beam_modulus_entry = tk.Entry(sub_window, textvariable=init_beam_modulus_val)
    beam_modulus_entry.grid(row=1, column=1)

    tk.Label(sub_window, text="Beam width:", bg=label_col).grid(row=2, column=0, sticky='w')
    beam_width_entry = tk.Entry(sub_window, textvariable=init_beam_width_val)
    beam_width_entry.grid(row=2, column=1)

    tk.Label(sub_window, text="Adhesive modulus:", bg=label_col).grid(row=3, column=0, sticky='w')
    adhesive_modulus_entry = tk.Entry(sub_window, textvariable=init_adhesive_modulus_val)
    adhesive_modulus_entry.grid(row=3, column=1)

    tk.Label(sub_window, text="Adhesive thickness:", bg=label_col).grid(row=4, column=0, sticky='w')
    adhesive_thickness_entry = tk.Entry(sub_window, textvariable=init_adhesive_thickness_val)
    adhesive_thickness_entry.grid(row=4, column=1)

    tk.Label(sub_window, text="Displacement:", bg=label_col).grid(row=5, column=0, sticky='w')
    displacement_entry = tk.Entry(sub_window, textvariable=init_displacement_val)
    displacement_entry.grid(row=5, column=1)

    tk.Label(sub_window, text="slope Comp Cal.:", bg=label_col).grid(row=6, column=0, sticky='w')
    n_CC_entry = tk.Entry(sub_window, textvariable=init_n_CC_val)
    n_CC_entry.grid(row=6, column=1)

    tk.Label(sub_window, text="intersect Comp. Cal.:", bg=label_col).grid(row=7, column=0, sticky='w')
    intersect_CC_entry = tk.Entry(sub_window, textvariable=init_intersect_CC_val)
    intersect_CC_entry.grid(row=7, column=1)

    tk.Label(sub_window, text="slope mod Comp Cal.:", bg=label_col).grid(row=8, column=0, sticky='w')
    n_CC_mod_entry = tk.Entry(sub_window, textvariable=init_n_CC_mod)
    n_CC_mod_entry.grid(row=8, column=1)

    tk.Label(sub_window, text="intersect mod Comp. Cal.:", bg=label_col).grid(row=9, column=0, sticky='w')
    intersect_CC_mod_entry = tk.Entry(sub_window, textvariable=init_int_CC_mod)
    intersect_CC_mod_entry.grid(row=9, column=1)

    # Comparison selection
    tk.Label(sub_window, text="Compare J-integral with:", bg=label_col).grid(row=10, column=0, sticky='w')

    tk.Radiobutton(sub_window, text="G_CC", variable=comparison_choice, value="G_CC", bg=back_col).grid(row=10,
                                                                                                        column=1,
                                                                                                        sticky='w')
    tk.Radiobutton(sub_window, text="G_CC_mod", variable=comparison_choice, value="G_CC_mod", bg=back_col).grid(row=11,
                                                                                                                column=1,
                                                                                                                sticky='w')
    tk.Radiobutton(sub_window, text="G_EFM", variable=comparison_choice, value="G_EFM", bg=back_col).grid(row=12,
                                                                                                          column=1,
                                                                                                          sticky='w')

    # Save button
    tk.Button(sub_window, text="Save settings", command=save_settings, bg=button_col).grid(row=13, columnspan=4,
                                                                                           pady=10)


# reduces data in the Evaluation file according to a minimum increase in crack propagation
# not used for plotting but only for exporting data to .csv file
# similar to reduce_data function, but for writing to file
def reduce_data_by_crack_prop(path, MinProp: float, SERR_col, crack_col):
    global D_HS, n_HS, G_thresh, A_HS
    df = pd.read_csv(path, encoding='latin1', delimiter=';')
    crack_length = np.array(df[crack_col].tolist())
    gmax = np.array(df[SERR_col].tolist())
    n = np.array(df['cycle'].tolist())
    crack_sum = 0.0
    iterlist = []

    for i in range(len(crack_length) - 1):
        crackdiff = crack_length[i + 1] - crack_length[i]
        crack_sum += crackdiff

        if crack_sum <= MinProp:
            crack_sum += crackdiff
            iterlist.append(0)

        elif crack_sum < 0:
            crack_sum = 0

        else:
            iterlist.append(1)
            crack_sum = 0

    iterlist.append(1)

    iterlist = [int(x) for x in iterlist]
    reduced_cracklength = [a * b for a, b in zip(iterlist, crack_length)]
    reduced_gmax = [a * b for a, b in zip(iterlist, gmax)]
    reduced_n = [a * b for a, b in zip(iterlist, n)]

    reduced_cracklength = [i for i in reduced_cracklength if i != 0]
    reduced_gmax = [i for i in reduced_gmax if i != 0]
    reduced_n = [i for i in reduced_n if i != 0]
    reduced_da_dN = []

    for i in range(len(reduced_n) - 1):
        da_dN = (reduced_cracklength[i + 1] - reduced_cracklength[i]) / (reduced_n[i + 1] - reduced_n[i])
        reduced_da_dN.append(da_dN)

    reduced_cracklength = np.array(reduced_cracklength)
    reduced_gmax = np.array(reduced_gmax)
    reduced_da_dN = np.array(reduced_da_dN)
    reduced_gmax = reduced_gmax[1:]
    reduced_cracklength = reduced_cracklength[1:]

    X = (reduced_gmax ** (1 / 2) - (G_thresh) ** (1 / 2)) / (
            (1 - (reduced_gmax / A_HS) ** (1 / 2)) ** (1 / 2))

    da_dN_HS = D_HS * X ** n_HS

    params = [D_HS, n_HS, A_HS, G_thresh]

    with open(os.path.join(os.path.dirname(path), elements["entry_1"].get() + '__' + elements["entry_6"].get() + '__' +
                                                  'postprocessed_crack_' + str(round(MinProp, 2)) +
                                                  '_' + '.csv'), mode='a') as csv_file:
        for count, value in enumerate(reduced_cracklength):
            if count == 0:
                csv_file.write('reduced a length;reduced G or J;reduced da/dN;da/dN Hartman-Schijve;'
                               'Fitting Parameters\n')
            if count < len(params):
                string = f'{reduced_cracklength[count]:.2f};{reduced_gmax[count]:.2f};' \
                         f'{round(reduced_da_dN[count], 10)};{round(da_dN_HS[count], 10)};{round(params[count], 8)}\n'
                csv_file.write(string)
            else:
                string = f'{reduced_cracklength[count]:.2f};{reduced_gmax[count]:.2f};' \
                         f'{round(reduced_da_dN[count], 10)};{round(da_dN_HS[count], 10)}\n'
                csv_file.write(string)


def start_image_threshold_app(root):
    folder_path = image_filepath
    image_files = sorted(os.listdir(folder_path), key=natural_sort_key)
    image_number_show = int(elements["entry_3"].get())
    image_file = image_files[image_number_show]
    image_path = os.path.join(folder_path, image_file)

    app = ImageThresholdApp(root, image_path, callback=thresh_callback)
    root.change_oval_color(oval_key="oval_2", color="#37ff00")
    app.open_in_subwindow()


def browse_for_images(root):
    global image_filepath, cycles

    def natural_sort_key_cyc(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    file_path = filedialog.askdirectory()
    print(file_path)
    image_filepath = file_path
    # Get and sort image filenames
    image_files = sorted(os.listdir(file_path), key=natural_sort_key_cyc)
    # Extract numbers before the 's' from filenames
    time_values = []
    for filename in image_files:
        match = re.search(r'_(\d+)s\.(jpg|png)$', filename)
        if match:
            time_values.append(int(match.group(1)))

    # calculates cycles based on the timestamp in the image name (+1 as image is taken from the next cycle)
    cycles = np.array(time_values)*frequency+1
    #cycles[0] = 1
    for number, value in enumerate(cycles):
        if number == 0:
            continue
        else:
            cycles[number] += cycles[number-1]
    root.change_oval_color(oval_key="oval_1", color="#37ff00")


def browse_for_instron(root):
    file_path = filedialog.askopenfilename()
    global instron_filepath
    print(file_path)
    instron_filepath = file_path
    root.change_oval_color(oval_key="oval_3", color="#37ff00")


def browse_for_output(root):
    file_path = filedialog.askdirectory()
    global outputpath
    print(file_path)
    outputpath = file_path
    root.change_oval_color(oval_key="oval_4", color="#37ff00")


def browse_for_evaluation(root):
    global evaluation_filepath
    file_path = filedialog.askopenfilename()
    print(file_path)
    evaluation_filepath = file_path
    root.change_oval_color(oval_key="oval_5", color="#37ff00")
    crack_graph_app = CrackGraphApp(root, callback=write_params)
    crack_graph_app.open_subwindow(path=evaluation_filepath, SERR_col=elements["entry_1"].get(),
                                   crack_col=elements["entry_6"].get())


def open_canny_app(root):
    global image_filepath
    app = CannyEdgeDetectionApp(root, image_path=image_filepath)
    root.change_oval_color(oval_key="oval_7", color="#37ff00")
    app.open_window()


# Function for selecting the image boundaries by drawing a rectangle
def image_cropping(root):
    global image_filepath, scale_factor
    global start_x, end_x, start_y, end_y

    def get_image_paths(folder_path):
        """Returns a list of image paths (png, jpg, jpeg) from the given folder."""
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        return sorted(image_paths)  # Ensure natural sorting if needed

    images = get_image_paths(image_filepath)
    original_img = cv2.imread(images[5])  # Ensure images[5] exists

    # Resize image for display
    original_height, original_width = original_img.shape[:2]
    new_size = (512, 512)  # Target display size

    # Compute scale factor
    scale_factor_x = new_size[0] / original_width
    scale_factor_y = new_size[1] / original_height
    scale_factor = min(scale_factor_x, scale_factor_y)  # Keep aspect ratio

    resized_img = cv2.resize(original_img, (int(original_width * scale_factor), int(original_height * scale_factor)),
                             interpolation=cv2.INTER_NEAREST)

    # Display resized image for selection
    cv2.imshow("Select Region", resized_img)
    cv2.setMouseCallback("Select Region", draw_rectangle, (resized_img, original_img))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    root.set_entry(entry_key="entry_4", entry=f"{start_x}, {end_x}")
    root.set_entry(entry_key="entry_5", entry=f"{start_y}, {end_y}")


# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, cropped_region, scale_factor
    resized_img, original_img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = resized_img.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Region", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        cv2.rectangle(resized_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("Select Region", resized_img)

        if start_x != end_x and start_y != end_y:
            # Convert coordinates back to original image scale
            orig_start_x = int(start_x / scale_factor)
            orig_start_y = int(start_y / scale_factor)
            orig_end_x = int(end_x / scale_factor)
            orig_end_y = int(end_y / scale_factor)

            start_x = orig_start_x
            end_x = orig_end_x
            start_y = orig_start_y
            end_y = orig_end_y

            # Load original image again and crop correctly
            cropped_region = original_img[orig_start_y:orig_end_y, orig_start_x:orig_end_x]

            cv2.destroyWindow("Select Region")
            cv2.imshow("Cropped Image", cropped_region)




if __name__ == "__main__":
    # fill color green: #37ff00, red: #ff7972
    app = JAnalyzer()

    # Create cycle numbers -- Uncomment if pictures do not have time stamp, adjust timing settings!
    #cycles = create_cycle_numbers()

    # Set initial entries for entry fields
    app.set_entry(entry_key="entry_1", entry="J-max")
    app.set_entry(entry_key="entry_2", entry="Evaluation.csv")
    app.set_entry(entry_key="entry_3", entry=5)
    app.set_entry(entry_key="entry_4", entry="100, 1900")
    app.set_entry(entry_key="entry_5", entry="650, 1150")
    app.set_entry(entry_key="entry_6", entry="a-CC")

    app.change_button_command(button_key="button_1", new_command=lambda: browse_for_images(app))
    app.change_button_command(button_key="button_2", new_command=lambda: browse_for_instron(app))
    app.change_button_command(button_key="button_3", new_command=lambda: browse_for_output(app))
    app.change_button_command(button_key="button_4", new_command=lambda: browse_for_evaluation(app))

    app.change_button_command(button_key="button_5", new_command=lambda: start_image_threshold_app(app))
    app.change_button_command(button_key="button_6", new_command=lambda: launch_rotation_tool(app))
    app.change_button_command(button_key="button_7", new_command=lambda: open_canny_app(app))
    app.change_button_command(button_key="button_8", new_command=show_picture)
    app.change_button_command(button_key="button_9", new_command=app.quit)
    app.change_button_command(button_key="button_10", new_command=lambda: image_cropping(app))
    app.change_button_command(button_key="button_11",
                              new_command=lambda: process_images_in_folder_threaded(
                                  cyc_force_data=couple_force_and_cycles(path=instron_filepath, cycles=cycles)))

    app.change_button_command(button_key="button_12", new_command=lambda: reduce_data_by_crack_prop(
                       path=evaluation_filepath,
                       MinProp=float(min_crack_prop),
                       SERR_col=elements["entry_1"].get(),
                       crack_col=elements["entry_6"].get()))
    app.change_button_command(button_key="button_13", new_command=lambda:open_subwindow(root=app,
                                                                                        button_col="#bababa",
                                                                                        back_col="#f0f0f0",
                                                                                        label_col= "#f0f0f0"))

    app.mainloop()

