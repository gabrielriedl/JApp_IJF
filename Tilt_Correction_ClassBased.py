import os
import re
import cv2
import numpy as np
from scipy.stats import linregress
from skimage.transform import rotate
from tkinter import Tk, Toplevel, Button, Label, filedialog, ttk, messagebox
import tkinter as tk
import matplotlib.pyplot as plt


class ImageRotationTool:
    def __init__(self, input_folder, y_cut, x_cut, threshold):
        self.input_folder = input_folder
        self.output_folder = os.path.join(input_folder, "Tilt Correction")
        self.y_cut = y_cut
        self.x_cut = x_cut
        self.threshold = threshold

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def process_images(self, progress_callback=None):
        os.makedirs(self.output_folder, exist_ok=True)
        image_files = sorted(
            [f for f in os.listdir(self.input_folder) if f.endswith(".png")],
            key=self.natural_sort_key
        )

        fig, axes = plt.subplots(3, 2, figsize=(12, 9))
        plot_index = 0

        for i, image_name in enumerate(image_files):
            try:
                image_path = os.path.join(self.input_folder, image_name)
                image_path = os.path.normpath(image_path)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = image[self.y_cut[0]:self.y_cut[1], self.x_cut[0]:self.x_cut[1]]

                _, binary = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
                height, width = binary.shape
                # Define the section of the image to analyze
                center_start = int(0.2 * width)
                center_end = int(0.8 * width)

                # Extract only the columns in the desired center section
                columns = binary[:, center_start:center_end]

                # Create array of x indices corresponding to the section
                x_indices = np.arange(center_start, center_end)

                # Create mask where pixel is 255
                mask = columns == 255  # shape: (height, section_width)

                # y_indices will broadcast to each column
                y_indices = np.arange(binary.shape[0])[:, np.newaxis]

                # Compute the sum of y indices where mask is True
                weighted_sum = (mask * y_indices).sum(axis=0)

                # Count the number of 255s per column
                count = mask.sum(axis=0)

                # Avoid division by zero: only compute mean where count > 0
                valid = count > 0
                center_y = np.zeros_like(count, dtype=float)
                center_y[valid] = weighted_sum[valid] / count[valid]

                # Zip x and center_y only for valid columns
                center_points = list(zip(x_indices[valid], center_y[valid]))
                center_points = np.array(center_points)

                angle = 0

                if len(center_points) > 1:
                    slope, _, _, _, _ = linregress(center_points[:, 0], center_points[:, 1])
                    angle = np.degrees(np.arctan(slope))

                rotated_image = rotate(image, angle, resize=False, mode="edge")
                rotated_image = (rotated_image * 255).astype(np.uint8)
                output_path = os.path.join(self.output_folder, image_name)
                cv2.imwrite(output_path, rotated_image)
                print(f"{image_name} was rotated by {angle}°")

                if i == 0 or i == int(len(image_files)/2) or i == len(image_files)-1:
                    if plot_index < 3:
                        x_vals = center_points[:, 0]
                        y_vals = center_points[:, 1]
                        axes[plot_index, 0].imshow(image, cmap="gray")
                        axes[plot_index, 0].plot(x_vals, y_vals, 'r-', linewidth=2)
                        axes[plot_index, 0].set_title(f"Original: {image_name}")

                        axes[plot_index, 1].imshow(rotated_image, cmap="gray")
                        axes[plot_index, 1].set_title(f"Rotated: {image_name}")
                        plot_index += 1

                if progress_callback:
                    progress_callback((i + 1) / len(image_files) * 100)

            except:
                # Create a root window (hidden) for the messagebox to work
                root_pop = tk.Tk()
                root_pop.withdraw()
                messagebox.showerror(
                    "Image Loading Error",
                    "An error occurred when trying to read the images.\n"
                    "Make sure you crop before trying image alignment."
                )
                break

        plt.tight_layout()
        plt.show()

        print("All images processed and saved in:", self.output_folder)


class RotationApp:
    def __init__(self, master, input_folder, y_cut, x_cut, threshold, completion_callback):
        self.master = master
        self.window = Toplevel(master)
        self.window.title("Tilt Correction Tool")
        self.window.geometry("400x200")

        self.tool = ImageRotationTool(input_folder, y_cut, x_cut, threshold)

        self.label = Label(self.window, text="Ready to rotate images.")
        self.label.pack(pady=10)

        self.progress = ttk.Progressbar(self.window, length=300, mode='determinate')
        self.progress.pack(pady=10)

        self.start_button = Button(self.window, text="Start", command=self.start_rotation)
        self.start_button.pack(pady=10)
        self.completion_callback = completion_callback

    def update_progress(self, value):
        self.progress["value"] = value
        self.master.update_idletasks()

    def start_rotation(self):
        self.label.config(text="Processing...")
        self.tool.process_images(progress_callback=self.update_progress)
        self.label.config(text="Done!")
        if self.completion_callback:
            self.completion_callback(self.tool.output_folder)


# MAIN APPLICATION (Example)
def open_rotation_tool():
    input_folder = filedialog.askdirectory(title="Select Image Folder")
    if not input_folder:
        return
    y_cut = (650, 1150)
    threshold = 38
    RotationApp(root, input_folder, y_cut, threshold, completion_callback=None)


if __name__ == "__main__":
    root = Tk()
    root.title("Main Application")
    root.geometry("300x150")

    open_button = Button(root, text="Open Rotation Tool", command=open_rotation_tool)
    open_button.pack(pady=40)

    root.mainloop()
