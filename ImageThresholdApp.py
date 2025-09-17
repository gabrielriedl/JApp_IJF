from tkinter import ttk
import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont


# Handles the image thresholding within the sub window, saves the user-defined threshold value in the global variable
class ImageThresholdApp:
    def __init__(self, parent, image_path, callback):
        self.parent = parent
        self.image_path = image_path
        self.threshold_value = None
        self.image_label = None  # Initialize image label attribute
        self.callback = callback

    def open_in_subwindow(self):
        self.sub_window = tk.Toplevel(self.parent)

        # Create a label to display the actual value of the slider
        slider_value_label = ttk.Label(self.sub_window, text="Slider Value:")
        slider_value_label.pack(padx=10, pady=5)

        self.slider_value_text = tk.StringVar()
        slider_value_entry = ttk.Entry(self.sub_window, textvariable=self.slider_value_text, state="readonly")
        slider_value_entry.pack(padx=10, pady=5)

        # Placeholder label creation
        self.image_label = tk.Label(self.sub_window)
        self.image_label.pack(padx=10, pady=10)

        self.slider = ttk.Scale(self.sub_window, from_=0, to=255, orient="horizontal", command=self.update_threshold, length=800)
        self.slider.set(127)  # Set initial threshold
        self.slider.pack(pady=10)

        return_button = ttk.Button(self.sub_window, text="Save threshold", command=self.return_threshold)
        return_button.pack(pady=5)

        self.display_image()

    def return_threshold(self):  # saves the image threshold value in the global variable
        self.callback(self.threshold_value)
        self.sub_window.destroy()

    # Applies the threshold in the picture according to slider value
    def update_threshold(self, threshold):
        threshold_value = int(float(threshold))
        original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, thresholded_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Convert the thresholded image to PIL format
        image_pil = Image.fromarray(thresholded_image)
        self.threshold_value = threshold_value
        self.slider_value_text.set(str(threshold_value))

        # Draw text overlay for the current threshold value
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()  # You can choose a different font if needed
        current_threshold = threshold_value  # Get current threshold value from slider
        draw.text((10, 10), f"Threshold: {current_threshold}", fill=(255,), font=font)

        # Resize the image to fit the window
        max_size = (512, 512)
        image_pil.thumbnail(max_size)

        # Convert the image to Tkinter format
        image_tk = ImageTk.PhotoImage(image_pil)

        # Update the label with the new image
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def display_image(self):
        # Load the original image
        original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Convert the thresholded image to PIL format
        image_pil = Image.fromarray(original_image)

        # Draw text overlay for the current threshold value
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()  # You can choose a different font if needed
        current_threshold = self.slider.get()  # Get current threshold value from slider
        draw.text((10, 10), f"Threshold: {current_threshold}", fill=(255,), font=font)

        # Resize the image to fit the window
        max_size = (512, 512)
        image_pil.thumbnail(max_size)

        # Convert the image to Tkinter format
        image_tk = ImageTk.PhotoImage(image_pil)


        # Update the label with the new image
        if self.image_label is None:  # Create label if not initialized
            self.image_label = tk.Label(self.sub_window, image=image_tk)
            self.image_label.pack(padx=10, pady=10)
        else:
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk