from tkinter import ttk
import tkinter as tk
import cv2
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import glob


def get_image_paths(folder_path):
    """Returns a list of image paths (png, jpg, jpeg) from the given folder."""
    image_extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_paths


class CannyEdgeDetectionApp:
    def __init__(self, parent, image_path):
        self.parent = parent
        self.window = None
        self.original_image = None
        self.resized_image = None
        self.edges = None
        self.edges_display = None
        self.image_path = image_path
        self.images = []

    def open_window(self):
        """
        Open the Canny Edge Detection application window.
        """
        self.window = tk.Toplevel(self.parent)
        self.window.title("Canny Edge Detection")

        # Frame for the image and edge detection sliders
        frame = ttk.Frame(self.window)
        frame.pack(padx=10, pady=10)

        self.images = get_image_paths(folder_path=self.image_path)

        # Button to select an image
        select_button = ttk.Button(frame, text="Select Image", command=self.select_image)
        select_button.grid(row=0, column=0, columnspan=2, pady=10)

        # Default Image
        select_button = ttk.Button(frame, text="Default Image", command=self.default_image)
        select_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Label for displaying the edge-detected image
        self.edge_label = ttk.Label(frame)
        self.edge_label.grid(row=1, column=0, columnspan=2)

        # Sliders for Threshold1 and Threshold2
        self.slider1 = tk.Scale(frame, from_=0, to=255, orient="horizontal", label="Threshold1", command=lambda x: self.update_edges())
        self.slider1.set(100)
        self.slider1.grid(row=2, column=0, padx=5, pady=5)

        self.slider2 = tk.Scale(frame, from_=0, to=255, orient="horizontal", label="Threshold2", command=lambda x: self.update_edges())
        self.slider2.set(200)
        self.slider2.grid(row=2, column=1, padx=5, pady=5)

        # Button to print the current threshold parameters
        print_button = ttk.Button(frame, text="Save Edges", command=self.print_parameters)
        print_button.grid(row=3, column=0, columnspan=2, pady=10)

    def update_edges(self):
        """
        Update the displayed edges based on the current slider values.
        """
        if self.resized_image is not None:
            threshold1 = self.slider1.get()
            threshold2 = self.slider2.get()
            self.edges = cv2.Canny(self.original_image, threshold1, threshold2)  # Use the original (non-resized) image for processing
            resized_edges = cv2.resize(self.edges, (512, 512))  # Resize edges for display
            self.edges_display = ImageTk.PhotoImage(image=Image.fromarray(resized_edges))
            self.edge_label.configure(image=self.edges_display)
            self.edge_label.image = self.edges_display

    def select_image(self):
        """
        Open a file dialog to select an image and display it with edge detection.
        """
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not self.image_path:
            return

        self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.resized_image = cv2.resize(self.original_image, (512, 512))  # Resize for display only
        self.update_edges()

    def default_image(self):
        """
        Open a file dialog to select an image and display it with edge detection.
        """
        try:
            self.original_image = cv2.imread(self.images[5], cv2.IMREAD_GRAYSCALE)
            self.resized_image = cv2.resize(self.original_image, (512, 512))  # Resize for display only
            self.update_edges()
        except Exception as e:
            print(f"An error occured {e}, try to select the image yourself using the Select Image Button")

    def print_parameters(self):
        """
        Print the selected threshold parameters to the console and save the edge-detected image.
        """
        if self.edges is not None and self.image_path:
            threshold1 = self.slider1.get()
            threshold2 = self.slider2.get()
            print(f"Threshold1: {threshold1}, Threshold2: {threshold2}")

            # Save the edge-detected image (non-resized)
            base, ext = os.path.splitext(self.image_path)
            output_path = f"{base}_edges{ext}"
            # cv2.imwrite(output_path, self.edges)
            # print(f"Edge-detected image saved to: {output_path}")

            # Return the thresholds to global variables
            global threshold1_global, threshold2_global, edge_image
            threshold1_global = threshold1
            threshold2_global = threshold2
            edge_image = self.edges