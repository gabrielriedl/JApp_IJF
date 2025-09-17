import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit

class CrackGraphApp:
    def __init__(self, root, callback):
        self.root = root
        self.min_crack_prop = 0.0
        self.D_HS = None
        self.n_HS = None
        self.A_HS = None
        self.G_thresh = None
        self.callback = callback  # Store the callback function

    def edit_parameters(self, path, crack_col, SERR_col):
        # Create a new subwindow
        self.subwindow = tk.Toplevel(self.root)
        self.subwindow.title("Crack Graph Subwindow")

        required_columns = ['image', 'cycle', 'max. force', 'slope lower', 'slope upper',
                            'slopes added', 'J-max', 'Gmax-CC', 'a-CC',
                            'G-CC-mod', 'a-CC-mod', 'R-sq up', 'R-sq low', 'Compliance',
                            'a-EFM', 'a-eq', 'G-EFM']

        # Read and preprocess the data
        df = pd.read_csv(path, encoding='latin1', delimiter=';', skipinitialspace=True,
                         usecols=required_columns).dropna().reset_index(drop=True)
        df = df.iloc[1:]

        a = np.array(df[crack_col].tolist())
        Gmax = np.array(df[SERR_col].tolist())
        count = np.array(df['cycle'].tolist())
        a, Gmax, count = self._sanitize_data(a, Gmax, count)

        reduced_a, reduced_Gmax, reduced_dadN = self.reduce_data(a, Gmax, count, self.min_crack_prop)

        # Create a frame for the subwindow
        sub_frame = ttk.Frame(self.subwindow)
        sub_frame.pack(padx=10, pady=10)

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(reduced_Gmax, reduced_dadN)
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('Gmax [J/m²]')
        self.ax.set_ylabel('da/dN [mm/cycle]')
        self.canvas = FigureCanvasTkAgg(self.fig, master=sub_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize default values for parameters
        initial_A_factor = 1.1  # Initial slider value for A factor
        initial_Gth_factor = 0.95  # Initial slider value for Gth factor
        self.A_HS = Gmax[1] * initial_A_factor
        self.G_thresh = Gmax[-1] * initial_Gth_factor

        # Add sliders and labels
        self._add_slider(sub_frame, "Data reduction:", 0, 1.4, 400,
                         lambda value: self.update_plot_min_prop(float(value)), "min crack propagation")
        self._add_slider(sub_frame, "Adjust Gth factor:", 0.9, 0.999, 400,
                         lambda value: self.update_Gth(Gmax, float(value)), "Gth factor", initial_Gth_factor)
        self._add_slider(sub_frame, "Adjust Parameter A factor:", 1.0001, 1.4, 400,
                         lambda value: self.update_A(Gmax, float(value)), "A factor", initial_A_factor)

        # Add a save parameters button
        save_params_button = ttk.Button(sub_frame, text="Save Parameters", command=self._save_parameters)
        save_params_button.pack(pady=(10, 0))

        self.path = path
        self.crack_col = crack_col
        self.SERR_col = SERR_col
        self.a = a
        self.Gmax = Gmax
        self.count = count

        # Trigger an initial update to render the graph
        self._update_plot()

    def _save_parameters(self):
        # Retrieve the parameters to pass to the main application
        params = [self.A_HS, self.D_HS, self.G_thresh, self.n_HS, self.min_crack_prop]

        # Call the callback function to pass parameters back to the main app
        self.callback(params)
        # Optionally, close the subwindow after saving
        self.subwindow.destroy()


    def update_Gth(self, Gmax, value):
        # Update G_thresh based on slider value and update the plot
        self.G_thresh = Gmax[-1] * value
        self._update_plot()

    def update_A(self, Gmax, value):
        # Update A_HS based on slider value and update the plot
        self.A_HS = Gmax[1] * value
        self._update_plot()

    def _sanitize_data(self, a, Gmax, count):
        a = np.delete(a, 0)
        Gmax = np.delete(Gmax, 0)
        count = np.delete(count, 0)
        mask = ~np.isnan(a) & ~np.isnan(Gmax) & ~np.isnan(count)
        return a[mask], Gmax[mask], count[mask]

    def _add_slider(self, frame, label_text, from_, to, length, command, label_prefix, initial_value=None):
        label = ttk.Label(frame, text=label_text)
        label.pack()
        slider_var = tk.DoubleVar(value=initial_value if initial_value is not None else from_)
        slider = ttk.Scale(frame, from_=from_, to=to, length=length, orient=tk.HORIZONTAL, variable=slider_var,
                           command=lambda value: command(float(value)))
        slider.pack()

        value_label = ttk.Label(frame, text=f"{label_prefix}: {slider_var.get():.3f}")
        value_label.pack()
        slider_var.trace_add("write", lambda *args: value_label.config(text=f"{label_prefix}: {slider_var.get():.3f}"))

    def _save_and_close(self):
        print(f"min_crack_prop: {self.min_crack_prop}")
        self.subwindow.destroy()

    def reduce_data(self, CrackLength, Gmax, count, MinProp):
        cracksum = 0.0
        iterlist = []
        for i in range(len(CrackLength) - 1):
            crackdiff = CrackLength[i + 1] - CrackLength[i]
            if crackdiff < 0:
                crackdiff = 0
            cracksum += crackdiff
            if cracksum <= MinProp:
                iterlist.append(0)
            else:
                iterlist.append(1)
                cracksum = 0
        iterlist.append(1)
        iterlist = [int(x) for x in iterlist]
        reduced_cracklength = [a * b for a, b in zip(iterlist, CrackLength)]
        reduced_Gmax = [a * b for a, b in zip(iterlist, Gmax)]
        reduced_n = [a * b for a, b in zip(iterlist, count)]
        reduced_cracklength = [i for i in reduced_cracklength if i != 0]
        reduced_Gmax = [i for i in reduced_Gmax if i != 0]
        reduced_n = [i for i in reduced_n if i != 0]
        reduced_da_dN = [(reduced_cracklength[i + 1] - reduced_cracklength[i]) / (reduced_n[i + 1] - reduced_n[i])
                         for i in range(len(reduced_n) - 1)]
        reduced_Gmax.pop(0)
        return np.array(reduced_cracklength), np.array(reduced_Gmax), np.array(reduced_da_dN)

    def update_plot_min_prop(self, value):
        self.min_crack_prop = value
        self._update_plot()

    def update_plot_Gth(self, value):
        self.G_thresh = value
        self._update_plot()

    def update_plot_A(self, value):
        self.A_HS = value
        self._update_plot()

    def _update_plot(self):
        reduced_a, reduced_Gmax, reduced_dadN = self.reduce_data(self.a, self.Gmax, self.count, self.min_crack_prop)
        X = self._calculate_X(reduced_Gmax)

        # Filter out invalid values
        valid_mask = np.isfinite(X) & np.isfinite(reduced_dadN) & (X > 0) & (reduced_dadN > 0)
        X_valid = np.log10(X[valid_mask])
        reduced_dadN_valid = np.log10(reduced_dadN[valid_mask])

        if len(X_valid) < 2 or len(reduced_dadN_valid) < 2:
            print("Insufficient valid data points for curve fitting.")
            return

        try:
            pars, _ = curve_fit(f=self._objective, xdata=X_valid, ydata=reduced_dadN_valid)
            self.D_HS, self.n_HS = 10 ** pars[0], pars[1]
            da_dN_HS = self.D_HS * X ** self.n_HS
        except Exception as e:
            print(f"Error during curve fitting: {e}")
            return

        self.ax.clear()
        self.ax.scatter(reduced_Gmax, reduced_dadN, label="Data")
        if len(valid_mask) > 0:
            self.ax.plot(reduced_Gmax[valid_mask], da_dN_HS[valid_mask], color='red', label="HS Fit")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('G or J [J/m²]')
        self.ax.set_ylabel('da/dN [mm/cycle]')
        self.ax.legend()
        self.canvas.draw()

    def _calculate_X(self, reduced_Gmax):
        return (reduced_Gmax ** 0.5 - self.G_thresh ** 0.5) / (1 - (reduced_Gmax / self.A_HS) ** 0.5) ** 0.5

    @staticmethod
    def _objective(x, D, n):
        return D + n * x

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = CrackGraphApp(root)
    app.open_subwindow(r"C:\Users\ak120\Desktop\Backup Riedl\PostDoc\02 Papers & Presentations\02 J integral DCB\NewCameraTrigger\AlDp490_23C_nT2\fatigue\AlEP_23C_newCam_secondTest1\Evaluation.csv", "a-CC", "J-max")
    root.mainloop()
