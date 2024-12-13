import tkinter as tk
import open3d as o3d
import numpy as np
from pose_estimation import Model, Scene, PointCloudManager, PoseEstimatorFPFH
from utils import display_point_clouds

# MAIN PARAMETERS
sModelPath = "Input/T-stuk-filled.stl"
sScenePath = "PointCloudImages/PointClouds_2024-12-09_15-31-59/2024-12-09_15-32-10/PointCloud_2024-12-09_15-32-10.ply"
sExtrinsicsPath = "CalibrationData/arrTransMat.npy"


class ModelViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface")
        self.root.geometry("800x600")

        self.oModel = None
        self.vis_model = None

        self.visualization_open = False  # Track the state of the visualization window

        self.label_text = tk.StringVar(value="Load the model first")
        self.label = tk.Label(self.root, textvariable=self.label_text, font=("Arial", 14))
        self.label.pack()

        self.loadModel_button = tk.Button(self.root, text="Load Model", command=self.load_model)
        self.loadModel_button.pack()

        self.showModel_button = tk.Button(self.root, text="Display Model", command=self.toggle_visualization)
        self.showModel_button.pack()

        self._update_interval = 100  # Tick interval in milliseconds (e.g., 100 ms = 0.1 seconds)

    def load_model(self):
        try:
            # Load your model or point cloud here
            # Simulating loading for now
            self.oModel = Model(sModelPath, 2000, 10)

            if self.oModel:
                self.loadModel_button.config(bg="light green")
                self.label_text.set("Model loaded successfully")

        except Exception as e:
            self.loadModel_button.config(bg="light red")
            self.label_text.set(f"Error loading model: {e}")

    def toggle_visualization(self):
        """Toggle Open3D visualization window."""
        if not self.visualization_open:
            # Open the Open3D visualization window
            self.display_model()
        else:
            # Close the Opqen3D visualization window
            self.close_visualization()

    def display_model(self):
        if self.oModel is None:
            self.label_text.set("Please load the model first!")
            return

        try:
            # Initialize Open3D visualization window
            self.vis_model = o3d.visualization.Visualizer()
            self.vis_model.create_window(window_name="Model Visualization", width=800, height=600)
            self.vis_model.add_geometry(self.oModel.pcdModel)

            # Mark visualization as open
            self.visualization_open = True

            # Start the Open3D event loop in the background
            self._run_open3d_visualizer()

            # Start periodic update (tick function)
            self.root.after(self._update_interval, self.tick)

        except Exception as e:
            self.label_text.set(f"Error displaying model: {e}")

    def tick(self):
        """This function is called repeatedly at a fixed interval."""
        if self.vis_model:
            self.vis_model.poll_events()  # Handle user input and events in Open3D
            self.vis_model.update_renderer()  # Update the renderer

        # Schedule the next tick after the specified interval
        self.root.after(self._update_interval, self.tick)

    def _run_open3d_visualizer(self):
        """Run the Open3D visualization loop in a non-blocking manner."""
        # This method runs Open3D's visualization loop in a way that doesn't block Tkinter's event loop
        while True:
            self.vis_model.poll_events()
            self.vis_model.update_renderer()
            # Add any periodic tasks here (like camera movement or model updates)

            # This makes sure Tkinter's mainloop isn't blocked, so the UI stays responsive
            self.root.update()

    def close_visualization(self):
        """Close the Open3D visualization window."""
        if self.vis_model:
            self.vis_model.destroy_window()
            self.visualization_open = False  # Mark visualization as closed

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelViewerApp(root)
    root.mainloop()
