import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
from analyzer import LotteryAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Lottery Predictor")
        self.analyzer = LotteryAnalyzer()
        self._setup_ui()

    def _setup_ui(self):
        """Build the user interface."""
        # File upload section
        ttk.Label(self.root, text="Upload Excel File:").grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(self.root, text="Browse", command=self.upload_file).grid(row=0, column=1, padx=10)

        # Date selection
        ttk.Label(self.root, text="Select Draw Date:").grid(row=1, column=0, padx=10)
        self.date_entry = DateEntry(self.root)
        self.date_entry.grid(row=1, column=1, padx=10)

        # Buttons
        ttk.Button(self.root, text="Analyze Data", command=self.run_analysis).grid(row=2, column=0, pady=10)
        ttk.Button(self.root, text="Generate Predictions", command=self.generate_predictions).grid(row=2, column=1)

        # Results display
        self.results_text = tk.Text(self.root, height=15, width=50)
        self.results_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    def upload_file(self):
        """Handle Excel file upload."""
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.analyzer.load_data(file_path)
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def run_analysis(self):
        """Run frequency analysis and display results."""
        self.analyzer.analyze_frequencies()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, 
            f"Hot Numbers: {self.analyzer.hot_numbers}\n"
            f"Cold Numbers: {self.analyzer.cold_numbers}\n"
            f"Overdue Numbers: {self.analyzer.overdue_numbers}\n"
        )
        self.analyzer.plot_frequencies()

    def generate_predictions(self):
        """Generate and display predictions."""
        if self.analyzer.data is None:
            messagebox.showwarning("Error", "Upload data first!")
            return
        
        self.analyzer.train_model()
        predictions = self.analyzer.generate_predictions(self.date_entry.get_date())
        self.results_text.delete(1.0, tk.END)
        for idx, pred in enumerate(predictions, 1):
            self.results_text.insert(tk.END,
                f"Combination {idx}: {pred['numbers']} (Confidence: {pred['confidence']:.2%})\n"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryApp(root)
    root.mainloop()
