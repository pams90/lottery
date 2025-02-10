import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LotteryAnalyzer:
    def __init__(self):
        self.data = None
        self.hot_numbers = []
        self.cold_numbers = []
        self.overdue_numbers = []
        self.model = RandomForestClassifier()

    def load_data(self, file_path):
        """Load and preprocess Excel data."""
        self.data = pd.read_excel(file_path)
        self._preprocess_data()
        return self.data

    def _preprocess_data(self):
        """Clean and derive features from raw data."""
        # Convert numbers to lists of integers
        if 'Winning Numbers' in self.data.columns:
            self.data['Winning Numbers'] = self.data['Winning Numbers'].apply(
                lambda x: list(map(int, str(x).split(',')))
        
        # Derive features
        self.data['Sum'] = self.data['Winning Numbers'].apply(sum)
        self.data['Even Count'] = self.data['Winning Numbers'].apply(
            lambda x: sum(1 for num in x if num % 2 == 0))
        self.data['Low Numbers'] = self.data['Winning Numbers'].apply(
            lambda x: sum(1 for num in x if num <= 25))

    def analyze_frequencies(self):
        """Identify hot/cold numbers and overdue numbers."""
        all_numbers = [num for sublist in self.data['Winning Numbers'] for num in sublist]
        freq = Counter(all_numbers)
        self.hot_numbers = [num for num, _ in freq.most_common(10)]
        self.cold_numbers = [num for num, _ in freq.most_common()[:-11:-1]]
        
        # Find numbers not drawn in the last 50 draws
        recent_numbers = set(num for sublist in self.data['Winning Numbers'].tail(50) for num in sublist)
        self.overdue_numbers = [num for num in range(1, 50) if num not in recent_numbers]

    def train_model(self):
        """Train a machine learning model to predict numbers."""
        X = self.data[['Sum', 'Even Count', 'Low Numbers']]
        y = self.data['Winning Numbers']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)

    def quantum_monte_carlo(self, num_simulations=1000):
        """Simulate quantum-inspired probabilistic draws."""
        simulated = []
        for _ in range(num_simulations):
            # Quantum-inspired weighted sampling (prioritize hot/overdue numbers)
            weights = [3 if num in self.hot_numbers else 1 if num in self.overdue_numbers else 0.5 
                       for num in range(1, 50)]
            draw = np.random.choice(range(1, 50), size=6, replace=False, p=np.array(weights)/sum(weights))
            simulated.append(sorted(draw))
        return simulated

    def generate_predictions(self, draw_date):
        """Generate predictions with confidence scores."""
        simulated = self.quantum_monte_carlo()
        counts = Counter(tuple(combo) for combo in simulated)
        top_combos = counts.most_common(10)
        predictions = [
            {"numbers": list(combo), "confidence": count / len(simulated)}
            for combo, count in top_combos
        ]
        return predictions

    def plot_frequencies(self):
        """Visualize number frequencies using Matplotlib."""
        all_numbers = [num for sublist in self.data['Winning Numbers'] for num in sublist]
        freq = Counter(all_numbers)
        plt.bar(freq.keys(), freq.values())
        plt.title("Number Frequencies (1993-2025)")
        plt.xlabel("Number")
        plt.ylabel("Frequency")
        plt.show()
