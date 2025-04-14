import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

class Tester:
    """
    Test the performance of price prediction models
    Enhanced to handle failures gracefully and provide more metrics
    """

    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = min(size, len(data))  # Ensure size doesn't exceed data length
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []  # Squared log errors
        self.colors = []
        self.failed = []

    def color_for(self, error, truth):
        """Determine color coding based on prediction error"""
        if error < 40 or error/truth < 0.2:
            return "green"
        elif error < 80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_datapoint(self, i):
        """Run prediction for a single data point with error handling"""
        datapoint = self.data[i]
        try:
            guess = self.predictor(datapoint)
            truth = datapoint.price
            error = abs(guess - truth)
            
            # Calculate log error with protection against negative values
            if guess <= 0:
                guess = 0.01  # Avoid log(0) issues
            log_error = math.log(truth+1) - math.log(guess+1)
            sle = log_error ** 2
            
            color = self.color_for(error, truth)
            title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
            
            self.guesses.append(guess)
            self.truths.append(truth)
            self.errors.append(error)
            self.sles.append(sle)
            self.colors.append(color)
            self.failed.append(False)
            
            print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")
            
        except Exception as e:
            # Handle failures gracefully
            print(f"{RED}{i+1}: FAILED - {str(e)}{RESET}")
            self.failed.append(True)

    def chart(self, title):
        """Create visualization of prediction performance"""
        if not self.guesses:
            print("No successful predictions to chart")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Create 45-degree reference line
        max_val = max(max(self.truths), max(self.guesses)) * 1.1
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6, label='Perfect Prediction')
        
        # Plot predictions with color coding
        colors_mapped = ['green' if c == 'green' else 'orange' if c == 'orange' else 'red' for c in self.colors]
        plt.scatter(self.truths, self.guesses, s=20, c=colors_mapped, alpha=0.7)
        
        # Add trendline
        if len(self.truths) > 1:
            z = np.polyfit(self.truths, self.guesses, 1)
            p = np.poly1d(z)
            plt.plot(sorted(self.truths), p(sorted(self.truths)), 
                    color='purple', linestyle='--', linewidth=2, 
                    label=f'Trend Line (y = {z[0]:.2f}x + {z[1]:.2f})')
        
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add error distribution inset
        if len(self.truths) >= 10:
            inset_ax = plt.axes([0.65, 0.2, 0.25, 0.25])
            relative_errors = [abs(g-t)/t if t > 0 else abs(g-t) for g, t in zip(self.guesses, self.truths)]
            inset_ax.hist(relative_errors, bins=20, color='skyblue', alpha=0.7)
            inset_ax.set_title('Relative Error Distribution')
            inset_ax.set_xlabel('Relative Error')
            inset_ax.set_ylabel('Count')
            
        plt.tight_layout()
        plt.show()

    def report(self):
        """Generate and display performance metrics"""
        if not self.guesses:
            print(f"{RED}No successful predictions - all attempts failed{RESET}")
            return
            
        success_count = len(self.guesses)
        fail_count = self.failed.count(True)
        
        # Calculate metrics
        average_error = sum(self.errors) / success_count
        median_error = sorted(self.errors)[success_count // 2] if success_count > 0 else 0
        rmsle = math.sqrt(sum(self.sles) / success_count) if success_count > 0 else float('inf')
        
        hits = sum(1 for color in self.colors if color == "green")
        hit_rate = hits / success_count * 100 if success_count > 0 else 0
        
        # Calculate additional metrics
        mean_absolute_percentage_error = sum(e/t if t > 0 else e for e, t in zip(self.errors, self.truths)) / success_count if success_count > 0 else float('inf')
        
        # Print comprehensive report
        print("\n" + "="*50)
        print(f"PERFORMANCE REPORT: {self.title}")
        print("="*50)
        print(f"Total items tested: {self.size}")
        print(f"Successful predictions: {success_count} ({success_count/self.size*100:.1f}%)")
        print(f"Failed predictions: {fail_count} ({fail_count/self.size*100:.1f}%)")
        print("-"*50)
        print(f"Mean Absolute Error (MAE): ${average_error:.2f}")
        print(f"Median Absolute Error: ${median_error:.2f}")
        print(f"Root Mean Squared Log Error (RMSLE): {rmsle:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error*100:.2f}%")
        print(f"Hit Rate (Error < 20% or $40): {hit_rate:.1f}%")
        print("="*50)
        
        # Chart title with key metrics
        title = f"{self.title}: MAE=${average_error:.2f}, RMSLE={rmsle:.2f}, Hit Rate={hit_rate:.1f}%"
        self.chart(title)

    def run(self):
        """Execute testing for all data points"""
        print(f"Testing {self.title} on {self.size} items...")
        
        for i in tqdm(range(self.size)):
            self.run_datapoint(i)
            
        self.report()

    @classmethod
    def test(cls, function, data, size=None):
        """Class method for convenient testing"""
        test_size = size or min(250, len(data))
        cls(function, data, size=test_size).run()
