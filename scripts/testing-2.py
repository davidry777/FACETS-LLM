import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from segmentation_agent import SegmentationAgent


def generate_sample_data(num_customers=500, output_path="sample_rfm_data.csv"):
    """
    Generate synthetic RFM data for testing
    """
    print(f"Generating sample RFM data with {num_customers} customers")
    
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f"CUST{i:05d}" for i in range(1, num_customers + 1)]
    
    # Create distinct customer segments with different characteristics
    # Segment 1: High-value loyal customers (low recency, high frequency, high monetary)
    # Segment 2: At-risk high-value (high recency, high frequency, high monetary)
    # Segment 3: New customers (low recency, low frequency, medium monetary)
    # Segment 4: Lost cheap customers (high recency, low frequency, low monetary)
    
    segment_sizes = [int(num_customers * p) for p in [0.2, 0.3, 0.25, 0.25]]
    # Adjust the last segment to ensure we have exactly num_customers
    segment_sizes[-1] = num_customers - sum(segment_sizes[:-1])
    
    # Generate data for each segment
    recency = []
    frequency = []
    monetary = []
    
    # Segment 1: High-value loyal customers
    recency.extend(np.random.normal(5, 3, segment_sizes[0]).clip(1, 30))
    frequency.extend(np.random.normal(12, 3, segment_sizes[0]).clip(5, 20))
    monetary.extend(np.random.normal(150, 30, segment_sizes[0]).clip(100, 300))
    
    # Segment 2: At-risk high-value
    recency.extend(np.random.normal(45, 15, segment_sizes[1]).clip(30, 90))
    frequency.extend(np.random.normal(10, 2, segment_sizes[1]).clip(5, 20))
    monetary.extend(np.random.normal(120, 25, segment_sizes[1]).clip(80, 250))
    
    # Segment 3: New customers
    recency.extend(np.random.normal(10, 5, segment_sizes[2]).clip(1, 20))
    frequency.extend(np.random.normal(3, 1, segment_sizes[2]).clip(1, 5))
    monetary.extend(np.random.normal(80, 20, segment_sizes[2]).clip(50, 150))
    
    # Segment 4: Lost cheap customers
    recency.extend(np.random.normal(75, 15, segment_sizes[3]).clip(50, 120))
    frequency.extend(np.random.normal(2, 1, segment_sizes[3]).clip(1, 5))
    monetary.extend(np.random.normal(40, 15, segment_sizes[3]).clip(10, 80))
    
    # Create the DataFrame
    data = {
        'Customer ID': customer_ids,
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample RFM data saved to {output_path}")
    
    return output_path


def test_segmentation_workflow(data_path=None):
    """
    Test the full segmentation workflow
    """
    print("===== Starting Segmentation Agent Test =====")
    
    # Generate sample data if not provided
    if data_path is None or not os.path.exists(data_path):
        data_path = generate_sample_data()
    
    # Initialize segmentation agent
    agent = SegmentationAgent()
    print("\n===== Agent initialized =====")
    
    # Load data
    success = agent.load_data(data_path)
    if not success:
        print("Failed to load data. Exiting test.")
        return
    print("\n===== Data loaded successfully =====")
    
    # Perform segmentation
    print("\n===== Performing K-means segmentation =====")
    results = agent.perform_segmentation()
    
    # Print segment summary
    print("\n===== Segmentation Results =====")
    print(f"Number of segments: {results['num_segments']}")
    print(f"Number of customers: {results['num_customers']}")
    
    print("\nSegment Breakdown:")
    for segment_id, segment in results['segments'].items():
        print(f"Segment {segment_id} - {segment['segment_name']}:")
        print(f"  Count: {segment['count']} ({segment['percentage']:.1f}%)")
        print(f"  Avg Recency: {segment['avg_recency']:.1f}")
        print(f"  Avg Frequency: {segment['avg_frequency']:.1f}")
        print(f"  Avg Monetary: ${segment['avg_monetary']:.2f}")
    
    # Generate visualization
    print("\n===== Generating segment visualization =====")
    viz_path = agent.plot_segments()
    print(f"Visualization saved to: {viz_path}")
    
    # Generate report
    print("\n===== Generating segmentation report =====")
    report_path = agent.generate_report()
    print(f"Report saved to: {report_path}")
    
    # Test individual customer segmentation
    print("\n===== Testing customer classification =====")
    
    # Test cases: customers with different RFM profiles
    test_customers = [
        {"Recency": 5, "Frequency": 15, "Monetary": 200},  # Likely high value
        {"Recency": 60, "Frequency": 10, "Monetary": 150},  # Likely at risk
        {"Recency": 8, "Frequency": 2, "Monetary": 70},     # Likely new
        {"Recency": 90, "Frequency": 1, "Monetary": 30}     # Likely lost
    ]
    
    for i, customer in enumerate(test_customers):
        print(f"\nCustomer {i+1}:")
        print(f"  Recency: {customer['Recency']}")
        print(f"  Frequency: {customer['Frequency']}")
        print(f"  Monetary: ${customer['Monetary']}")
        
        result = agent.segment_customer(customer)
        
        print(f"  Assigned to: {result['segment_name']} (ID: {result['segment_id']})")
    
    print("\n===== Test completed successfully =====")


if __name__ == "__main__":
    test_segmentation_workflow()