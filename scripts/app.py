import gradio as gr
import pandas as pd
import numpy as np
import os
import json
import tempfile
from typing import Dict, Any, List
from pathlib import Path

# Import agents
try:
    from segmentation_agent import SegmentationAgent
    from ensemble_agent import EnsembleAgent
    # You would also import other agents like ensemble_agent here
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing agents: {str(e)}")
    AGENTS_AVAILABLE = False

# Initialize global state
class AppState:
    def __init__(self):
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize agents if available
        if AGENTS_AVAILABLE:
            self.segmentation_agent = SegmentationAgent()
            self.ensemble_agent = EnsembleAgent()
            # Initialize other agents as needed
        else:
            self.segmentation_agent = None
        
        # Track uploaded data
        self.rfm_data_loaded = False
        self.sales_data_loaded = False
        self.temp_files = []
    
    def update_output_files(self):
        """Get list of files in output directory"""
        files = []
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, file)
                if os.path.isfile(file_path):
                    files.append(file)
        return files
    
    def get_file_path(self, filename):
        """Get full path to a file in the output directory"""
        path = os.path.join(self.output_dir, filename)
        if os.path.isfile(path):
            return path
        return None
    
    def cleanup(self):
        """Clean up temporary files"""
        for file in self.temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error removing file {file}: {e}")
        self.temp_files = []

# Create state
state = AppState()

def save_upload_file(file_obj):
    """Save uploaded file to temp location and return path"""
    if file_obj is None:
        return None
    
    # Create temp file with same extension
    suffix = Path(file_obj.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_obj.read())
        state.temp_files.append(tmp.name)
        return tmp.name

def process_message(message, rfm_file, sales_file, chat_history):
    """Process user message and data files"""
    response = ""
    
    # Process file uploads
    if rfm_file is not None and not state.rfm_data_loaded:
        rfm_path = save_upload_file(rfm_file)
        try:
            success = state.segmentation_agent.load_data(rfm_path)
            if success:
                response += "✅ RFM data loaded successfully!\n\n"
                state.rfm_data_loaded = True
            else:
                response += "❌ Failed to load RFM data. Please check the format.\n\n"
        except Exception as e:
            response += f"❌ Error loading RFM data: {str(e)}\n\n"
    
    if sales_file is not None and not state.sales_data_loaded:
        sales_path = save_upload_file(sales_file)
        # Here you would process sales data with appropriate agent
        response += "✅ Sales data received! (Sales analysis features coming soon)\n\n"
        state.sales_data_loaded = True
    
    # Process user queries
    if not message:
        if not response:
            response = "Hello! I'm your FACETS assistant. How can I help you today?"
    elif "segment" in message.lower():
        if not state.rfm_data_loaded:
            response += "Please upload RFM data first before performing segmentation."
        elif state.segmentation_agent:
            try:
                # Perform segmentation analysis
                results = state.segmentation_agent.perform_segmentation()
                
                # Generate report and visualizations
                report_path = state.segmentation_agent.generate_report()
                viz_path = state.segmentation_agent.plot_segments()
                
                # Create response with insights
                response += f"✨ Customer segmentation complete! Found {len(results['segments'])} distinct segments.\n\n"
                response += "📊 Key segments:\n"
                for seg_id, segment in results['segments'].items():
                    response += f"- {segment['segment_name']}: {segment['count']} customers ({segment['percentage']:.1f}%)\n"
                
                response += "\n📄 Generated files are available in the 'Download Files' section below."
            except Exception as e:
                response += f"❌ Error during segmentation: {str(e)}"
    elif "help" in message.lower():
        response = """
# How to use this FACETS assistant:

1. **Upload your data files:**
   - RFM data (Customer ID, Recency, Frequency, Monetary)
   - Sales data (Store ID, Day of Week, Date, Sales, Customers, Promotion)

2. **Ask me questions like:**
   - "Perform customer segmentation" - I'll analyze your RFM data into customer segments
   - "What are my best customer segments?" - After segmentation, I can provide insights

3. **Download generated files:**
   - Reports (markdown format)
   - Visualizations (PNG images)
   - Use the refresh button to see newly generated files

What would you like me to help with?
"""
    else:
        response = "I'm not sure how to help with that query. Try asking me to 'perform customer segmentation' or type 'help' to see available options."
    
    # Add to chat history
    chat_history.append((message, response))
    return "", chat_history

def generate_sample_data():
    """Generate sample RFM data for testing"""
    try:
        # Create sample RFM data
        np.random.seed(42)
        num_customers = 500
        
        # Create customer segments
        segment_sizes = [int(num_customers * p) for p in [0.2, 0.3, 0.25, 0.25]]
        # Adjust last segment to ensure total is correct
        segment_sizes[-1] = num_customers - sum(segment_sizes[:-1])
        
        # Generate data
        customer_ids = [f"CUST{i:05d}" for i in range(1, num_customers + 1)]
        recency, frequency, monetary = [], [], []
        
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
        
        # Create DataFrame
        data = {
            'Customer ID': customer_ids,
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary
        }
        
        df = pd.DataFrame(data)
        
        # Save to output directory
        output_path = os.path.join(state.output_dir, "sample_rfm_data.csv")
        df.to_csv(output_path, index=False)
        
        return f"✅ Sample RFM data created! You can download it from the 'Download Files' section."
    except Exception as e:
        return f"❌ Error creating sample data: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# FACETS Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Upload Your Data")
            with gr.Tab("RFM Data"):
                rfm_file = gr.File(label="Upload RFM Data (CSV/Parquet)")
                gr.Markdown("*Should contain: Customer ID, Recency, Frequency, Monetary*")
            
            with gr.Tab("Sales Data"):
                sales_file = gr.File(label="Upload Sales Data (CSV/Parquet)")
                gr.Markdown("*Should contain: Store ID, Day, Date, Sales, Customers, Promotion*")
            
            sample_btn = gr.Button("Generate Sample RFM Data")
            sample_output = gr.Markdown("")
            
            gr.Markdown("## Download Files")
            file_list = gr.Dropdown(
                label="Available files", 
                choices=state.update_output_files(), 
                interactive=True
            )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh")
                download_btn = gr.Button("📥 Download")
            file_output = gr.File(label="Download", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat with Analytics Assistant",
                height=600,
                bubble=True
            )
            msg = gr.Textbox(
                label="Ask a question",
                placeholder="Try 'perform customer segmentation' or 'help'"
            )
            with gr.Row():
                submit_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear Chat")
    
    # Set up event handlers
    submit_btn.click(process_message, [msg, rfm_file, sales_file, chatbot], [msg, chatbot])
    msg.submit(process_message, [msg, rfm_file, sales_file, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot)
    refresh_btn.click(lambda: gr.Dropdown.update(choices=state.update_output_files()), None, file_list)
    download_btn.click(lambda x: state.get_file_path(x), [file_list], [file_output])
    sample_btn.click(generate_sample_data, None, sample_output)
    
    # Initialize with welcome message
    app.load(lambda: process_message("", None, None, []), [msg, rfm_file, sales_file, chatbot], [msg, chatbot])

# Launch the app
if __name__ == "__main__":
    app.launch()