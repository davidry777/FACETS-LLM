import gradio as gr
import pandas as pd
import numpy as np
import os
import json
import shutil
import tempfile
from typing import Dict, Any, List
from pathlib import Path

# Safer path manipulation - don't modify sys.path directly
import sys
from os.path import join, dirname, abspath

# Add scripts directory to Python path safely
current_dir = dirname(abspath(__file__))
scripts_dir = join(current_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import agents with safer error handling
AGENTS_AVAILABLE = False
try:
    from scripts.segmentation_agent import SegmentationAgent
    from scripts.frontier_agent import FrontierAgent
    from scripts.retail_price_specialist_agent import RetailPriceSpecialistAgent
    from scripts.sales_specialist_agent import SalesSpecialistAgent
    from scripts.varimax_agent import VarimaxAgent
    from scripts.random_forest_agent import RandomForestAgent
    from scripts.ensemble_agent import EnsembleAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing agents: {str(e)}")

# Initialize global state
class AppState:
    def __init__(self):
        self.output_dir = join(current_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize agents with the output directory
        self.segmentation_agent = SegmentationAgent(output_dir=self.output_dir) if AGENTS_AVAILABLE else None
        self.ensemble_agent = EnsembleAgent() if AGENTS_AVAILABLE else None
        
        # Initialize price prediction agents
        self.retail_price_specialist_agent = RetailPriceSpecialistAgent() if AGENTS_AVAILABLE else None
        self.sales_specialist_agent = SalesSpecialistAgent() if AGENTS_AVAILABLE else None
        self.varimax_agent = VarimaxAgent() if AGENTS_AVAILABLE else None
        self.frontier_agent = FrontierAgent() if AGENTS_AVAILABLE else None
        self.random_forest_agent = RandomForestAgent() if AGENTS_AVAILABLE else None
        
        # Track uploaded data
        self.rfm_data_loaded = False
        self.sales_data_loaded = False
        self.temp_files = []
    
    def update_output_files(self):
        """Get list of files in output directory"""
        files = []
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                file_path = join(self.output_dir, file)
                if os.path.isfile(file_path):
                    files.append(file)
        return files
    
    def get_file_path(self, filename):
        """Get full path to a file in the output directory"""
        if filename is None:
            return None
        path = join(self.output_dir, filename)
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
        
    def predict_price(self, description):
        """Use ensemble agent to predict price for a product description"""
        if not AGENTS_AVAILABLE:
            return {"error": "Price prediction agents are not available"}
        
        results = {}
        
        # Get predictions from individual agents with error handling
        try:
            if self.retail_price_specialist_agent:
                specialist_price = self.retail_price_specialist_agent.price(description)
                results["retail_specialist"] = round(specialist_price, 2)
        except Exception as e:
            print(f"Retail Price Specialist agent error: {str(e)}")
            results["retail_specialist"] = None
            
        try:
            if self.sales_specialist_agent:
                sales_specialist_price = self.sales_specialist_agent.price(description)
                results["sales_specialist"] = round(sales_specialist_price, 2)
        except Exception as e:
            print(f"Sales Specialist agent error: {str(e)}")
            results["sales_specialist"] = None
            
        try:
            if self.varimax_agent:
                varimax_price = self.varimax_agent.price(description)
                results["varimax"] = round(varimax_price, 2)
        except Exception as e:
            print(f"Varimax agent error: {str(e)}")
            results["varimax"] = None
            
        try:
            if self.frontier_agent:
                frontier_price = self.frontier_agent.price(description)
                results["frontier"] = round(frontier_price, 2)
        except Exception as e:
            print(f"Frontier agent error: {str(e)}")
            results["frontier"] = None
            
        try:
            if self.random_forest_agent:
                rf_price = self.random_forest_agent.price(description)
                results["random_forest"] = round(rf_price, 2)
        except Exception as e:
            print(f"Random Forest agent error: {str(e)}")
            results["random_forest"] = None
            
        # Get ensemble prediction
        try:
            if self.ensemble_agent:
                ensemble_price = self.ensemble_agent.price(description)
                results["ensemble"] = round(ensemble_price, 2)
        except Exception as e:
            print(f"Ensemble agent error: {str(e)}")
            results["ensemble"] = None
            
        return results

# Create state
state = AppState()

def save_upload_file(file_obj):
    """Save uploaded file to temp location and return path"""
    if file_obj is None:
        return None
    
    try:
        # Get file name and create proper extension
        file_name = file_obj.name
        suffix = Path(file_name).suffix
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Handle different types of file objects from Gradio
            if hasattr(file_obj, 'read'):
                # File-like object with read method
                tmp.write(file_obj.read())
            elif hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                # Object with a name attribute pointing to a valid file
                with open(file_obj.name, 'rb') as f:
                    tmp.write(f.read())
            else:
                # Assume it's the file path as a string
                shutil.copy(str(file_obj), tmp.name)
                
            state.temp_files.append(tmp.name)
            return tmp.name
    except Exception as e:
        print(f"Error saving uploaded file: {str(e)}")
        return None

def process_message(message, rfm_file, sales_file, chat_history):
    """Process user message and data files"""
    # Initialize chat history if None
    if chat_history is None:
        chat_history = []
    
    response = ""
    
    # Process file uploads
    if rfm_file is not None and not state.rfm_data_loaded:
        rfm_path = save_upload_file(rfm_file)
        if rfm_path and state.segmentation_agent:
            try:
                success = state.segmentation_agent.load_data(rfm_path)
                if success:
                    response += "✅ RFM data loaded successfully!\n\n"
                    state.rfm_data_loaded = True
                else:
                    response += "❌ Failed to load RFM data. Please check the format.\n\n"
            except Exception as e:
                response += f"❌ Error loading RFM data: {str(e)}\n\n"
        else:
            response += "❌ Could not process RFM data file.\n\n"
    
    if sales_file is not None and not state.sales_data_loaded:
        sales_path = save_upload_file(sales_file)
        if sales_path:
            # Here you would process sales data with appropriate agent
            response += "✅ Sales data received! (Sales analysis features coming soon)\n\n"
            state.sales_data_loaded = True
        else:
            response += "❌ Could not process sales data file.\n\n"
    
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
    elif "price" in message.lower() or "cost" in message.lower() or "worth" in message.lower():
        # Extract product description for price prediction
        # Check for common formats like "How much is [product]" or "Price for [product]"
        description = message
        for prefix in ["how much is", "price for", "predict price for", "price of", "cost of", "value of"]:
            if prefix in message.lower():
                description = message.lower().split(prefix, 1)[1].strip()
                break
        
        # Get price predictions
        predictions = state.predict_price(description)
        
        if "error" in predictions:
            response += f"❌ {predictions['error']}"
        else:
            response += f"📊 **Price Predictions for**: \"{description}\"\n\n"
            
            if predictions.get("retail_specialist") is not None:
                response += f"- 🧠 Retail Price Specialist: ${predictions['retail_specialist']:.2f}\n"
            
            if predictions.get("sales_specialist") is not None:
                response += f"- 📈 Sales Specialist: ${predictions['sales_specialist']:.2f}\n"
            
            if predictions.get("varimax") is not None:
                response += f"- 📊 Varimax Agent: ${predictions['varimax']:.2f}\n"
            
            if predictions.get("frontier") is not None:
                response += f"- 🔍 Frontier Agent: ${predictions['frontier']:.2f}\n"
            
            if predictions.get("random_forest") is not None:
                response += f"- 🌲 Random Forest Agent: ${predictions['random_forest']:.2f}\n"
            
            response += "\n"
            
            if predictions.get("ensemble") is not None:
                response += f"**🌟 Ensemble Prediction: ${predictions['ensemble']:.2f}**"
            else:
                # Calculate a simple average if ensemble is not available
                available_prices = [p for p in [predictions.get('retail_specialist'),
                                              predictions.get('sales_specialist'),
                                              predictions.get('varimax'),
                                              predictions.get('frontier'), 
                                              predictions.get('random_forest')] 
                                  if p is not None]
                if available_prices:
                    avg_price = sum(available_prices) / len(available_prices)
                    response += f"**🌟 Average Prediction: ${avg_price:.2f}**"
                else:
                    response += "❌ No price predictions available"
    elif "help" in message.lower():
        response = """
# How to use this FACETS assistant:

1. **Upload your data files:**
   - RFM data (Customer ID, Recency, Frequency, Monetary)
   - Sales data (Store ID, Day of Week, Date, Sales, Customers, Promotion)

2. **Customer Segmentation:**
   - "Perform customer segmentation" - I'll analyze your RFM data into customer segments
   - "What are my best customer segments?" - After segmentation, I can provide insights
   - "Show me segment distribution" - I'll display how customers are distributed across segments
   - "Identify my high-value customers" - I'll highlight your most valuable customer segments

3. **Price Prediction:**
   - "Predict price for [product description]" - I'll estimate the price using multiple prediction models
   - "How much is [product description]?" - Get price estimates from different analytical approaches
   - "What should I charge for [product description]?" - Get pricing recommendations
   - "Price analysis for [product description]" - Get detailed price breakdown from multiple agents

4. **Download generated files:**
   - Reports (markdown format)
   - Visualizations (PNG images)
   - Data exports (CSV format)
   - Use the refresh button to see newly generated files

5. **Sample Data:**
   - Click "Generate Sample RFM Data" to create test data if you don't have your own

What would you like me to help with today?
"""
    else:
        response = "I'm not sure how to help with that query. Try asking me to 'perform customer segmentation', 'predict price for [product]', or type 'help' to see available options."
    
    # Add to chat history - use the proper messages format
    if message:  # Only add user message if it's not empty
        chat_history.append({"role": "user", "content": message})
    if response:  # Only add assistant response if it's not empty
        chat_history.append({"role": "assistant", "content": response})
    
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
        output_path = join(state.output_dir, "sample_rfm_data.csv")
        df.to_csv(output_path, index=False)
        
        return f"✅ Sample RFM data created! You can download it from the 'Download Files' section."
    except Exception as e:
        return f"❌ Error creating sample data: {str(e)}"

# Helper for resetting the chat
def reset_chat():
    return []

def refresh_file_list():
    """Fix for the Dropdown.update error"""
    files = state.update_output_files()
    return gr.Dropdown(choices=files)

# Create the Gradio interface
with gr.Blocks(theme="soft") as app:
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
            # Initialize chatbot with correct message format
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hello! I'm your FACETS assistant. How can I help you today?"}],
                label="Chat with Analytics Assistant",
                height=600,
                type='messages'
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
    clear_btn.click(reset_chat, None, chatbot)
    refresh_btn.click(refresh_file_list, None, file_list)
    download_btn.click(lambda x: state.get_file_path(x), [file_list], [file_output])
    sample_btn.click(generate_sample_data, None, sample_output)

# Launch the app
if __name__ == "__main__":
    app.launch(debug=False, show_error=True)