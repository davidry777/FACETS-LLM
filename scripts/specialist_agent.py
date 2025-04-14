import modal
from agent import Agent


class SpecialistAgent(Agent):
    """
    An Agent that runs the DeepSeek model for price prediction
    """

    name = "Retail Price Specialist"
    color = Agent.BLUE

    def __init__(self):
        """
        Set up this Agent by connecting to the Modal service
        """
        self.log("Retail Price Specialist is initializing - connecting to modal")
        
     
        try:
            self.price_func = modal.Function.lookup("pricer-service", "price")
            self.use_class = False
            self.log("Retail Price Specialist is ready (function mode)")
        except Exception as e:
            self.log(f"Could not connect to price function: {e}")
            self.log("Using local fallback mode")
            self.use_class = None  # No connection
        
    def price(self, description: str) -> float:
        """
        Predict the price of a retail item based on its description
        """
        self.log(f"Analyzing: {description}")
        
        # Try the remote prediction if connected
        if self.use_class is not None:
            try:
                # Call the function-based implementation on Modal
                result = self.price_func.remote(description)
                self.log(f"Price prediction: ${result:.2f}")
                return result
            except Exception as e:
                self.log(f"Remote prediction error: {e}")
                self.log("Falling back to local estimation")
        

        words = description.lower().split()
        
        # Base price
        estimated_price = 5.00
        
        
        if len(words) > 10:
            estimated_price += 2.50
        
        
        premium_keywords = ['gold', 'silver', 'set', 'vintage', 'large', 'luxury', 'wooden', 
                          'glass', 'limited', 'edition', 'special', 'handmade', 'ceramic']
        
        
        keyword_matches = sum(1 for word in words if word in premium_keywords)
        estimated_price += keyword_matches * 1.25
        
        # Cap at reasonable range for retail dataset
        estimated_price = max(1.00, min(estimated_price, 50.00))
        
        self.log(f"Local price estimate: ${estimated_price:.2f}")
        return estimated_price
    
    def train_model(self, dataset_path='online_retail_cleaned.parquet') -> str:
        """
        Trigger model fine-tuning process with the specified dataset
        """
        self.log("Initiating model training")
        
        try:
            
            train_func = modal.Function.lookup("pricer-service", "train_model")
            
            result = train_func.remote(dataset_path)
            self.log(f"Training completed: {result}")
            return result
        except Exception as e:
            self.log(f"Training failed: {e}")
            return f"Could not train model: {str(e)}"
    
    def get_item_details(self, description: str) -> dict:
        """
        Get comprehensive details about a retail item based on its description
        """
        self.log(f"Analyzing item: {description}")
        
        # Get the price prediction
        price = self.price(description)
        
        
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['glass', 'cup', 'mug', 'dish', 'plate', 'bowl']):
            category = "Kitchenware"
        elif any(word in description_lower for word in ['christmas', 'heart', 'gift', 'decoration', 'card']):
            category = "Seasonal & Gifts"
        elif any(word in description_lower for word in ['sign', 'frame', 'box', 'candle', 'holder']):
            category = "Home Decor"
        elif any(word in description_lower for word in ['bag', 'hanger', 'storage', 'box']):
            category = "Storage & Organization"
        else:
            category = "General Merchandise"
        
        
        if price < 3.00:
            popularity = "High"
        elif price > 15.00:
            popularity = "Low"
        else:
            popularity = "Medium"
            
        return {
            "description": description,
            "price": price,
            "estimated_stock": "Medium",  # Default value
            "category": category,
            "popularity": popularity
        }
