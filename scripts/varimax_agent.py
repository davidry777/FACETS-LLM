import os
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from scipy import stats
import warnings

from agent import Agent

class VarimaxAgent(Agent):
    """
    An agent that uses VARMAX time series model for sales prediction
    """
    name = "Varimax Agent"
    color = Agent.GREEN
    
    def __init__(self, model_path='./models/varimax_model.pkl'):
        """
        Initialize the Varimax agent with a trained model or create a simple one
        """
        self.log("Initializing Varimax Agent")
        
        try:
            # Try to load the trained model
            self.model = joblib.load(model_path)
            self.log("Loaded Varimax model from disk")
            self.has_model = True
        except Exception as e:
            self.log(f"Could not load Varimax model: {e}")
            self.log("Using simple time series forecasting")
            self.has_model = False
            
            # Create a simple fallback model (will be trained on first prediction)
            self.model = None
            self.baseline_sales = 14500  # Average sales from the provided dataset
            
            # Try to load store data if available
            try:
                self.store_data = pd.read_csv('rossmann_store_data.csv')
                self.log("Loaded Rossmann store data")
                
                # Calculate average sales per store
                if 'Avg_Sales' in self.store_data.columns:
                    self.store_averages = self.store_data.groupby('Store')['Avg_Sales'].mean().to_dict()
                    self.baseline_sales = self.store_data['Avg_Sales'].mean()
                    self.log(f"Calculated baseline sales: {self.baseline_sales:.2f}")
            except Exception as e:
                self.log(f"Could not load store data: {e}")
                self.store_data = None
                self.store_averages = {}
        
        self.log("Varimax Agent is ready")
    
    def _extract_store_features(self, description):
        """
        Extract store features from the description
        """
        features = {}
        
        # Try to extract store number
        store_match = re.search(r'store\s*(\d+)', description.lower())
        if store_match:
            features['store_id'] = int(store_match.group(1))
        
        # Try to extract date
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{2}\.\d{2}\.\d{4})'  # DD.MM.YYYY
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, description)
            if date_match:
                try:
                    date_str = date_match.group(1)
                    if '/' in date_str:
                        date = datetime.strptime(date_str, '%m/%d/%Y')
                    elif '.' in date_str:
                        date = datetime.strptime(date_str, '%d.%m.%Y')
                    else:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    features['date'] = date
                    features['month'] = date.month
                    features['year'] = date.year
                    features['day_of_week'] = date.weekday()
                    break
                except:
                    pass
        
        # Try to extract store type
        store_types = ['a', 'b', 'c', 'd']
        for st in store_types:
            if f"type {st}" in description.lower() or f"type: {st}" in description.lower():
                features['store_type'] = st
                break
        
        # Try to extract if there's a promotion
        if 'promo' in description.lower() or 'promotion' in description.lower():
            features['promo'] = 1
        else:
            features['promo'] = 0
            
        return features
    
    def _predict_with_model(self, store_id, date=None):
        """
        Make a prediction using the trained VARMAX model
        """
        if not self.has_model or self.model is None:
            # Return baseline prediction
            if store_id in self.store_averages:
                return self.store_averages[store_id]
            return self.baseline_sales
        
        try:
            # Extract features for prediction
            # This would need to be adapted based on how your model was trained
            # For now, return a simple prediction
            return self.model.predict(start=0, end=0)[0]
        except Exception as e:
            self.log(f"Error in model prediction: {e}")
            # Fall back to average
            if store_id in self.store_averages:
                return self.store_averages[store_id]
            return self.baseline_sales
    
    def _train_simple_model(self, store_id=None):
        """
        Train a simple time series model if we have data available
        """
        if self.store_data is None or 'Avg_Sales' not in self.store_data.columns:
            return None
            
        try:
            # Filter for the specific store if provided
            if store_id is not None and 'Store' in self.store_data.columns:
                store_data = self.store_data[self.store_data['Store'] == store_id]
                if len(store_data) == 0:
                    self.log(f"No data available for store {store_id}")
                    return None
            else:
                store_data = self.store_data
                
            # Simple AR(1) model for quick prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    store_data['Avg_Sales'].values,
                    order=(1, 0, 0),
                    seasonal_order=(0, 0, 0, 0)
                )
                fitted_model = model.fit(disp=False)
                return fitted_model
                
        except Exception as e:
            self.log(f"Error training simple model: {e}")
            return None
    
    def predict_sales(self, description):
        """
        Predict sales based on the store description
        """
        self.log(f"Analyzing store description: {description}")
        
        # Extract features from the description
        features = self._extract_store_features(description)
        
        # Get store ID if available
        store_id = features.get('store_id', None)
        
        # If we don't have a model and have store data, try to train a simple one
        if not self.has_model and self.model is None and store_id is not None:
            self.log(f"Training simple model for store {store_id}")
            self.model = self._train_simple_model(store_id)
            if self.model is not None:
                self.has_model = True
        
        # Make prediction using the model or fallback
        if store_id is not None:
            sales_prediction = self._predict_with_model(store_id, features.get('date', None))
        else:
            # Use the baseline if we can't identify the store
            sales_prediction = self.baseline_sales
            
            # Adjust based on features we could extract
            if 'store_type' in features:
                # Store type adjustments based on the dataset
                if features['store_type'] == 'a':
                    sales_prediction *= 1.1  # Type a stores tend to have higher sales
                elif features['store_type'] == 'b':
                    sales_prediction *= 1.0  # Type b stores are average
                elif features['store_type'] == 'c':
                    sales_prediction *= 0.9  # Type c stores tend to have lower sales
                elif features['store_type'] == 'd':
                    sales_prediction *= 1.2  # Type d stores tend to have highest sales
            
            if 'promo' in features and features['promo'] == 1:
                sales_prediction *= 1.15  # Promotions tend to increase sales by ~15%
                
            if 'day_of_week' in features:
                # Day of week adjustments (0=Monday, 6=Sunday)
                dow_factors = [1.0, 0.95, 0.9, 0.85, 1.1, 1.3, 0.5]  # Example factors
                sales_prediction *= dow_factors[features['day_of_week']]
        
        self.log(f"Varimax Agent predicts sales of {sales_prediction:.2f}")
        return sales_prediction
        
    def price(self, description):
        """
        Compatible interface with the price prediction interface
        """
        # For compatibility with the price agents, we'll use the same method name
        # but internally we're predicting sales
        return self.predict_sales(description)
