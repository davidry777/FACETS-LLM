import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

from agent import Agent
from specialist_agent import SpecialistAgent
from frontier_agent import FrontierAgent
from random_forest_agent import RandomForestAgent

class EnsembleAgent(Agent):
    """
    An ensemble model that combines predictions from multiple price prediction agents
    """
    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, collection=None, model_path='./models/ensemble_model.pkl'):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        
        # Initialize all component agents
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.random_forest = RandomForestAgent()
        
        # Try to load the ensemble model
        try:
            self.model = joblib.load(model_path)
            self.log("Loaded ensemble weights from disk")
        except Exception as e:
            self.log(f"Could not load ensemble model: {e}")
            self.log("Using default equal weights")
            self.model = None
            
        self.log("Ensemble Agent is ready")

    def _default_ensemble(self, predictions):
        """
        Default ensemble method when model weights aren't available
        """
        # Simple median with outlier removal
        sorted_preds = sorted(predictions)
        
        # If we have 3 or more predictions, remove the highest and lowest
        if len(sorted_preds) >= 3:
            sorted_preds = sorted_preds[1:-1]
            
        # Return the average of remaining predictions
        return sum(sorted_preds) / len(sorted_preds)

    def price(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - gathering predictions from all agents")
        
        # Get predictions from each agent with error handling
        try:
            specialist_price = self.specialist.price(description)
            self.log(f"Specialist Agent prediction: ${specialist_price:.2f}")
        except Exception as e:
            self.log(f"Error from Specialist Agent: {e}")
            specialist_price = None
            
        try:
            frontier_price = self.frontier.price(description)
            self.log(f"Frontier Agent prediction: ${frontier_price:.2f}")
        except Exception as e:
            self.log(f"Error from Frontier Agent: {e}")
            frontier_price = None
            
        try:
            random_forest_price = self.random_forest.price(description)
            self.log(f"Random Forest Agent prediction: ${random_forest_price:.2f}")
        except Exception as e:
            self.log(f"Error from Random Forest Agent: {e}")
            random_forest_price = None
            
        # Collect all valid predictions
        valid_predictions = [p for p in [specialist_price, frontier_price, random_forest_price] if p is not None]
        
        if not valid_predictions:
            self.log("No valid predictions from any agent - using fallback")
            # Very basic fallback
            return 9.99
            
        # If we have the ensemble model, use it
        if self.model is not None:
            # Fill in missing values with the mean of valid predictions
            mean_price = sum(valid_predictions) / len(valid_predictions)
            specialist_price = specialist_price if specialist_price is not None else mean_price
            frontier_price = frontier_price if frontier_price is not None else mean_price
            random_forest_price = random_forest_price if random_forest_price is not None else mean_price
            
            # Create features for the ensemble model
            X = pd.DataFrame({
                'Specialist': [specialist_price],
                'Frontier': [frontier_price],
                'RandomForest': [random_forest_price],
                'Min': [min(specialist_price, frontier_price, random_forest_price)],
                'Max': [max(specialist_price, frontier_price, random_forest_price)],
            })
            
            # Make prediction and ensure it's positive
            prediction = max(0, self.model.predict(X)[0])
            self.log(f"Ensemble Agent (weighted model) - predicting ${prediction:.2f}")
            return prediction
        else:
            # Use default ensemble method
            prediction = self._default_ensemble(valid_predictions)
            self.log(f"Ensemble Agent (default average) - predicting ${prediction:.2f}")
            return prediction
