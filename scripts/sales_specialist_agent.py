import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

from agent import Agent

class SalesSpecialistAgent(Agent):
    """
    A specialist agent designed for sales prediction using
    domain knowledge of retail operations and store characteristics
    """
    name = "Sales Specialist Agent"
    color = Agent.PURPLE
    
    def __init__(self, store_data_path="rossmann_store_data.csv"):
        """Initialize the Sales Specialist Agent with store data"""
        self.log("Initializing Sales Specialist Agent")
        
        # Load store data if available
        try:
            self.store_data = pd.read_csv(store_data_path)
            self.log(f"Loaded store data with {len(self.store_data)} records")
            
            # Calculate store statistics
            self.store_averages = self.store_data.groupby('Store')['Avg_Sales'].mean().to_dict()
            
            if 'StoreType' in self.store_data.columns:
                self.type_averages = self.store_data.groupby('StoreType')['Avg_Sales'].mean().to_dict()
            else:
                self.type_averages = {}
                
            self.global_average = self.store_data['Avg_Sales'].mean()
            self.log(f"Calculated store statistics, global average: {self.global_average:.2f}")
            
        except Exception as e:
            self.log(f"Error loading store data: {e}")
            self.store_data = None
            self.store_averages = {}
            self.type_averages = {'a': 4500, 'b': 5000, 'c': 6000}
            self.global_average = 5000
        
        self.log("Sales Specialist Agent is ready")
    
    def _extract_features(self, description):
        """Extract key features from a store description"""
        features = {}
        
        # Store ID
        store_match = re.search(r'store\s*(\d+)', description.lower())
        if store_match:
            features['store_id'] = int(store_match.group(1))
        
        # Store type
        for st in ['a', 'b', 'c', 'd']:
            if f"type {st}" in description.lower():
                features['store_type'] = st
                break
        
        # Assortment
        if 'basic assortment' in description.lower():
            features['assortment'] = 'a'
        elif 'extended assortment' in description.lower():
            features['assortment'] = 'b'
        elif 'extra assortment' in description.lower():
            features['assortment'] = 'c'
        
        # Competition
        features['has_competition'] = 'competition' in description.lower()
        if features['has_competition']:
            dist_match = re.search(r'(\d+)\s*meters', description.lower())
            if dist_match:
                features['competition_distance'] = int(dist_match.group(1))
        
        # Promotions
        features['has_promo'] = 'promotion' in description.lower() or 'promo' in description.lower()
        
        # Date
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', description)
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                features['date'] = date
                features['month'] = date.month
                features['day_of_week'] = date.weekday()
                features['is_weekend'] = date.weekday() >= 5
            except:
                pass
        
        return features
    
    def predict_sales(self, description):
        """Predict sales based on store description"""
        self.log(f"Analyzing: {description}")
        
        # Extract features
        features = self._extract_features(description)
        self.log(f"Extracted features: {features}")
        
        # Base prediction
        if 'store_id' in features and features['store_id'] in self.store_averages:
            # If we know the store, use its average
            base_prediction = self.store_averages[features['store_id']]
            self.log(f"Using store-specific average: {base_prediction:.2f}")
        elif 'store_type' in features and features['store_type'] in self.type_averages:
            # If we know the store type, use type average
            base_prediction = self.type_averages[features['store_type']]
            self.log(f"Using store type average: {base_prediction:.2f}")
        else:
            # Otherwise use global average
            base_prediction = self.global_average
            self.log(f"Using global average: {base_prediction:.2f}")
        
        # Apply adjustments
        adjusted_prediction = base_prediction
        
        # Store type adjustment (if we didn't use it for base)
        if 'store_id' in features and 'store_type' in features:
            if features['store_type'] == 'a':
                adjusted_prediction *= 1.0  # baseline
            elif features['store_type'] == 'b':
                adjusted_prediction *= 1.1  # slightly higher
            elif features['store_type'] == 'c':
                adjusted_prediction *= 0.9  # slightly lower
            elif features['store_type'] == 'd':
                adjusted_prediction *= 1.2  # premium
        
        # Assortment effect
        if 'assortment' in features:
            if features['assortment'] == 'a':  # basic
                adjusted_prediction *= 1.0
            elif features['assortment'] == 'b':  # extended
                adjusted_prediction *= 1.1
            elif features['assortment'] == 'c':  # extra
                adjusted_prediction *= 1.2
        
        # Competition effect
        if features.get('has_competition', False):
            if 'competition_distance' in features:
                dist = features['competition_distance']
                if dist < 1000:
                    adjusted_prediction *= 0.9  # Nearby competition reduces sales
                elif dist > 10000:
                    adjusted_prediction *= 1.05  # Far competition has little effect
            else:
                adjusted_prediction *= 0.95  # Unknown distance - assume moderate effect
        else:
            adjusted_prediction *= 1.05  # No competition is good
        
        # Promotion effect
        if features.get('has_promo', False):
            adjusted_prediction *= 1.15  # Promotions boost sales
        
        # Time effects
        if 'day_of_week' in features:
            dow = features['day_of_week']
            if dow == 0:  # Monday
                adjusted_prediction *= 1.0
            elif dow == 1:  # Tuesday
                adjusted_prediction *= 0.95
            elif dow == 2:  # Wednesday
                adjusted_prediction *= 0.9
            elif dow == 3:  # Thursday
                adjusted_prediction *= 0.95
            elif dow == 4:  # Friday
                adjusted_prediction *= 1.1
            elif dow == 5:  # Saturday
                adjusted_prediction *= 1.2
            elif dow == 6:  # Sunday
                adjusted_prediction *= 0.7  # Many stores closed
        
        # Seasonal effects
        if 'month' in features:
            month = features['month']
            if month == 12:  # December holiday boost
                adjusted_prediction *= 1.2
            elif month == 1:  # January slump
                adjusted_prediction *= 0.85
            elif month == 7 or month == 8:  # Summer
                adjusted_prediction *= 1.1
        
        # Add small random variation
        adjusted_prediction *= np.random.uniform(0.98, 1.02)
        
        self.log(f"Final prediction: {adjusted_prediction:.2f}")
        return adjusted_prediction
    
    def price(self, description):
        """Compatibility method with price prediction interface"""
        return self.predict_sales(description)
