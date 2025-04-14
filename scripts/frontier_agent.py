# imports

import os
import re
import math
import json
from typing import List, Dict
import requests
from sentence_transformers import SentenceTransformer
from agent import Agent

# Optional imports that may not be available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class FrontierAgent(Agent):
    """
    An agent that uses LLMs (OpenAI, DeepSeek, or other) with RAG to estimate prices
    """
    name = "Frontier Agent"
    color = Agent.BLUE

    DEFAULT_MODEL = "deepseek-coder-7b-instruct"  # Changed to DeepSeek as default
    
    def __init__(self, collection=None):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing Frontier Agent")
        
        # Set up the LLM client
        self.model_name = os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self.client = self._setup_client()
        
        # Set up the vector store
        self.collection = collection
        
        # Set up the embedding model
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.log("Sentence transformer loaded successfully")
        except Exception as e:
            self.log(f"Error loading sentence transformer: {e}")
            self.model = None
            
        self.log("Frontier Agent is ready")

    def _setup_client(self):
        """
        Set up the client for the appropriate LLM service based on environment variables
        """
        # Check for DeepSeek API key
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            if OPENAI_AVAILABLE:
                client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
                self.model_name = "deepseek-chat"
                self.log("Frontier Agent is set up with DeepSeek API")
                return client
            else:
                self.log("OpenAI package not installed, can't use DeepSeek API")
        
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and OPENAI_AVAILABLE:
            client = OpenAI(api_key=openai_api_key)
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.log(f"Frontier Agent is set up with OpenAI model: {self.model_name}")
            return client
        
        # If no valid API setup, use local inference
        self.log("No valid API keys found, using local inference mode")
        return None

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI/DeepSeek
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by the API
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        
        # Format differently for DeepSeek models
        if "deepseek" in self.model_name.lower():
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "Price is $"}
            ]
        else:
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "Price is $"}
            ]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log("Frontier Agent is searching for similar products")
        
        if self.collection is None or self.model is None:
            self.log("Vector search unavailable - using fallback")
            return self._fallback_similars(description)
            
        try:
            vector = self.model.encode([description])
            results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
            documents = results['documents'][0][:]
            prices = [m['price'] for m in results['metadatas'][0][:]]
            self.log("Frontier Agent has found similar products")
            return documents, prices
        except Exception as e:
            self.log(f"Error during vector search: {e}")
            return self._fallback_similars(description)
            
    def _fallback_similars(self, description: str):
        """
        Provide fallback similar products when vector search is unavailable
        """
        # Create some generic similar products based on keywords in the description
        desc_lower = description.lower()
        similars = []
        prices = []
        
        # Generic items and prices based on common categories
        if any(word in desc_lower for word in ['glass', 'cup', 'mug', 'plate']):
            similars.extend([
                "SET OF 4 BLUE GLASS MUGS",
                "CERAMIC WHITE COFFEE CUP",
                "VINTAGE GLASS TUMBLER SET"
            ])
            prices.extend([12.99, 8.50, 15.75])
            
        if any(word in desc_lower for word in ['christmas', 'holiday', 'decoration']):
            similars.extend([
                "CHRISTMAS TREE DECORATION SET",
                "HOLIDAY HANGING ORNAMENTS"
            ])
            prices.extend([7.99, 9.25])
            
        if any(word in desc_lower for word in ['box', 'storage', 'wooden']):
            similars.extend([
                "WOODEN STORAGE BOX WITH LID",
                "SMALL TRINKET BOX"
            ])
            prices.extend([14.50, 6.75])
            
        # Fill with generic items if needed
        generic_items = [
            "VINTAGE METAL SIGN",
            "CERAMIC FLOWER POT",
            "DECORATIVE WALL HANGING",
            "SET OF SCENTED CANDLES",
            "METAL STORAGE BASKET"
        ]
        generic_prices = [11.99, 8.50, 15.25, 12.00, 9.75]
        
        # Ensure we have at least 5 items
        while len(similars) < 5 and generic_items:
            similars.append(generic_items.pop(0))
            prices.append(generic_prices.pop(0))
            
        self.log("Using fallback similar products")
        return similars, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def _local_inference(self, description: str, similars: List[str], prices: List[float]) -> float:
        """
        Provide a local price estimate when API isn't available
        This is a simple heuristic based on similar items
        """
        # Calculate a price based on similar items (if available)
        if similars and prices:
            # Average of similar prices with more weight to the first (most similar) ones
            weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(prices)]
            weighted_sum = sum(p * w for p, w in zip(prices, weights))
            weighted_avg = weighted_sum / sum(weights[:len(prices)])
            
            # Adjust based on description length and keywords
            premium_keywords = ['luxury', 'vintage', 'special', 'handmade', 'set', 'collection']
            budget_keywords = ['small', 'mini', 'basic', 'simple']
            
            desc_lower = description.lower()
            premium_count = sum(1 for word in premium_keywords if word in desc_lower)
            budget_count = sum(1 for word in budget_keywords if word in desc_lower)
            
            adjustment = (premium_count - budget_count) * 0.15  # 15% adjustment per keyword
            
            final_price = weighted_avg * (1 + adjustment)
            return max(0.99, final_price)  # Ensure minimum price
        else:
            # Fallback to basic estimate
            return 9.99  # Default price

    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, prices = self.find_similars(description)
        
        # If we have a client, use the API
        if self.client:
            try:
                self.log(f"Frontier Agent is calling {self.model_name} with similar product context")
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=self.messages_for(description, documents, prices),
                    seed=42,
                    max_tokens=5
                )
                reply = response.choices[0].message.content
                result = self.get_price(reply)
                self.log(f"Frontier Agent completed - predicting ${result:.2f}")
                return result
            except Exception as e:
                self.log(f"API call failed: {e}")
                self.log("Falling back to local inference")
                result = self._local_inference(description, documents, prices)
                self.log(f"Frontier Agent fallback - predicting ${result:.2f}")
                return result
        else:
            # Use local inference if no API client
            self.log("Using local inference (no API client)")
            result = self._local_inference(description, documents, prices)
            self.log(f"Frontier Agent local inference - predicting ${result:.2f}")
            return result
