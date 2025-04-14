from typing import Optional
from transformers import AutoTokenizer
import re

# Updated to use DeepSeek model instead of Llama
BASE_MODEL = "deepseek-ai/deepseek-coder-7b-instruct" 
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """
    
    # Load tokenizer with error handling for better reliability
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Warning: Could not load DeepSeek tokenizer: {e}")
        print("Using fallback tokenization mechanisms")
        tokenizer = None
    
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include = False

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub_details(self):
        """
        Clean up the details string by removing common text that doesn't add value
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range,
        then set include to True
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            
            # Use tokenizer if available, otherwise fall back to character-based limits
            if self.tokenizer:
                try:
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > MIN_TOKENS:
                        tokens = tokens[:MAX_TOKENS]
                        text = self.tokenizer.decode(tokens)
                        self.make_prompt(text)
                        self.include = True
                except Exception as e:
                    print(f"Tokenization error: {e}")
                    self._fallback_parse(text)
            else:
                self._fallback_parse(text)
    
    def _fallback_parse(self, text):
        """
        Fallback parsing mechanism when tokenizer isn't available
        """
        # Simple character-based limit as fallback
        if len(text) > MIN_CHARS:
            self.make_prompt(text[:CEILING_CHARS//2])  # Conservative limit
            self.include = True
            self.token_count = len(text) // 4  # Rough token estimate

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        Format adjusted for DeepSeek model's expected chat format
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        
        # Calculate token count if tokenizer is available
        if self.tokenizer:
            try:
                self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))
            except:
                self.token_count = len(self.prompt) // 4  # Rough estimate
        else:
            self.token_count = len(self.prompt) // 4  # Rough estimate

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX
    
    def deepseek_chat_format(self):
        """
        Return the prompt formatted for DeepSeek's chat format
        """
        if not self.prompt:
            return None
        
        base_prompt = self.prompt.split(self.PREFIX)[0]
        return f"<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant\n{self.PREFIX}"

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return f"<{self.title} = ${self.price}>"
