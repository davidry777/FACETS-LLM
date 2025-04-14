import modal
from modal import App, Volume, Image

# Setup - define our infrastructure with code!

app = modal.App("pricer-service")
volume = Volume.persisted("retail-model-cache")  # Local persistent storage
image = Image.debian_slim().pip_install(
    "torch", 
    "transformers", 
    "bitsandbytes", 
    "accelerate", 
    "peft", 
    "pandas", 
    "pyarrow", 
    "datasets"
)

# Constants

GPU = "T4"  # GPU type
BASE_MODEL = "deepseek-ai/deepseek-coder-7b-instruct"  # Free, open-source DeepSeek model
MODEL_DIR = "/cache/models/"  # Local storage for models
BASE_DIR = MODEL_DIR + "deepseek-coder-7b-instruct"
LOCAL_MODEL_DIR = "/cache/results/retail-price-model"  # Where fine-tuned model will be stored
DATASET_PATH = "/cache/data/online_retail_cleaned.parquet"  # Local path for dataset

# Prompt format
QUESTION = "How much does this retail item cost to the nearest dollar?"
PREFIX = "Price is $"


@app.cls(image=image, gpu=GPU, timeout=1800, volumes={"/cache": volume})
class Pricer:
    @modal.build()
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download
        import os
        
        # Create directories if they don't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs("/cache/data", exist_ok=True)
        os.makedirs("/cache/results", exist_ok=True)
        
        # Download base model from Hugging Face
        print(f"Downloading base model {BASE_MODEL}...")
        try:
            snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
            print("Base model downloaded successfully")
        except Exception as e:
            print(f"Error downloading base model: {e}")
            print("Will need to download at runtime")

    @modal.enter()
    def setup(self):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        
        # Quantization configuration for efficient loading
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    
        # Check if base model was downloaded during build phase
        if os.path.exists(BASE_DIR):
            # Load tokenizer from local storage
            print("Loading tokenizer from local storage...")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)
            
            # Load base model from local storage
            print("Loading base model from local storage...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_DIR, 
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            # Download model and tokenizer directly from Hugging Face
            print(f"Downloading tokenizer from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
            print(f"Downloading base model from Hugging Face...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, 
                quantization_config=quant_config,
                device_map="auto"
            )
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    
        # Try to load locally fine-tuned model if it exists
        if os.path.exists(LOCAL_MODEL_DIR):
            try:
                print("Loading locally fine-tuned model...")
                from peft import PeftConfig
                
                # First check if this is a PEFT model
                try:
                    PeftConfig.from_pretrained(LOCAL_MODEL_DIR)
                    self.model = PeftModel.from_pretrained(self.base_model, LOCAL_MODEL_DIR)
                    print("PEFT fine-tuned model loaded successfully")
                except Exception:
                    # Try loading as a full model
                    print("Loading as full model...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        LOCAL_MODEL_DIR,
                        quantization_config=quant_config,
                        device_map="auto"
                    )
                    print("Full fine-tuned model loaded successfully")
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")
                print("Using base model instead")
                self.model = self.base_model
        else:
            print("No fine-tuned model found, using base model")
            self.model = self.base_model
        
        print("Model and tokenizer ready")

    @modal.method()
    def price(self, description: str) -> float:
        """
        Predict the price of a retail item based on its description
        """
        import re
        import torch
        from transformers import set_seed
    
        # Set seed for reproducible results
        set_seed(42)
        
        # Format prompt for DeepSeek model's chat format
        prompt = f"<|im_start|>user\n{QUESTION}\n\n{description}<|im_end|>\n<|im_start|>assistant\n{PREFIX}"
        
        try:
            # Encode the prompt for the model
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            attention_mask = torch.ones(inputs.shape, device="cuda")
            
            # Generate a price prediction with controlled parameters
            outputs = self.model.generate(
                inputs, 
                attention_mask=attention_mask, 
                max_new_tokens=10,        # We only need a short response for the price
                num_return_sequences=1,   # Just one prediction
                temperature=0.1,          # Lower temperature for more precise pricing
                do_sample=False,          # Deterministic output for price
                repetition_penalty=1.2    # Avoid repeating text
            )
            
            # Decode the model output
            result = self.tokenizer.decode(outputs[0])
            print(f"Model output: {result}")
        
            # Extract the price using regex - try multiple patterns
            
            # Pattern 1: Extract after the PREFIX
            if PREFIX in result:
                contents = result.split(PREFIX)[1]
                contents = contents.replace(',','')
                match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
                if match:
                    return float(match.group())
            
            # Pattern 2: Look for price with dollar sign
            match = re.search(r"\$\s*([\d,]+\.?\d*|\d*\.?\d+)", result)
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 3: Find any number after "price" or "cost" keywords
            match = re.search(r"(?:price|cost|value)[^\d]*?([\d,]+\.?\d*|\d*\.?\d+)", result.lower())
            if match:
                return float(match.group(1).replace(',', ''))
            
            # Pattern 4: Find any number in the output as last resort
            match = re.search(r"[-+]?\d*\.\d+|\d+", result)
            if match:
                return float(match.group())
            
            # If no price found, return default
            print("No price found in output")
            return 0
            
        except Exception as e:
            print(f"Error during price prediction: {e}")
            # Return a fallback average price
            return 4.75

    @modal.method()
    def train_model(self, dataset_path=None) -> str:
        """
        Method to fine-tune the DeepSeek model on the retail dataset
        without requiring any API keys or credentials
        """
        import os
        import pandas as pd
        import torch
        from transformers import (
            TrainingArguments, 
            Trainer, 
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import datasets
        
        # Use provided dataset path or default
        if dataset_path is None:
            dataset_path = DATASET_PATH
            
        print(f"Loading dataset from {dataset_path}")
        try:
            df = pd.read_parquet(dataset_path)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return f"Error: Could not load dataset - {str(e)}"
        
        # Data statistics for logging
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
        print(f"Average price: ${df['Price'].mean():.2f}")
        
        # Prepare training data - sample for faster training
        print("Preparing training data")
        examples = []
        # Use a smaller sample if dataset is very large
        sample_size = min(10000, len(df))
        for _, row in df.sample(n=sample_size, random_state=42).iterrows():
            description = row['Description']
            price = row['Price']
            # Format for DeepSeek model
            instruction = f"{QUESTION}\n\n{description}"
            response = f"{PREFIX}{price:.2f}"
            examples.append({
                "instruction": instruction,
                "response": response
            })
        
        # Convert to HuggingFace dataset
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(examples))
        
        # Prepare prompts formatted for DeepSeek model
        def prepare_prompt(example):
            return {
                "text": f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
            }
        
        print("Tokenizing dataset")
        tokenized_dataset = dataset.map(
            lambda example: self.tokenizer(prepare_prompt(example), truncation=True, max_length=512),
            remove_columns=dataset.column_names
        )
        
        # Configure LoRA for efficient fine-tuning with minimal GPU memory
        print("Configuring LoRA")
        lora_config = LoraConfig(
            r=16,                # Rank
            lora_alpha=32,       # Scaling factor
            lora_dropout=0.05,   # Dropout probability
            bias="none",
            task_type="CAUSAL_LM",
            # Target modules specific to DeepSeek architecture
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Prepare model for fine-tuning
        print("Preparing model for fine-tuning")
        training_model = prepare_model_for_kbit_training(self.base_model)
        training_model = get_peft_model(training_model, lora_config)
        
        # Set up training arguments - save locally
        print("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir=LOCAL_MODEL_DIR,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            save_strategy="epoch",
            logging_steps=100,
            save_total_limit=1,        # Save disk space by keeping only best model
            push_to_hub=False,         # Don't push to HuggingFace (no credentials needed)
            report_to="none"           # Don't report to wandb or other services
        )
        
        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        print("Initializing trainer")
        trainer = Trainer(
            model=training_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Fine-tune the model
        print("Starting training...")
        trainer.train()
        
        # Save the fine-tuned model locally
        print(f"Saving model to {LOCAL_MODEL_DIR}")
        training_model.save_pretrained(LOCAL_MODEL_DIR)
        self.tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        
        # Update the model in memory
        print("Updating model in memory")
        self.model = training_model
        
        return f"Model fine-tuned and saved to {LOCAL_MODEL_DIR}"

    @modal.method()
    def upload_dataset(self, dataset_bytes) -> str:
        """
        Method to upload a dataset to the persistent volume
        """
        import os
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        
        try:
            # Save dataset
            with open(DATASET_PATH, 'wb') as f:
                f.write(dataset_bytes)
            
            # Verify the dataset was saved correctly
            if os.path.exists(DATASET_PATH):
                size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
                return f"Dataset uploaded to {DATASET_PATH} ({size_mb:.2f} MB)"
            else:
                return f"Error: Dataset file not found after upload"
        except Exception as e:
            return f"Error uploading dataset: {str(e)}"

    @modal.method()
    def analyze_dataset(self) -> dict:
        """
        Analyze the uploaded dataset to get basic statistics
        """
        import pandas as pd
        
        try:
            # Load the dataset
            df = pd.read_parquet(DATASET_PATH)
            
            # Calculate basic statistics
            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "price_stats": {
                    "min": float(df["Price"].min()),
                    "max": float(df["Price"].max()),
                    "mean": float(df["Price"].mean()),
                    "median": float(df["Price"].median())
                },
                "country_counts": df["Country"].value_counts().to_dict(),
                "sample_items": df["Description"].sample(5).tolist()
            }
            
            return stats
        except Exception as e:
            return {"error": str(e)}

    @modal.method()
    def wake_up(self) -> str:
        """
        Simple method to wake up the container and verify it's operational
        """
        import torch
        
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            gpu_info = "No GPU available"
            
        return f"Retail Price Specialist is awake and ready. {gpu_info}"
            
    @modal.method()
    def predict_batch(self, descriptions: list) -> list:
        """
        Predict prices for multiple items at once
        """
        results = []
        
        for desc in descriptions:
            price = self.price(desc)
            results.append({"description": desc, "price": price})
            
        return results
