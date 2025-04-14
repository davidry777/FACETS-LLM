import modal
from modal import App, Image



app = modal.App("pricer-service")
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft", "pandas", "pyarrow")



GPU = "T4"  
BASE_MODEL = "deepseek-ai/deepseek-coder-7b-instruct"  


@app.function(image=image, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    QUESTION = "How much does this retail item cost to the nearest dollar?"
    PREFIX = "Price is $"

   
    prompt = f"<|im_start|>user\n{QUESTION}\n\n{description}<|im_end|>\n<|im_start|>assistant\n{PREFIX}"
    
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        quantization_config=quant_config,
        device_map="auto"
    )

    # Generate price prediction
    set_seed(42)  
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask, 
        max_new_tokens=10, 
        num_return_sequences=1,
        temperature=0.1  # Lower temperature for more deterministic outputs
    )
    result = tokenizer.decode(outputs[0])

    
    try:
        if PREFIX in result:
            contents = result.split(PREFIX)[1]
            contents = contents.replace(',','')
            match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
            if match:
                return float(match.group())
        
        
        match = re.search(r"\$\s*([\d,]+\.?\d*|\d*\.?\d+)", result)
        if match:
            return float(match.group(1).replace(',', ''))
        
        
        match = re.search(r"[-+]?\d*\.\d+|\d+", result)
        return float(match.group()) if match else 0
    except Exception as e:
        print(f"Error extracting price: {e}")
        print(f"Raw output: {result}")
        return 0


@app.function(image=image, gpu=GPU, timeout=1800)
def train_model(dataset_path='online_retail_cleaned.parquet'):
    """
    Function to fine-tune the DeepSeek model on the retail dataset,
    saving results locally without requiring credentials.
    """
    import pandas as pd
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments, 
        Trainer, 
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import datasets
    import os
    
    
    output_dir = "./results/retail-price-model"
    os.makedirs(output_dir, exist_ok=True)
    
    
    try:
        df = pd.read_parquet(dataset_path)
        print(f"Successfully loaded dataset from {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return f"Error: Could not load dataset - {str(e)}"
    
    
    print("Preparing training data")
    examples = []
    for _, row in df.sample(n=min(10000, len(df)), random_state=42).iterrows():
        description = row['Description']
        price = row['Price']
        
        
        instruction = f"How much does this retail item cost to the nearest dollar?\n\n{description}"
        response = f"Price is ${price:.2f}"
        examples.append({
            "instruction": instruction,
            "response": response
        })
    
    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(examples))
    
    # Load tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    def prepare_prompt(example):
        return {
            "text": f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        }
    
    print("Tokenizing dataset")
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(prepare_prompt(example), truncation=True, max_length=512),
        remove_columns=dataset.column_names
    )
    
    
    print("Configuring LoRA for efficient fine-tuning")
    lora_config = LoraConfig(
        r=16,                 # Rank
        lora_alpha=32,        # Alpha parameter 
        lora_dropout=0.05,    # Dropout probability 
        bias="none",          
        task_type="CAUSAL_LM",  
        # Target the key modules for DeepSeek model architecture
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Quantization config for efficient model loading
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load and prepare model for fine-tuning
    print("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    print("Preparing model for training")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Training arguments - save locally without requiring HF upload
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=3,
        push_to_hub=False  
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    print("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Fine-tune the model
    print("Starting training")
    trainer.train()
    
    # Save the fine-tuned model locally
    print(f"Saving fine-tuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return f"Model fine-tuned and saved to {output_dir}"
