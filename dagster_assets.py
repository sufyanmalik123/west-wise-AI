from dagster import asset, MaterializeResult
import pandas as pd

@asset
def load_google_ads_data():
    # Read CSV
    df = pd.read_csv("google_ads_campaign_data.csv")
    
    # WRITE to new file
    df.to_csv("loaded_data.csv", index=False)
    
    return MaterializeResult(
        metadata={
            "num_records": len(df),
            "output_file": "loaded_data.csv"
        }
    )

@asset(deps=[load_google_ads_data])
def check_phi2_model():
    """Check if Phi-2 model is available"""
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_found = False
    
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if 'phi-2' in item.name.lower():
                model_found = True
                break
    
    return MaterializeResult(
        metadata={
            "model_cached": model_found,
            "cache_location": str(cache_dir)
        }
    )

@asset(deps=[check_phi2_model])
def analyze_keywords_with_phi2():
    """Run Phi-2 analysis on keywords"""
    from phi2_keyword_analyzer import GoogleAdsKeywordAnalyzer
    
    # Initialize and run analyzer
    analyzer = GoogleAdsKeywordAnalyzer()
    results = analyzer.run_analysis("loaded_data.csv")
    
    return MaterializeResult(
        metadata={
            "keywords_analyzed": len(results),
            "output_file": "eligible_keywords_conversions_25plus.csv"
        }
    )



@asset(deps=[analyze_keywords_with_phi2])
def validate_and_finetune():
    """Validate results and fine-tune model if needed"""
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset
    
    # Read output
    df = pd.read_csv("eligible_keywords_conversions_25plus.csv")
    
    # Validate
    invalid_rows = df[df['Conversions'] < 25]
    
    if len(invalid_rows) > 0:
        print(f"⚠️ Found {len(invalid_rows)} invalid keywords. Fine-tuning model...")
        
        # Prepare training data
        correct_df = pd.read_csv("google_ads_campaign_data.csv")
        correct_df['Conversions'] = correct_df['Conversions'].astype(int)
        
        # Create training examples
        train_texts = []
        for _, row in correct_df.iterrows():
            if row['Conversions'] >= 25:
                train_texts.append(f"Keyword: {row['Keyword']}, Conversions: {row['Conversions']} - ELIGIBLE")
            else:
                train_texts.append(f"Keyword: {row['Keyword']}, Conversions: {row['Conversions']} - NOT ELIGIBLE")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Replace the tokenize function and dataset creation:

        def tokenize(examples):
            tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
            tokenized['labels'] = tokenized['input_ids'].copy()  # Add this line
            return tokenized

        dataset = Dataset.from_dict({"text": train_texts})
        tokenized = dataset.map(tokenize, batched=True)
        
        # Training config
        training_args = TrainingArguments(
            output_dir="./finetuned_phi2",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=10,
            logging_steps=10,
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
        )
        trainer.train()
        
        # Save fine-tuned model
        model.save_pretrained("./finetuned_phi2")
        tokenizer.save_pretrained("./finetuned_phi2")
        
        # Fix output with corrected data
        valid_df = correct_df[correct_df['Conversions'] >= 25][['Keyword', 'Conversions']]
        valid_df.to_csv("finetuned_keywords.csv", index=False)
        
        return MaterializeResult(
            metadata={
                "status": "FINE-TUNED",
                "invalid_found": len(invalid_rows),
                "model_saved": "./finetuned_phi2",
                "output": "finetuned_keywords.csv"
            }
        )
    
    return MaterializeResult(
        metadata={
            "status": "PASSED",
            "keywords": len(df)
        }
    )