from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import os

# === Configuration ===
BASE_MODEL_NAME = "yaygomii/whisper-small-ta-fyp"
LORA_MODEL_NAME = "yaygomii/whisper-small-ta-peft-lora"
LOCAL_MODEL_DIR = "./local_model"

def download_and_save_model():
    """Download and save the Tamil Whisper model with PEFT adapter"""
    
    # Create local model directory
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    print(f"Created directory: {LOCAL_MODEL_DIR}")
    
    print("=== Downloading Tamil Whisper Model ===")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"LoRA adapter: {LORA_MODEL_NAME}")
    print(f"Saving to: {LOCAL_MODEL_DIR}")
    print()
    
    try:
        # Download processor
        print("Downloading processor...")
        processor = WhisperProcessor.from_pretrained(
            BASE_MODEL_NAME, 
            language="tamil", 
            task="transcribe"
        )
        print("✓ Processor downloaded")
        
        # Download base model
        print("Downloading base model...")
        base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
        print("✓ Base model downloaded")
        
        # Download PEFT model
        print("Downloading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_NAME)
        print("✓ PEFT adapter downloaded")
        
        # Save models locally with full paths
        processor_path = os.path.join(LOCAL_MODEL_DIR, "processor")
        model_path = os.path.join(LOCAL_MODEL_DIR, "model")
        
        print("Saving models locally...")
        processor.save_pretrained(processor_path)
        model.save_pretrained(model_path)
        print("✓ Models saved successfully!")
        
        print(f"\nModels saved to: {LOCAL_MODEL_DIR}")
        print("You can now use these models with the server.py script.")
        
        # Verify files were created
        files = os.listdir(LOCAL_MODEL_DIR)
        print(f"\nFiles in {LOCAL_MODEL_DIR}:")
        for file in files:
            file_path = os.path.join(LOCAL_MODEL_DIR, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  ✓ {file} ({size} bytes)")
            else:
                print(f"  - {file} (directory)")
        
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print("Please check:")
        print("1. Internet connection")
        print("2. Model names on Hugging Face Hub")
        print("3. Hugging Face CLI configuration")
        return False
    
    return True

if __name__ == "__main__":
    success = download_and_save_model()
    if success:
        print("\n=== Next Steps ===")
        print("1. Run the server: python server.py")
        print("2. Open your browser to: http://localhost:8000")
        print("3. Click 'Request Mic Access' to start recording")
    else:
        print("\nPlease fix the issues above and try again.")
