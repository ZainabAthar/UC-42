"""
test_florence_final_fixed.py - Florence-2 with proper decoding
"""

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import re

def clean_florence_output(text):
    """Clean up Florence-2 output"""
    # Remove special tokens like <loc_...>
    text = re.sub(r'<[^>]+>', '', text)
    # Remove repetitive number patterns
    text = re.sub(r'(\d+[-\/\\]){4,}', '', text)
    # Remove multiple dots
    text = re.sub(r'\.{2,}', '.', text)
    # Clean up spaces
    text = ' '.join(text.split())
    return text.strip()

def test_florence():
    print("\n" + "="*60)
    print("FLORENCE-2 - K-ELECTRIC BILL EXTRACTION")
    print("="*60)

    # Load model
    print("\n🔄 Loading Florence-2...")
    model_id = 'microsoft/Florence-2-base'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    print("✓ Model loaded!")

    # Your exact image path
    image_path = r"C:\Users\DELL\PycharmProjects\uc-42-v2\test_data\ke_bills\genuine\ke_bill_paid1.jpg"
    print(f"\n📷 Loading: {image_path}")

    try:
        image = Image.open(image_path)
        # Resize for faster processing
        image.thumbnail((1200, 1200))
        print(f"✓ Image loaded: {image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Define extraction tasks
    tasks = [
        ("Full OCR", "<OCR>"),
        ("Address", "What is the complete mailing address? Extract plot number, block, and area."),
        ("Customer Name", "What is the customer name?"),
        ("Account Number", "What is the 13-digit account number starting with 04?"),
        ("Amount Payable", "What is the amount payable in rupees?"),
        ("Due Date", "What is the due date?")
    ]

    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)

    for task_name, prompt in tasks:
        print(f"\n🔍 {task_name}:")
        print(f"   Prompt: {prompt}")
        print("-" * 40)

        # Prepare inputs
        inputs = processor(text=prompt, images=image, return_tensors='pt')

        # Generate with better parameters
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                max_new_tokens=200,
                num_beams=3,
                temperature=0.1,
                do_sample=False,
                early_stopping=True
            )

        # Decode properly (skip_special_tokens=True removes <s>, </s> etc.)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Clean up
        cleaned = clean_florence_output(generated_text)

        # Remove the prompt if it appears in the output
        if prompt in cleaned:
            cleaned = cleaned.replace(prompt, '').strip()

        print(f"📝 {cleaned}")

        # Special check for address
        if task_name == "Address":
            if any(k in cleaned.upper() for k in ['PLOT', 'HOUSE', 'BLOCK', 'H NO', 'FLAT', 'STREET']):
                print("✅ Address found!")
            else:
                print("⚠️ No address detected in output")

    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_florence()