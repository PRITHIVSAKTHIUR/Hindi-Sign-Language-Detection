import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Hindi-Sign-Language-Detection"  # Replace with actual path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Hindi label mapping
id2label = {
    "0": "ऋ", "1": "क", "2": "ख", "3": "ग", "4": "घ",
    "5": "ङ", "6": "च", "7": "छ", "8": "ज", "9": "झ",
    "10": "ट", "11": "ठ", "12": "ड", "13": "ण", "14": "त",
    "15": "थ", "16": "द", "17": "न", "18": "प", "19": "फ",
    "20": "ब", "21": "भ", "22": "म", "23": "य", "24": "र",
    "25": "ल", "26": "व", "27": "स", "28": "ह"
}

def classify_hindi_sign(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_hindi_sign,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3, label="Hindi Sign Classification"),
    title="Hindi-Sign-Language-Detection",
    description="Upload an image of a Hindi sign language hand gesture to identify the corresponding character."
)

if __name__ == "__main__":
    iface.launch()
