
![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/uy6R1tGM1bgalpyRPwvZu.png)

# Hindi-Sign-Language-Detection

> Hindi-Sign-Language-Detection is a vision-language model fine-tuned from google/siglip2-base-patch16-224 for multi-class image classification. It is trained to detect and classify Hindi sign language hand gestures into corresponding Devanagari characters using the SiglipForImageClassification architecture.

```py
Classification Report:
              precision    recall  f1-score   support

           ऋ     0.9832    0.9121    0.9463       512
           क     0.9433    0.9357    0.9395       498
           ख     0.9694    0.9589    0.9641       462
           ग     0.9961    0.8996    0.9454       568
           घ     0.8990    0.9784    0.9370       464
           ङ     0.9758    0.9869    0.9813       612
           च     0.9223    0.9519    0.9368       561
           छ     0.9226    0.9597    0.9408       571
           ज     0.9346    0.9709    0.9524       412
           झ     0.9051    0.9978    0.9492       449
           ट     0.9670    0.8998    0.9322       489
           ठ     0.8992    0.9954    0.9449       439
           ढ     0.9392    0.9984    0.9679       634
           ण     0.9102    0.9383    0.9240       648
           त     0.8167    0.9938    0.8966       650
           थ     0.9720    0.9616    0.9668       651
           द     0.8162    0.9185    0.8643       319
           न     0.9711    0.8971    0.9327       525
           प     0.9642    0.9360    0.9499       719
           फ     0.9847    0.7700    0.8642       500
           ब     0.9447    0.9364    0.9406       566
           भ     0.8779    0.9656    0.9197       581
           म     0.9968    0.9920    0.9944       624
           य     0.9600    0.9829    0.9713       586
           र     0.9613    0.9268    0.9437       724
           ल     0.9719    0.8993    0.9342       576
           व     0.9619    0.8547    0.9052       709
           स     1.0000    0.9721    0.9859       502
           ह     0.9899    0.9441    0.9665       626

    accuracy                         0.9425     16177
   macro avg     0.9433    0.9426    0.9413     16177
weighted avg     0.9457    0.9425    0.9425     16177
```

---

## Label Space: 29 Classes

The model classifies a hand sign into one of the following 29 Hindi characters:

```json
"id2label": {
  "0": "ऋ",
  "1": "क",
  "2": "ख",
  "3": "ग",
  "4": "घ",
  "5": "ङ",
  "6": "च",
  "7": "छ",
  "8": "ज",
  "9": "झ",
  "10": "ट",
  "11": "ठ",
  "12": "ड",
  "13": "ण",
  "14": "त",
  "15": "थ",
  "16": "द",
  "17": "न",
  "18": "प",
  "19": "फ",
  "20": "ब",
  "21": "भ",
  "22": "म",
  "23": "य",
  "24": "र",
  "25": "ल",
  "26": "व",
  "27": "स",
  "28": "ह"
}
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio
```

---

## Inference Code

```python
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
```

---

## Intended Use

Hindi-Sign-Language-Detection can be used in:

* Educational tools for learning Indian sign language.
* Assistive technology for hearing and speech-impaired individuals.
* Real-time sign-to-text translation applications.
* Human-computer interaction for Hindi users. 
