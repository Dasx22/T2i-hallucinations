!pip install datasets torch torchvision torch_xla transformers datasets pandas requests


import requests
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import pandas as pd
import os

# Load the dataset
ds = load_dataset("NasrinImp/Defactify4_Train")

# Load the pre-trained model and processor
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

# Check if TPU is available
device = xm.xla_device() if 'COLAB_TPU_ADDR' in os.environ else torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize CSV file
output_file = "detailed_output.csv"
if not os.path.exists(output_file):
    df = pd.DataFrame(columns=["index", "ds_caption", "generated_caption"])
    df.to_csv(output_file, index=False)

# Process images from index 101 to 200
start_index = 0
end_index = 100
batch_size = 10

# Initialize a list to store results temporarily
results = []

for i in range(start_index, end_index + 1):
    try:
        # Access the dataset row
        item = ds['train'][i]
        ds_caption = item['caption']
        image = item['coco_image']  # Directly use the PIL.Image object

        # Define a more detailed prompt
        prompt = ("<grounding>An image that includes a detailed description of the scene, "
                  "objects, colors, actions, and any text present in the image. Mention "
                  "the setting, people, and notable features.")
        ## prompt = "<grounding>An image of"
        

        # Preprocess the image and text input
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        # Generate the caption
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=256,  # Increased token limit for more details
        )

        # Decode the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Post-process the text (clean and extract)
        processed_text, entities = processor.post_process_generation(generated_text)

        # Append the results to the list
        results.append({
            "index": i,
            "ds_caption": ds_caption,
            "generated_caption": processed_text
        })

        # Save checkpoint every 10 images
        if (i - start_index + 1) % batch_size == 0 or i == end_index:
            print(f"Processed {i - start_index + 1} images. Saving checkpoint...")
            df = pd.DataFrame(results)
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
            results = []  # Clear the results list after saving

    except Exception as e:
        print(f"Error processing image at index {i}: {e}")

print("Processing complete. Results saved to detailed_output.csv.")


!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html


import pandas as pd
new_data = pd.read_csv("/content/detailed_output.csv")
new_data["generated_caption"] = new_data["generated_caption"].str.replace("An image that includes a detailed description of the scene, objects, colors, actions, and any text present in the image. Mention the setting, people, and notable features.", "", regex=False)
new_data.drop(columns=["index"], inplace=True)
new_data.to_csv("100_summaries.csv")
