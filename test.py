from huggingface_hub import hf_hub_download
import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()

huggingface_token = os.getenv('HF_TOKEN')

login(huggingface_token)

file = hf_hub_download(
    repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

# Prediction task with 7 input channels and prediction length is 96
config = PatchTSTConfig.from_pretrained("namctin/patchtst_etth1_forecast")
config.prediction_length = 336
model = PatchTSTForPrediction(config)


# during training, one provides both past and future values
outputs = model(
    past_values=batch["past_values"],
    future_values=batch["future_values"],
)

loss = outputs.loss
loss.backward()

# during inference, one only provides past values, the model outputs future values
outputs = model(past_values=batch["past_values"])
prediction_outputs = outputs.prediction_outputs
print(prediction_outputs)
