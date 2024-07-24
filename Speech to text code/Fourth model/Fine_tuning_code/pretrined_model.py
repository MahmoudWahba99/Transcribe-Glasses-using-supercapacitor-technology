import os
import pandas as pd
from datasets import Dataset, load_metric
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
import torch
import matplotlib.pyplot as plt
import shutil

# Set CUDA_LAUNCH_BLOCKING for accurate error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set temporary directory to D drive
os.environ['TEMP'] = 'D:\\temp'
os.environ['TMP'] = 'D:\\temp'

# Ensure the temporary directory exists
os.makedirs('D:\\temp', exist_ok=True)

# Function to check disk space
def check_disk_space(directory):
    total, used, free = shutil.disk_usage(directory)
    print(f"Disk space for {directory}: Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB")
    return free // (2**30)

# Check disk space before starting
check_disk_space('D:\\')
cache_dir = 'D:\\temp'

# Set the cache directory for the datasets library
os.environ['HF_DATASETS_CACHE'] = cache_dir

from datasets import load_dataset, DatasetDict
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# Authenticate to the Hugging Face Hub
from huggingface_hub import login

# Use your Hugging Face API token
login(token='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')

# Load the Speech Commands dataset
speech_commands = DatasetDict()
speech_commands["train"] = load_dataset("speech_commands", "v0.01", split="train", cache_dir=cache_dir, trust_remote_code=True)
speech_commands["test"] = load_dataset("speech_commands", "v0.01", split="test", cache_dir=cache_dir, trust_remote_code=True)

# Initialize the tokenizer and processor
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe", cache_dir=cache_dir)
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe", cache_dir=cache_dir)

# Function to prepare the dataset
def prepare_dataset(batch):
    # Load and resample audio data to 16kHz
    audio = batch["audio"]
    # Compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # Ensure label is a string
    label = str(batch["label"])
    # Encode target text to label ids
    batch["labels"] = tokenizer(label).input_ids
    return batch

# Apply the preparation function to the dataset
speech_commands = speech_commands.map(prepare_dataset, remove_columns=speech_commands["train"].column_names, num_proc=1)

# Initialize the model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", cache_dir=cache_dir)
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Define a data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Define the metric for evaluation
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Adjust training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=r"D:\whisper_fine_tuned",
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=4,  # Increased gradient accumulation steps
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=False,  # Disabled gradient checkpointing
    fp16=False,  # Disable fp16
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,   # Reduced eval batch size
    predict_with_generate=True,
    generation_max_length=10,  # Single-word commands
    save_steps=1000,  # Save checkpoint every 1000 steps
    save_total_limit=3,  # Keep only the 3 most recent checkpoints
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="amhard/whisper_fine_tuned",
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=speech_commands["train"],
    eval_dataset=speech_commands["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Compute and print the number of epochs
num_train_samples = len(speech_commands["train"])
batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
max_steps = training_args.max_steps
num_epochs = (max_steps * batch_size) / num_train_samples

print(f"Number of epochs: {num_epochs:.2f}")

# Train the model
#trainer.train()

# To resume training from the last checkpoint (if necessary)
# Uncomment the following line:
trainer.train(resume_from_checkpoint=True)