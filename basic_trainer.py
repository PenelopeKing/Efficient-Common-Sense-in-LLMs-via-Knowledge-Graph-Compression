import torch
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

def get_basic_trainer(
    source_path: str,
    target_path: str,
    model_name: str = "facebook/bart-large",
    output_dir: str = "my_bart_finetuned",
    max_len: int = 128,
    epochs: int = 3,
    train_batch_size: int = 60
):
    """
    Returns a Hugging Face Trainer instance for fine-tuning BART.
      - source_path: Path to train.source file
      - target_path: Path to train.target file
      - model_name:  Pretrained model identifier
      - output_dir:  Directory to save outputs
      - max_len:     Max input/target length
      - epochs:      Number of training epochs
      - train_batch_size: Batch size

    Usage:
        trainer = get_bart_trainer("train.source", "train.target")
        trainer.train()
    """

    # ---------------------
    # Load raw data lines
    # ---------------------
    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src]
        targets = [line.strip() for line in f_tgt]

    # ---------------------
    # Build HF Dataset
    # ---------------------
    train_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]
    train_dataset = Dataset.from_list(train_data)

    # ---------------------
    # Load tokenizer & model
    # ---------------------
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------------
    # Preprocessing
    # ---------------------
    def preprocess_function(examples):
        # Encode the source
        model_inputs = tokenizer(examples["source"], max_length=max_len, truncation=True)
        # Encode the target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target"], max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)

    # ---------------------
    # DataCollator
    # ---------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ---------------------
    # Training Arguments
    # ---------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        eval_strategy="no",  # Change to "steps" or "epoch" if you have a val set
    )

    # ---------------------
    # Initialize Trainer
    # ---------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer
