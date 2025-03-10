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
    test_source_path: str = None,
    test_target_path: str = None,
    model_name: str = "facebook/bart-large",
    output_dir: str = "my_bart_finetuned",
    max_len: int = 128,
    epochs: int = 3,
    train_batch_size: int = 60,
    eval_batch_size: int = 20,
    num_points: int = 25_597
):
    """
    Returns a Hugging Face Trainer instance for fine-tuning BART.
    """
    
    # ---------------------
    # Load raw training data
    # ---------------------
    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src][:num_points]
        targets = [line.strip() for line in f_tgt][:num_points]

    train_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]
    train_dataset = Dataset.from_list(train_data)

    # ---------------------
    # Load raw testing data if provided
    # ---------------------
    test_dataset = None
    if test_source_path is not None and test_target_path is not None:
        with open(test_source_path, "r", encoding="utf-8") as f_src_test, open(test_target_path, "r", encoding="utf-8") as f_tgt_test:
            test_sources = [line.strip() for line in f_src_test]
            test_targets = [line.strip() for line in f_tgt_test]
        test_data = [{"source": s, "target": t} for s, t in zip(test_sources, test_targets)]
        test_dataset = Dataset.from_list(test_data)
        # Save the raw target (with multiple candidates) in a separate field.
        def add_raw_target(example):
            example["raw_target"] = example["target"]
            return example
        test_dataset = test_dataset.map(add_raw_target)
    
    # ---------------------
    # Load tokenizer & model
    # ---------------------
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ---------------------
    # Preprocessing function for training data
    # ---------------------
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["source"], max_length=max_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target"], max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    
    # ---------------------
    # Preprocessing function for test data
    # ---------------------
    if test_dataset is not None:
        def preprocess_test(example):
            model_inputs = tokenizer(example["source"], max_length=max_len, truncation=True)
            # Retain the raw target for custom loss computation.
            model_inputs["raw_target"] = example["raw_target"]
            # Create a dummy label from the first candidate answer.
            first_candidate = example["raw_target"].split("\t")[0]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(first_candidate, max_length=max_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        test_dataset = test_dataset.map(preprocess_test)
    
    # ---------------------
    # DataCollator
    # ---------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # ---------------------
    # preprocess_logits_for_metrics function to reduce memory usage during evaluation
    # ---------------------
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids.cpu(), labels

    # ---------------------
    # Custom compute_metrics function:
    # For each test example, we compute the loss over all candidate answers (separated by tabs)
    # and report both the minimum loss and the average loss.
    # ---------------------
    def compute_metrics(eval_pred):
        # We ignore eval_pred because we compute our own losses over the test dataset.
        losses_min = []
        losses_avg = []
        
        # Iterate over the test dataset.
        for example in test_dataset:
            source_text = example["source"]
            # Split the raw target to obtain candidate answers.
            candidates = example["raw_target"].split("\t")
            model_inputs = tokenizer(source_text, max_length=max_len, truncation=True, return_tensors="pt")
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            candidate_losses = []
            for candidate in candidates:
                with torch.no_grad():
                    with tokenizer.as_target_tokenizer():
                        candidate_tokenized = tokenizer(candidate, max_length=max_len, truncation=True, return_tensors="pt")
                    candidate_labels = candidate_tokenized["input_ids"].to(device)
                    outputs = model(**model_inputs, labels=candidate_labels)
                    candidate_losses.append(outputs.loss.item())
            losses_min.append(min(candidate_losses))
            losses_avg.append(sum(candidate_losses) / len(candidate_losses))
            torch.cuda.empty_cache()
        
        avg_loss_min = sum(losses_min) / len(losses_min) if losses_min else 0.0
        avg_loss_avg = sum(losses_avg) / len(losses_avg) if losses_avg else 0.0
        return {"avg_loss_min": avg_loss_min, "avg_loss_avg": avg_loss_avg}
    
    # ---------------------
    # Training Arguments
    # ---------------------
    eval_strategy = "epoch" if test_dataset is not None else "no"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        save_steps=5000,
        save_total_limit=10,
        logging_steps=500,
        evaluation_strategy='no',
        fp16=False,
    )
    
    # ---------------------
    # Initialize Trainer with our custom compute_metrics and preprocess_logits_for_metrics functions.
    # ---------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    return trainer
