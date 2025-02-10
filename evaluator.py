import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import Dataset
from bert_score import score
import pandas as pd
import os
from tqdm.auto import tqdm

def evaluate_model(
    model,
    tokenizer,
    data_dir,
    batch_size=8,
    max_length=60,
    num_beams=4,
    lang='en'
):
    """
    Evaluates a fine-tuned BART model using BERTScore on validation and test datasets.

    Parameters:
    - model (BartForConditionalGeneration): The fine-tuned BART model.
    - tokenizer (BartTokenizer): The tokenizer corresponding to the model.
    - data_dir (str): Directory containing 'val.source', 'val.target', 'test.source', 'test.target'.
    - batch_size (int, optional): Batch size for generating predictions. Defaults to 8.
    - max_length (int, optional): Maximum length for generated explanations. Defaults to 60.
    - num_beams (int, optional): Number of beams for beam search during generation. Defaults to 4.
    - lang (str, optional): Language code for BERTScore. Defaults to 'en'.
    - output_csv_path (str, optional): Path to save the evaluation results as a CSV. Defaults to "bert_score_evaluation_results.csv".

    Returns:
    - pandas.DataFrame: A DataFrame containing BERTScore metrics for validation and test sets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def load_custom_dataset(source_path, target_path):
        """
        Loads source and target files and returns a Hugging Face Dataset.
        """
        with open(source_path, "r", encoding="utf-8") as src_file, open(target_path, "r", encoding="utf-8") as tgt_file:
            sources = [line.strip() for line in src_file]
            targets = [line.strip() for line in tgt_file]
        data = {"source": sources, "target": targets}
        dataset = Dataset.from_dict(data)
        return dataset

    def preprocess_function(examples, tokenizer, max_length=128):
        """
        Tokenizes the source and target texts.
        """
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def generate_predictions(model, tokenizer, dataset, batch_size=8, max_length=60, num_beams=4):
        """
        Generates predictions for the given dataset.

        Parameters:
        - model: The fine-tuned BART model.
        - tokenizer: The corresponding tokenizer.
        - dataset: A Hugging Face Dataset.
        - batch_size (int): Number of examples to process at once.
        - max_length (int): Maximum length of the generated explanation.
        - num_beams (int): Beam search width.

        Returns:
        - List[str]: A list of generated explanations.
        """
        predictions = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions.extend(decoded_preds)
        return predictions

    def compute_bertscore(predictions, references, lang='en'):
        """
        Computes BERTScore for given predictions and references.

        Parameters:
        - predictions (List[str]): Generated explanations.
        - references (List[str]): Ground truth explanations.
        - lang (str): Language code (default is English).

        Returns:
        - Dict[str, float]: A dictionary with BERTScore metrics (precision, recall, f1).
        """
        P, R, F1 = score(predictions, references, lang=lang, verbose=True)
        metrics = {
            "BERTScore_P": P.mean().item(),
            "BERTScore_R": R.mean().item(),
            "BERTScore_F1": F1.mean().item()
        }
        return metrics

    # Define paths to validation and test sets
    val_source = os.path.join(data_dir, "val.source")
    val_target = os.path.join(data_dir, "val.target")
    test_source = os.path.join(data_dir, "test.source")
    test_target = os.path.join(data_dir, "test.target")

    # Load datasets
    val_dataset = load_custom_dataset(val_source, val_target)
    test_dataset = load_custom_dataset(test_source, test_target)

    # Preprocess datasets
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length=128),
        batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length=128),
        batched=True
    )

    # Set format for PyTorch
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Generate predictions
    print("Generating predictions for Validation Set...")
    val_predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=val_dataset,
        batch_size=batch_size,
        max_length=max_length,
        num_beams=num_beams
    )

    print("Generating predictions for Test Set...")
    test_predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=batch_size,
        max_length=max_length,
        num_beams=num_beams
    )

    # Decode ground truth labels to text
    val_references = val_dataset['labels']
    test_references = test_dataset['labels']

    val_ground_truth = tokenizer.batch_decode(val_references, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    test_ground_truth = tokenizer.batch_decode(test_references, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Compute BERTScore
    print("Computing BERTScore for Validation Set...")
    val_bertscore = compute_bertscore(val_predictions, val_ground_truth, lang=lang)

    print("Computing BERTScore for Test Set...")
    test_bertscore = compute_bertscore(test_predictions, test_ground_truth, lang=lang)

    # Compile results into a DataFrame
    evaluation_results = pd.DataFrame({
        "Dataset": ["Validation", "Test"],
        "BERTScore_P": [val_bertscore["BERTScore_P"], test_bertscore["BERTScore_P"]],
        "BERTScore_R": [val_bertscore["BERTScore_R"], test_bertscore["BERTScore_R"]],
        "BERTScore_F1": [val_bertscore["BERTScore_F1"], test_bertscore["BERTScore_F1"]]
    })

    return evaluation_results
