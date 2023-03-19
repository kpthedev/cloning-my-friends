import math

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def tokenize_data(data, tokenizer):
    """Tokenizes the input data using the specified tokenizer."""
    tokenized_data = data.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=data.column_names,
    )
    return tokenized_data


def group_text_blocks(data, block_size):
    """Groups the text in the dataset into blocks of length <= block_size."""

    def group_texts(example):
        combined = {
            k: [item for sublist in v for item in sublist] for k, v in example.items()
        }
        combined_length = len(list(combined.values())[0])

        if combined_length >= block_size:
            combined_length = (combined_length // block_size) * block_size

        result = {}
        for k, t in combined.items():
            result[k] = [
                t[i : i + block_size] for i in range(0, combined_length, block_size)
            ]
        result["labels"] = result["input_ids"].copy()
        return result

    return data.map(group_texts, batched=True)


def train_model(training_args, train_data, eval_data, tokenizer, model_name_or_path):
    """Trains a causal language model using the specified training and evaluation data."""
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer


if __name__ == "__main__":
    # Filenames
    pretrained_model = "gpt2-large"
    output_model = "finetuned-gpt2-large"
    training_input_file = "input.txt"

    # Load and split dataset
    data = load_dataset(
        "text",
        data_files=training_input_file,
        split=["train[5%:]", "train[:5%]"],
        cache_dir="./cache",
        keep_linebreaks=True,
    )
    train_data, eval_data = data[0], data[1]

    # Tokenize the data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    train_data = tokenize_data(train_data, tokenizer)
    eval_data = tokenize_data(eval_data, tokenizer)

    # Group the data
    block_size = 1024
    train_data = group_text_blocks(train_data, block_size)
    eval_data = group_text_blocks(eval_data, block_size)

    training_args = TrainingArguments(
        output_dir=f"./models/{output_model}",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        optim="adamw_bnb_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
    )

    # Set seed
    set_seed(42)

    # Train the model
    trainer = train_model(
        training_args,
        train_data,
        eval_data,
        tokenizer,
        pretrained_model,
    )

    # Save the model
    trainer.save_model()

    # Evaluate the model
    eval_loss = trainer.evaluate()["eval_loss"]
    perplexity = math.exp(eval_loss)
    print(f"Perplexity: {perplexity:.2f}")
