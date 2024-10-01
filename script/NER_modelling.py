import os
from datasets import Dataset
from transformers import TrainingArguments
from transformers import AutoModelForTokenClassification, Trainer
from sklearn.metrics import classification_report

class Modelling:
    def load_conll_dataset(self,file_path):
        # parse CoNLL data and return it as a Hugging Face Dataset
        def parse_conll(file_path):
            sentences = []
            current_sentence = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "": 
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                        # if len(sentences) >= max_sentences:
                        #     break
                    else:
                        word, label = line.split()  # Assumes word and label are separated by space
                        current_sentence.append((word, label))
                
                if current_sentence :# Catch any remaining sentenc
                    sentences.append(current_sentence)
            
            return sentences

        # Parse the data
        parsed_sentences = parse_conll(file_path)

        # Prepare the data in dictionary format
        data = {
            "tokens": [[word for word, label in sentence] for sentence in parsed_sentences],
            "ner_tags": [[label for word, label in sentence] for sentence in parsed_sentences],
        }

        # Create and return a Hugging Face dataset
        dataset = Dataset.from_dict(data)
        
        return dataset   
    
    def merge_conll_files(self,input_files, output_file):
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in input_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
                        outfile.write("\n")  # Add a newline to separate sentences from different files
                else:
                    print(f"Warning: The file {file_path} does not exist and will be skipped.")
    
    
    def tokenize_and_align_labels(self,dataset, tokenizer, label_all_tokens=False):
        def tokenize_and_align(examples):
            tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)  # Special tokens
                    elif word_idx != previous_word_idx:  # Only label first subword
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_dataset = dataset.map(tokenize_and_align, batched=True)
        return tokenized_dataset

    
    def setup_training_args(self,output_dir):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",  # Evaluate after every epoch
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        return training_args

    def fine_tune_model(self,dataset,model_name, train_dataset, val_dataset,tokenizer, training_args):
        num_labels = len(set([label for labels in dataset['ner_tags'] for label in labels]))
        # Load pre-trained model
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )
        
        # Fine-tune the model
        trainer.train()
        
        return model

    def evaluate_model(self,dataset,model, val_dataset):
        # Get unique labels from the dataset
        unique_labels = set(label for sentence_labels in dataset['ner_tags'] for label in sentence_labels)

        # Sort the labels for consistency (e.g., from 'B-PER' to 'O')
        label_list = sorted(unique_labels)
        # Get predictions
        predictions, labels, _ = model.predict(val_dataset)
        
        # Flatten predictions and labels
        predictions = predictions.argmax(axis=-1)
        true_labels = labels.flatten()
        predicted_labels = predictions.flatten()
        
        # Get a classification report
        report = classification_report(true_labels, predicted_labels, target_names=label_list)
        print(report)

    def save_model(model, tokenizer,output_dir):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

