
# LLM's for Amharic Named Entity Recognition (NER)

This repository contains scripts and data related to the fine-tuning of a Language Model (LLM) for **Amharic Named Entity Recognition (NER)**. The project focuses on processing data collected from Telegram-based e-commerce channels in Ethiopia and involves labeling, tokenizing, and scrapping product names, prices, and locations.

## Project Structure
├── .github.<br>
│   └── workflow<br>
│       └── test.yaml<br>
├── .venv <br>
├── .vscode<br>
│   └── settings.json<br>
├── data<br>
│   ├── adamagebeya_telegram_data.csv <br>
│   ├── conll_format_data.txt <br>
│   └── telegram_data.csv<br>
├── notebook<br>
│   ├── data_processing_labeling.ipynb<br>
│   └── scrap_label_tokenize.ipynb<br>
├── script
│   ├── data_processor_labler.py<br>
│   └── telegram_scrapper.py<br>
├── .env<br>
├── .gitignore<br>
├──  README.MD<br>
├──  requirements.txt <br>

The following is an overview of the file structure for the project:
## Contents

### 1. `.github/workflow/test.yaml`
- Contains a GitHub Actions configuration for Continuous Integration (CI), specifically designed to run unit tests across multiple Python versions.

### 2. `data/`
- Contains various datasets used for NER tasks:
  - **adamgebeya_telegram_data.csv**: Dataset scraped from the 'Adama Gebeya' Telegram channel.
  - **conll_format_data.txt**: Data in CoNLL format, useful for training NER models.

### 3. `notebook/`
- Jupyter Notebooks:
  - **data_processing_labeling.ipynb**: Steps for cleaning, labeling, and processing the scraped data.
  - **scrap_label_tokenize.ipynb**: Notebook for scraping Telegram channels and tokenizing messages.

### 4. `script/`
- Python scripts:
  - **data_processor_labler.py**: Script to label and process the data for NER training.
  - **telegram_scrapper.py**: Script for scraping Telegram channels to gather data for the project.

### 5. `.env`
- Configuration for environment variables.

### 6. `.gitignore`
- Specifies which files and directories Git should ignore.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wolderufael/LLM-s-for-Amharic-Named-Entity-Recognition-NER.git
   cd LLM-s-for-Amharic-Named-Entity-Recognition-NER
2. **Create and activate a virtual environment**:
   ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. **Install dependencies**:
   ```bash
    pip install -r requirements.txte
4. **Set environment variables**:
    * Modify the values in `.env` as needed.
5. **Run Jupyter Notebooks**:
    * You can launch the Jupyter notebooks located in the `notebook/` folder to see data processing and scraping steps.
6. **Running scripts**
    * Use the `data_processor_labler.py` to process and label the dataset.
    * Use the `telegram_scrapper.py` to scrape data from Telegram channels.

## Contributing
1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature`').
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
## License
This project is licensed under the terms of the MIT License. See the `LICENSE` file for details.