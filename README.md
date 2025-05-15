# Topic Clustering Tool

A Python tool with a streamlit web interface that uses natural language processing to automatically cluster similar topics from text data. It leverages sentence transformers and cosine similarity to group related content.

## Features

- Interactive web interface built with Streamlit
- Real-time clustering progress visualization
- Text clustering using state-of-the-art sentence transformers
- Support for large datasets with batch processing
- GPU acceleration when available
- Excel file input/output support
- Adjustable similarity threshold with interactive controls
- Automatic fallback to CPU if CUDA is not available
- In-browser file upload and download

## Project Structure

```
topic-clustering/
├── src/
│   ├── analyzer/
│   │   ├── text_analyzer.py   # Core clustering logic with progress tracking
│   │   └── __init__.py
│   ├── utils/
│   │   ├── embeddings.py      # Embedding utilities
│   │   └── __init__.py
│   ├── main.py               # Streamlit web interface
│   └── __init__.py
├── requirements.txt          # Project dependencies
└── README.md
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/topic-clustering.git
cd topic-clustering
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Activate your virtual environment:
```bash
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

2. Start the Streamlit web interface:
```bash
streamlit run src/main.py
```

3. Use the web interface:
   - Upload your Excel file using the file uploader
   - Select the column containing text to cluster from the dropdown
   - Adjust the similarity threshold (0.0-1.0) using the slider
   - Set the batch size using the number input (default: 500)
   - Click "Start Clustering" to begin the process
   - Monitor progress in real-time
   - Download results directly in the browser when complete

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- sentence-transformers >= 2.2.0
- scikit-learn >= 0.24.0
- torch >= 1.9.0
- openpyxl >= 3.0.7
- streamlit >= 1.22.0

## How It Works

1. File Upload: Upload your Excel file through the web interface
2. Text Embedding: Converts text into numerical vectors using sentence transformers (with progress tracking)
3. Similarity Calculation: Computes cosine similarity between text embeddings (with progress tracking)
4. Clustering: Groups similar texts based on the similarity threshold
5. Results: Download the Excel file with topic assignments directly in your browser

The interface provides real-time progress updates for each step of the process, making it easy to monitor the status of large clustering tasks.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
This project is provided by the RDB L&D Team.