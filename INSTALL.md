# ShoesAI Development Environment Setup

This document provides instructions for setting up your development environment for the ShoesAI project.

## Prerequisites

- Python 3.10+ installed
- CUDA-compatible GPU recommended for optimal performance
- Git installed (for version control)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd talk-shoesai/testdrive
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

```bash
# Using venv (Python's built-in virtual environment)
python -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
# venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required packages:

```bash
pip install -r requirements.txt
```

Note: Some packages like torch might require specific versions depending on your CUDA version. If you encounter issues with CUDA compatibility, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Download Model Files

The project uses several pre-trained models. These should be placed in the `models/` directory. The models include:

- `yolo-peito.pt`: YOLOv8 model for detecting race bibs
- `yolo11l.pt`: YOLOv8 model for person detection
- `classify_shoes_model.pth`: ViT16 model for shoe classification
- `classify_gender_model.pt`: PyTorch model for gender classification

If these files aren't already included in the repository, you'll need to download them from the project's shared storage or train them using the provided scripts.

### 5. Environment Variables (Optional)

If you're using Google's Generative AI API, you'll need to set up your API key:

```bash
# On Linux/macOS
export GOOGLE_API_KEY="your-api-key"

# On Windows PowerShell
# $env:GOOGLE_API_KEY="your-api-key"
```

### 6. Running the Application

To run the Streamlit web application:

```bash
streamlit run app.py
```

The application should now be running at [http://localhost:8501](http://localhost:8501).

## Development Tools

### Running Scripts

Various utility scripts are available in the `scripts/` directory:

```bash
# Example: Running the script to extract shoes from images
python scripts/cut_shoes_from_folder.py --input_folder /path/to/images --output_folder output/
```

To generate JSON analysis with metadata using the detector:

```bash
python detector.py path/to/images --output output/my_dataset.json
```

The generated JSON will include metadata with:
- Datetime of the operation
- Name and version of the classify shoes model used
- Confidence threshold and margin of error (MoE) of the model

### Testing

To run tests:

```bash
# Run all tests
python -m unittest discover

# Run a specific test
python -m unittest test_shoes_analyzer
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: If you encounter GPU memory issues, try reducing batch sizes or model complexity in the config.yaml file.

2. **ImportError for insightface or easyocr**: These packages sometimes have additional dependencies that may need to be installed separately:
   ```bash
   pip install cmake
   ```

3. **Model Loading Errors**: Ensure that all model files are correctly placed in the models/ directory and that the paths in config.yaml match the actual file locations.

## Additional Resources

- Check the README.md for project overview and usage instructions
- Refer to the config.yaml file for configuration options

## Contact

For any further assistance, please contact the project maintainers.
