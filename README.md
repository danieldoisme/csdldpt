# Project Setup Instructions

## Prerequisites

- Python 3.12 or higher
- Git

## Setting Up the Development Environment

### Installing uv

`uv` is a fast Python package installer and resolver that we use for dependency management.

#### On macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Verify installation:

```bash
uv --version
```

### Installing Project Dependencies

1. Clone this repository (if you haven't already):

   ```bash
   git clone <repository-url>
   cd csdldpt
   ```

2. Create and activate a virtual environment using uv:

   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/macOS
   # Or on Windows:
   # .venv\Scripts\activate
   ```

3. Install dependencies from requirements.txt:
   ```bash
   uv pip install -r requirements.txt
   ```

## Running the Project

The project is a leaf image similarity search system with three main operation modes:

### 1. Preprocessing Images

To preprocess raw images:

```bash
uv run python main.py --mode preprocess
```

### 2. Building the Feature Database

To build the feature database:

```bash
uv run python main.py --mode train
```

For using deep learning features (CNN):

```bash
uv run python main.py --mode train --use_deep
```

### 3. Searching for Similar Images

To search for images similar to a query image:

```bash
uv run python main.py --mode search --query test_images/test1.jpg
```

For searching with deep learning features:

```bash
uv run python main.py --mode search --query test_images/test1.jpg --use_deep
```

To search for more than 3 similar images:

```bash
uv run python main.py --mode search --query test_images/test1.jpg --top_k 5
```

### 4. Run All Steps at Once

To run preprocessing, database building, and search in one command:

```bash
uv run python main.py --mode all --query test_images/test1.jpg --use_deep
```

### Additional Parameters

- `--top_k`: Number of similar images to return (default: 3)
- `--result_dir`: Directory to save results (default: results)
