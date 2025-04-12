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

[Add instructions for running your project here]

## Additional Information

[Any other relevant information about your project]
