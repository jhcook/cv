# AI CV Maker

An advanced, AI-powered tool for tailoring your CV and generating cover letters for specific job descriptions.

## Features

- **Multi-Provider LLM Support**:
  - **Google Vertex AI**: Enterprise-grade performance (Priority).
  - **Google AI Studio**: Access Gemini 1.5 Pro/Flash models.
  - **OpenAI**: Support for GPT-3.5/4.
  - **Auto-Discovery**: Automatically finds and caches available Gemini models.
  - **Mock Data**: Fallback mode for testing without API keys.

- **Smart Ingestion**:
  - **Job Descriptions**: URL (with Playwright fallback for JS sites), PDF, DOCX, or Text files.
  - **Master Library**: Local folder or **Google Drive / OneDrive** shared folders.
  - **GitHub**: Fetches your top repositories to populate a "Technical Portfolio" section.

- **Template Engine**:
  - **Style Preservation**: Heuristically detects and applies fonts, headers, and bullet styles from your template.
  - **Format Support**: Generates polished DOCX output.
  - **Cover Letter**: Automatically generates a matching cover letter.

- **Cloud Integration**:
  - **GCS Support**: Directly upload outputs to Google Cloud Storage (`gs://`).

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/jhcook/cv.git
    cd cv
    ```

2. **Install dependencies**:

    ```bash
    # Create a virtual environment recommended
    python3 -m venv .venv
    source .venv/bin/activate
    
    pip install -r requirements.txt
    
    # Check if Playwright is needed (for JS-heavy JD URLs)
    playwright install chromium
    ```

## Configuration

Set up your API keys in your environment variables:

```bash
# For Google AI Studio
export GEMINI_API_KEY="your-key"

# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Vertex AI
# Ensure you have run: gcloud auth application-default login
```

## Usage

### Smart Path Resolution

The CLI automatically looks in `user_content/` subdirectories if a file isn't found locally:

- **`--jd`**: Checks `user_content/inputs/`
- **`--template`**: Checks `user_content/templates/`
- **`--library`**: Defaults to `user_content/library/`

### Examples

```bash
# Finds 'SRE_Role.txt' in user_content/inputs/
python run.py --jd SRE_Role.txt

# Uses 'Agency_Template.docx' from user_content/templates/
python run.py --jd SRE_Role.txt --template Agency_Template.docx
```

### Full Options

```bash
python run.py \
  --jd "SRE_Role.txt" \
  --output "Tailored_CV.docx" \
  --github "jhcook" \
  --verbose
```

### Arguments

| Argument | Description | Default Search Path |
| :--- | :--- | :--- |
| `--jd` | **Required**. Job Description file or URL. | `user_content/inputs/` |
| `--library` | Master CVs folder (DOCX/PDF). | `user_content/library/` (Default) |
| `--template` | Custom DOCX template file. | `user_content/templates/` |
| `--output` | Output filename or path (supports `gs://`). | `user_content/generated_cvs/` |
| `--github` | GitHub username for portfolio section. | |
| `--suggestions` | Comma-separated template overrides (e.g. 'font,header'). | |
| `--summarize` | Years of recent experience to detail (default: 10). | |
| `--list-models`| List available LLM models and exit. | |
| `-v` / `--verbose` | Increase verbosity level. | |
| `-q` / `--quiet` | Suppress status output (ERROR only). | |

## Directory Structure

The project isolates user data from source code:

- **`user_content/`**: All your local data.
  - `library/`: Place your Master CVs here.
  - `inputs/`: Default folder for Job Descriptions.
  - `templates/`: Default folder for custom templates.
  - `generated_cvs/`: Where tailored CVs are saved.
  - `logs/`: Application logs (`cv.log`).
  - `library_cache/`: Cached downloads from Cloud Drives.
  - `.model_cache.json`: Cache of discovered LLM models.
- **`src/`**: Application source code.

## Development

- **`inspect_template.py`**: Debug script to analyze DOCX styles.
- **`compare_docs.py`**: Debug script to verify formatting preservation.
