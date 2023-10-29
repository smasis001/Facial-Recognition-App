# Facial Recognition App

## Project Structure

```
.
└── Facial-Recognition-App/
    ├── facial_recognition_app/
    │   ├── __init__.py
    │   └── .py
    ├── facial_recognition_webapp/
    │   ├── .html
    │   ├── .js
    │   └── .css
    ├── data
    │   ├── raw
    │   └── processed
    ├── config
    ├── models
    ├── test
    │   └── .py
    ├── .env
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── mkdocs.yaml
    ├── pyproject.toml
    └── README.md
```

## Steps to get started

1. Create Python 3.11 environment
    * `python -m venv .venv` or `python3.11 -m venv .venv`
2. Initialize poetry
    * `pip install --upgrade pip` and `pip install poetry` (if not already installed)
    * `poetry env use .venv/bin/python3.11`
    * `python -m poetry install`
    * `source .venv/bin/activate`
3. Optionally, download the open source [models](/models/README.md)
4. Adapt [configs](/config/) files:
    * If you plan to use downloaded models, make sure they point to the model(s) you downloaded or
    * Otherwise, ensure that ChatGPT in the config files has the right API keys.
5. Adapt the "session" variables in the [clients](/clients/)
6. Run the api:
    * `python clients/.py`
    * `python -m chainlit run clients/t.py --port 7861` (it will open a browser window)
    * if you need to kill chainlit client look for pid with `lsof -i tcp:7861` and `kill -9 [pid]` or go `lsof -t -i tcp:7861 | xargs kill`

## Example Outputs

