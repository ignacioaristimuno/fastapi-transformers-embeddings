# FastAPI & Transfomers Embeddings template

Cookiecutter template for creating a Transformers Embeddings API with `FastAPI` and `transformers`.

## Description

This project template allows you to quickly set up an embeddings API using transformers models. It offers a straightforward structure for loading a pre-trained model, defining API endpoints, and serving predictions over HTTP. Additionally, Dockerfile and docker-compose files are provided to streamline the deployment process.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Configuration](#configuration)
- [License](#license)

## Features

- Load a pre-trained transformers embedding model.
- Expose API endpoints for text encoding.
- Easy-to-use interface for model inference.
- Dockerfile and docker-compose files for simplified deployment.

## Setup

### Requirements

- Python 3.x (tested on Python 3.10.13)
- [pip](https://pip.pypa.io/en/stable/)
- [Cookiecutter](https://cookiecutter.readthedocs.io/en/latest/)
- [virtualenv](https://virtualenv.pypa.io/en/stable/) (recommended)
- [Docker](https://www.docker.com/) (recommended)
- [Docker Compose](https://docs.docker.com/compose/) (recommended)

### Installation

1. First, install Cookiecutter if you haven't already:

   ```bash
   pip install cookiecutter
   ```
2. Now, generate your project from this template using Cookiecutter:

   ```bash
   cookiecutter <path_to_template>
   ```

   Follow the prompts to fill in the required variables for your project.
3. After generating the project, navigate into the project directory:

   ```bash
   cd <your_project_name>
   ```
4. Create and activate a virtual environment (recommended):

   ```bash
   virtualenv venv
   source venv/bin/activate
   ```
5. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
6. **Option 1: Run with Docker Compose**

   ```bash
   docker-compose up
   ```
7. **Option 2: Run without Docker Compose**

   Start the API server:

   ```bash
   python3 src/api/app.py
   ```
8. Send requests to the API following the usage instructions below.

## Usage

1. **Start the API server**:

   ```bash
   python app.py
   ```
2. **Send requests**:

   - Endpoint: `/encoding/texts`
   - Method: POST
   - Payload:
     ```json
     {
         "texts": [
             "This is an example of a sentence",
             "Another example of a another sentence"
         ],
         "instruct": false
     }
     ```
   - Example request using cURL:
     ```bash
     curl -X POST -H "Content-Type: application/json" -d '{"texts": ["This is an example of a sentence", "Another example of a another sentence"], "instruct": false}' http://localhost:8080/encoding/texts
     ```
3. **Response**:

   The API will respond with a JSON object containing the embeddings along with the used model's metadata.

## Endpoints

- `/health`: Endpoint for checking the status of the API.
- `/encoding/texts`: Endpoint for encoding a batch of text.

## Configuration

- Modify `src/settings/config.yml` to specify the embedding model from [Hugging Face](https://huggingface.co/models?pipeline_tag=feature-extraction&sort=trending).
- Update the `src/models` based on your specific needs.
- Add the necessary endpoints for your use case within the `src/api/routers` folder.

## License

This project is licensed under the [MIT License](LICENSE).
