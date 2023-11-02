# Facial Recognition App

## Project Structure

```
.
└── Facial-Recognition-App/
    ├── facial_recognition/
    │   ├── __init__.py
    │   ├── _utils_.py
    │   ├── detect.py
    │   ├── align.py
    │   ├── represent.py
    │   └── identify.py
    ├── webapp/
    │   ├── app.py
    │   ├── static
    │   │   ├── app.js
    │   │   └── styles.css
    │   └── templates
    │       └── index.html
    ├── scripts
    │   ├── cameras.py
    │   ├── preprocess_images.py
    │   ├── demo1.py
    │   └── demo2.py
    ├── data
    │   ├── raw
    │   └── processed
    │   └── vectordb
    ├── models
    │   └── dlib_face_recognition_resnet_model_v1.dat
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── mkdocs.yaml
    ├── pyproject.toml
    └── README.md
```

## Documentation

For technical documentation of code under `facial_recognition` please refer to [github.io](https://smasis001.github.io/Facial-Recognition-App/). All other code make use of `facial_recognition` and has internal documentation explaining how it uses it.

## Getting Started

### Install Basic Requirements

You'll need Python 3.9 and above, and GIT. On Windows, the best way is via the [Python website](https://python.org/downloads/windows/) and [GIT website](https://git-scm.com/downloads/win). On other operating systems command line options are the easiest:

**MacOs**:

```sh
brew install python git
```
**Linux**:

```sh
sudo apt install python3 git
```

It is recommended that you open the folder where this repository will be cloned with an IDE like VSCode or PyCharm.

### Create Environment

First you need to clone this repository, and then create the environment. You can use `pyenv`, `conda` or whichever one you want but for this example we are using `venv` which is included in `python`.

```sh
cd /path/where/you/want/to/clone/repo
git clone https://github.com/smasis001/Facial-Recognition-App.git
python -m venv .venv
```

Then for managing the packages, this repository comes with `pyproject.toml` for `poetry`, but you can use `requirements.txt` and `pip`. For poetry, this is how it goes:

```sh
pip install poetry
python -m poetry env use .venv/bin/python
python -m poetry install
source .venv/bin/activate
```

For installing the packages with `pip`, you must activate the virtual enviroment first.

```sh
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Raw Data (optional)

The Vector database (`/data/vectordb`) has facial biometrics for celebrity faces which was a sample of 5% of the images from [here](https://github.com/prateekmehta59/Celebrity-Face-Recognition-Dataset). If you want the raw sources used for the vector DB or to rebuild the vector DB with your own data, get it from [here](https://drive.google.com/file/d/1-K29GGW-xBBUvV2UlSn3c3TtmO08wgwi/view?usp=sharing) and unzip it into the `data/raw` folder.

### Rebuild Vector DB (optional)

If you want to process the `raw` images and store corresponding embeddings into Vector DB, run this:

```sh
python scripts/preproces_images.py
```

It assumes that the names of the folders under `data/raw` are the names of the people. There can only be one face per image in these folders for it to work.

### Run Web App (locally)

To see the facial recognition system work via browser, run:

```sh
python webapp/app.py
```

and then go to [http://127.0.0.1/](http://127.0.0.1/) preferably in a **Chrome or Chromium-based browser** (Edge, Opera, etc). It does work in Safari and some Firefox versions but there's no guarantees.


When the page opens **please camera permissions**, and press ctrl-C to stop the webapp in your command line.

### Run Desktop App (locally)

To demo face detection with Python, run:

```sh
python scripts/demo1.py
```

And for the facial recognition demo:

```sh
python scripts/demo2.py
```

Press `esc` to close the windows.
If for some reason it's using the wrong camera, run:

```sh
python scripts/cameras.py
```

and arrow left and right through your cameras till you find the one you prefer to use. Take note of the camera number. Then, open `scripts/demo1.py` and `scripts/demo2.py` and change the line that says `CAMERA_ID = 0` for the number you prefer.
