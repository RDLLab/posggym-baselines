FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml ./
COPY setup.py ./
COPY ./posggym-baselines/__init__.py /app/posggym-baselines/__init__.py

pip install -e .[exps]

COPY . .
