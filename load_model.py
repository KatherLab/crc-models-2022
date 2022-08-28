from fastai.vision.learner import load_learner
from pathlib import Path


def load_model(path_to_pkl: Path, cpu: bool=True):
    learner=load_learner(path_to_pkl,cpu=cpu)
    return learner.model

