from django import forms
from typing import Tuple
from django.conf import settings
import glob
import os


def available_models() -> Tuple[Tuple[str, str]]:
    """ Get available models from MODELS_DIR """
    out = []
    for i, fname in enumerate(glob.glob(f"{settings.MODELS_DIR}*.h5")):
        out.append((str(i), os.path.basename(fname)))
    return tuple(out)


class RunSessionForm(forms.Form):

    images = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}))
    model = forms.ChoiceField(choices=available_models())
    features = forms.MultipleChoiceField(choices=settings.FEATURES)
