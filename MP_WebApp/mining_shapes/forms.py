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
    styling = {'class': "form-control mx-1 w-75 mb-2"}
    images = forms.ImageField(
        widget=forms.ClearableFileInput(attrs=dict({'multiple': True}, **styling)))
    model = forms.ChoiceField(choices=available_models(
    ), widget=forms.widgets.Select(attrs=styling))
    features = forms.MultipleChoiceField(
        choices=settings.FEATURES, widget=forms.widgets.SelectMultiple(attrs=styling))
    catalog = forms.CharField(
        widget=forms.widgets.Textarea(attrs=dict({'rows': 1}, **styling)))
