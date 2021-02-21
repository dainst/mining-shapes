from django import forms


class RunSessionForm(forms.Form):
    images = forms.ImageField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}))
