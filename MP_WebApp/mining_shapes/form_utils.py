from django.forms.boundfield import BoundField
from django.conf import settings
from typing import List


def get_name_of_choice_field(choice_filed: BoundField) -> str:
    return choice_filed.subwidgets[int(choice_filed.data)].choice_label


def get_features_from_feature_field(choices: List[str]):
    """ Get chosen feature types from FEATURES tuple in settings.py """
    return [settings.FEATURES[int(i)-1][1] for i in choices]
