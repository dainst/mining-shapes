from django.forms.boundfield import BoundField


def get_name_of_choice_field(choice_filed: BoundField) -> str:
    return choice_filed.subwidgets[int(choice_filed.data)].choice_label
