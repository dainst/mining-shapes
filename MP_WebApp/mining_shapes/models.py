from django.db import models
from django.db.models.fields.files import ImageField

from user.models import User


class FetureType(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self) -> str:
        return f"{self.name}"


class Session(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    model = models.CharField(max_length=255)
    features = models.ManyToManyField(
        FetureType, symmetrical=False, blank=True)

    def __str__(self) -> str:
        return f"{self.pk}_{self.user}"


class VesselProfile(models.Model):
    filename = models.CharField(max_length=255)
    input_image = ImageField(upload_to="orig_image")
    segmented_image = models.ImageField(
        upload_to="seg_images", blank=True, null=True),
    catalog = models.CharField(max_length=255)
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    # ADD vector fields

    def __str__(self) -> str:
        return f"{self.filename}_{self.catalog}"
