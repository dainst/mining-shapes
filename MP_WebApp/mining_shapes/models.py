from django.db import models
from django.dispatch import receiver
import os

from user.models import User


class FetureType(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self) -> str:
        return f"{self.name}"


class Session(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    model = models.CharField(max_length=255)
    catalog = models.CharField(max_length=255)
    features = models.ManyToManyField(
        FetureType, symmetrical=False, blank=True)

    def __str__(self) -> str:
        return f"{self.pk}_{self.user}_{self.catalog}"

# pylint: disable=no-member


class VesselProfile(models.Model):
    filename = models.CharField(max_length=255)
    input_image = models.ImageField(upload_to="orig_image")
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    segmented_image = models.FileField(
        blank=True, null=True, upload_to="seg_images")
    # ADD vector fields

    def __str__(self) -> str:
        return f"{self.pk}_{self.filename}_{self.session.catalog}"

# pylint: disable=unused-argument


@receiver(models.signals.post_delete, sender=VesselProfile)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """
    Deletes file from filesystem
    when corresponding `MediaFile` object is deleted.
    """
    if instance.input_image:
        if os.path.isfile(instance.input_image.path):
            os.remove(instance.input_image.path)
    if instance.segmented_image:
        if os.path.isfile(instance.segmented_image.path):
            os.remove(instance.segmented_image.path)


@receiver(models.signals.pre_save, sender=VesselProfile)
def auto_delete_file_on_change(sender, instance, **kwargs):
    """
    Deletes old file from filesystem
    when corresponding `MediaFile` object is updated
    with new file.
    """
    if not instance.pk:
        return False

    try:
        old_file = sender.objects.get(pk=instance.pk).segmented_image
    except sender.DoesNotExist:
        return False

    new_file = instance.segmented_image
    if not old_file == new_file:
        if os.path.isfile(old_file.path):
            os.remove(old_file.path)
