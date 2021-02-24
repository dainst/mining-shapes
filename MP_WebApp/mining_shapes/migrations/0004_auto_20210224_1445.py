# Generated by Django 3.1.7 on 2021-02-24 14:45

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('mining_shapes', '0003_auto_20210223_1811'),
    ]

    operations = [
        migrations.DeleteModel(
            name='SegmentationImage',
        ),
        migrations.RemoveField(
            model_name='vesselprofile',
            name='catalog',
        ),
        migrations.AddField(
            model_name='session',
            name='catalog',
            field=models.CharField(default=django.utils.timezone.now, max_length=255),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='vesselprofile',
            name='segmented_image',
            field=models.FileField(blank=True, null=True, upload_to='seg_images'),
        ),
    ]
