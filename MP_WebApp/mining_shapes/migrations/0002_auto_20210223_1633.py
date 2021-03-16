# Generated by Django 3.1.7 on 2021-02-23 16:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mining_shapes', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='SegmentationImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(default='/home/Code/MP_WebApp/pattern1.png', upload_to='seg_images')),
            ],
        ),
        migrations.AlterField(
            model_name='vesselprofile',
            name='segmented_image',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='mining_shapes.segmentationimage'),
        ),
    ]