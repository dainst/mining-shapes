from django.contrib import admin
from .models import Session, VesselProfile, FetureType, SegmentationImage

admin.site.register(Session)
admin.site.register(VesselProfile)
admin.site.register(FetureType)
admin.site.register(SegmentationImage)
