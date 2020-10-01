from django.conf.urls import include, url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Examples:
    # url(r'^$', 'mask_detector_api.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^mask_detection/detect/$', 'mask_detector.views.detect'),
    url(r'^mask_detection/stream_frame/$', 'mask_detector.views.stream_frame'),
    url(r'^mask_detection/detect_from_stream/$', 'mask_detector.views.detect_from_stream'),
    url(r'^mask_detection/validate_cert/$', 'mask_detector.views.validate_cert'),
   # url(r'^mask_detection/get_latest_face/$', 'mask_detector.views.get_latest_face'),
    url(r'^mask_detection/test/$', 'mask_detector.views.test'),
	
    url(r'^admin/', include(admin.site.urls)),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)