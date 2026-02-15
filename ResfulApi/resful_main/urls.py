from django.conf.urls import url, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
#from rest_framework import routers
app_name = 'restful_main'

#router = routers.DefaultRouter()
#router.register(r'todfo', views.Todo_subject_restful_main)
urlpatterns = [
    path('', views.index, name='index'),
    url('api-auth/', include('rest_framework.urls')),
    url(r'^$', views.Board_restful.as_view(), name='main'),
    url(r'^board/$', views.Board_restful.as_view(), name='board_list'),
    url(r'^board/(?P<board_idx>\d+)/$', views.Board_restful_detail.as_view(), name='board_detail'),
    url(r'^board/create/$', views.Board_restful_create.as_view(), name='board_create'),
    url(r'^board/(?P<board_idx>\d+)/update/$', views.Board_restful_update.as_view(), name='board_update'),
    url(r'^board/(?P<board_idx>\d+)/delete/$', views.Board_restful_delete.as_view(), name='board_delete'),
    # url(r'^ajax/ajaxEnsembleDist/(?P<batch_id>\d+)/(?P<model_no>\d+)$', views.ajax_ensemble_dist, name='ensemble'),
    url(r'^ensembleDist.ajax/$', views.ajax_ensemble_dist, name='ensemble'),
    # url(r'^ajax/future/$', views.ajax_future, name='future'),
    # url(r'^ajax/joblib/$', views.ajax_joblib, name='joblib'),
    # url(r'^board/future', views.check_post, name='board'),
    # url(r'^board/joblib', views.check_post, name='board'),
    # url(r'^board/insert/(?P<no>\d+)$', views.check_post, name='board'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)