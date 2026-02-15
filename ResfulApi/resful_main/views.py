from django.conf.urls import url, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
#from rest_framework import routers
app_name = 'restful_main'
from .models import BoardList
from .serializers import BoardSerializer, BoardDetailSerializer, BoardCreateSerializer, BoardUpdateSerializer
# from rest_framework.decorators import api_view
from rest_framework.generics import ListAPIView, RetrieveAPIView, UpdateAPIView, DestroyAPIView, CreateAPIView
from django.http import JsonResponse
from django.shortcuts import render

import os
import sys
import json

os.chdir("..")
sys.path.append(".")

from RestfulApi.ResultView import ResultView

# from env.util import *
# from DataManager_pkg.DataManager import DataManager
# from Logger_pkg.Logger import Logger

def index(request):
    return render(request, "index.html")

class Board_restful(ListAPIView):
    queryset = BoardList.objects.all()
    serializer_class = BoardSerializer

class Board_restful_detail(RetrieveAPIView):
    lookup_field = 'board_idx'
    queryset = BoardList.objects.all()
    serializer_class = BoardDetailSerializer

class Board_restful_create(CreateAPIView):
    queryset = BoardList.objects.all()
    serializer_class = BoardCreateSerializer

class Board_restful_update(UpdateAPIView):
    lookup_field = 'board_idx'
    queryset = BoardList.objects.all()
    serializer_class = BoardUpdateSerializer

class Board_restful_delete(DestroyAPIView):
    lookup_field = 'board_idx'
    queryset = BoardList.objects.all()
    serializer_class = BoardSerializer

def ajax_ensemble_dist(request, *callback_args, **callback_kwargs):
    resultView = ResultView()

    # batch_id = callback_kwargs.get("batch_id")
    # model_no = callback_kwargs.get("model_no")
    batch_id = request.GET.get("batchId")
    model_no = request.GET.get("modelNo")
    model_sub_no = request.GET.get("modelSubNo")

    batch_info = resultView.get_batch_info(batch_id)
    result = resultView.get_ensemble_dist(batch_id, model_no, model_sub_no)

    to_json = {
        "batchInfo": json.loads(batch_info.to_json(orient="records")),
        "result": json.loads(result.to_json(orient="records"))
    }

    if request.headers.get("Proxy") == "DeepCredit":
        return JsonResponse(to_json)

    # return JsonResponse(json.loads(to_json), safe=False)
    return JsonResponse(to_json, safe=False)


# def ajax_future(request):
#     logger = Logger(PATH.LOG, "django")
#     dataManager = DataManager(PATH.INPUT, logger)
#
#     future = dataManager.get_future_slice_data(**dict(base_date='2020-04-03', days=20))
#
#     result = future.to_json(orient="table")
#     return JsonResponse(json.loads(result), safe=False)

# def ajax_joblib(request):
#     data = joblib.load(join(
#         "C:\\Users\\wooha\\PycharmProjects\\IndexTracking\\5_output\\2017-04-03\\min_volatility opt=measure_risk_future\\[2017-03-30~2017-05-31]",
#         "data.joblib"))
#     # data = json.dumps(str(data))
#     newData = '{"port":' + pd.DataFrame(data[0]['train']['validation']['port']).to_json(orient="table") + '",' + \
#               '"weight":' + pd.DataFrame(data[0]['train']['validation']['port']).to_json(orient="table") + '}'
#     return JsonResponse(json.loads(newData), safe=False)

# SAMPLE
# def check_post(request, no=None, pk=None):
#     # when POST
#     if request.method == "POST":
#         if str(request.path).split("/board/")[1].split("/")[0] == "insert":
#             request.POST.get('title')
#
#             return JsonResponse({'text': '저장되었습니다.'})
#     # GET
#     else:
#         if str(request.path).split("/board/")[1].split("/")[0] == "insert":
#             return JsonResponse({'text': '저장되었습니다.'})