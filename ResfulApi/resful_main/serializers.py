from .models import BoardList
from rest_framework import serializers

class BoardSerializer(serializers.ModelSerializer):
    class Meta:
        model = BoardList
        fields = ('board_idx', 'title', 'hit_cnt', 'created_datetime', 'creator_id', 'updated_datetime',
                  'updater_id', 'deleted_yn')

class BoardDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = BoardList
        fields = ('title', 'contents', 'hit_cnt', 'created_datetime', 'creator_id', 'updated_datetime',
                  'updater_id', 'deleted_yn')

class BoardCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = BoardList
        fields = ('title', 'contents', 'creator_id')

class BoardUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = BoardList
        fields = ('title', 'contents', 'updated_datetime', 'updater_id')