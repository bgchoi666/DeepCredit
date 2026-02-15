from django.db import models

class BoardList(models.Model):
    board_idx = models.AutoField(db_column='board_idx', primary_key=True)  # Field name made lowercase.
    title = models.CharField(db_column='title', max_length=300)  # Field name made lowercase.
    contents = models.CharField(db_column='contents', max_length=65535, blank=True, null=True)  # Field name made lowercase.
    hit_cnt = models.IntegerField(db_column='hit_cnt', blank=True, null=True)  # Field name made lowercase.
    created_datetime = models.DateTimeField(db_column='created_datetime', max_length=20, blank=True, null=True)  # Field name made lowercase.
    creator_id = models.CharField(db_column='creator_id', max_length=50, blank=True, null=True)  # Field name made lowercase.
    updated_datetime = models.DateTimeField(db_column='updated_datetime', max_length=20, blank=True, null=True)  # Field name made lowercase.
    updater_id = models.CharField(db_column='updater_id', max_length=50, blank=True, null=True)  # Field name made lowercase.
    deleted_yn = models.CharField(db_column='deleted_yn', max_length=1, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 't_board'
