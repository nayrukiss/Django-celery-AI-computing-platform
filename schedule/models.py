from django.db import models


class Schedule(models.Model):
    user = models.ForeignKey('user.User', on_delete=models.CASCADE)
    parameters = models.IntegerField(default=0)
    result = models.TextField(default="")
    result2 = models.TextField(default="")
    result3 = models.TextField(default="")
    status = models.CharField(max_length=20, default='pending')  # 'pending', 'processing', 'completed'
    priority = models.IntegerField(default=0)
    process_info = models.CharField(max_length=255, blank=True, null=True)
    process_num = models.IntegerField(default=-1)
    image_oss_url = models.CharField(max_length=200, default='')
    p1_oss_url = models.CharField(max_length=200, default='')
    p2_oss_url = models.CharField(max_length=200, default='')
    p3_oss_url = models.CharField(max_length=200, default='')
    task_type = models.IntegerField(default=0)
    ga_function_code = models.TextField(default="")  # 函数代码
    process1 = models.FloatField(default=0)
    process2 = models.FloatField(default=0)
    process3 = models.FloatField(default=0)

    class Meta:
        verbose_name = '任务'
        verbose_name_plural = verbose_name
