# schedule/celery1.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.schedules import crontab
from celery.utils.log import get_task_logger

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'task.settings')

app = Celery('schedule')

app.config_from_object('django.conf:settings', namespace='CELERY')
app.conf.broker_url = 'redis://localhost:6379/0'
# app.conf.broker_url = 'redis://192.168.188.1:6379/0'
# 启用工作进程自动重启以实现负载均衡
app.conf.worker_pool_restarts = True
app.conf.task_default_retry_delay = 5
app.conf.beat_schedule = {
    'check_queues_periodically': {
        'task': 'schedule.task.monitor_queues',  # 指定任务函数的路径
        'schedule': crontab(minute='*', hour='*', day_of_week='*'),  # 每分钟执行一次
        'options': {'queue': 'monitor'},  # 将定时任务发送到 monitor 队列
    },
}
app.autodiscover_tasks()

