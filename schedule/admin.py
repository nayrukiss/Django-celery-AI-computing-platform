from django.contrib import admin
from .models import Schedule


class ScheduleManager(admin.ModelAdmin):
    list_display = ['user', 'parameters', 'result', 'status', 'priority']
    # list_display_links 修改页连接
    # list_filter 过滤器添加
    # search_field 模糊查询
# Register your models here.


admin.site.register(Schedule, ScheduleManager)