from django.contrib import admin
from .models import User
# Register your models here.


class UserManager(admin.ModelAdmin):
    list_display = ['username', 'name', 'tel_number', 'email', 'user_bool']
    # list_display_links 修改页连接
    # list_filter 过滤器添加
    # search_field 模糊查询


admin.site.register(User, UserManager)
