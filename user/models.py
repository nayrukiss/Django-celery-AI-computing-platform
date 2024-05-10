from django.db import models
# Create your models here.


class User(models.Model):
    username = models.CharField("工号", max_length=30, unique=True)
    password = models.CharField("密码", max_length=32)
    name = models.CharField("姓名", max_length=32)
    kind = models.IntegerField("权限", default=2)
    tel_number = models.CharField("电话号码", max_length=12)
    email = models.EmailField("邮箱号码")
    user_bool = models.BooleanField("存在", default='True')
    created_time = models.DateTimeField("创建时间", auto_now_add=True)
    updated_time = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = '用户'
        verbose_name_plural = verbose_name
