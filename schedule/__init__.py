from __future__ import absolute_import, unicode_literals

# 导入Celery应用实例
from .celery1 import app as celery_app  # 根据实际文件结构调整导入路径

# 确保Django加载Celery配置
__all__ = ('celery_app',)
