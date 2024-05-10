"""
Django settings for task project.

Generated by 'django-admin startproject' using Django 2.0.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.0/ref/settings/
"""

import os
from celery.schedules import crontab
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 't@*4+2d&(nhs%e4on97(ku)rf)t0+7fn@*-1-+m_4ra3&pj-++'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_results',
    'django_celery_beat',
    'user',
    'index',
    'schedule',
]

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
# Celery settings
CELERY_RESULT_BACKEND = 'db+mysql://root:123456@localhost/task'
CELERY_BROKER_URL = 'redis://localhost:6379/0'
# CELERY_RESULT_BACKEND = 'db+mysql://root:123456@192.168.188.1/task'
# CELERY_BROKER_URL = 'redis://192.168.188.1:6379/0'
# add
# CELERY_BROKER_URL += ',redis://192.168.188.100:6379/0'
# CELERY_BROKER_URL += ',redis://192.168.188.101:6379/0'
# 启用优先级队列功能
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
TASK_DEFAULT_QUEUE = 'low_priority'
CELERY_QUEUES = {
    'high_priority': {
        'exchange': 'high_priority',
        'routing_key': 'high_priority',
        'queue_arguments': {'priority': 1},
    },
    'low_priority': {
        'exchange': 'low_priority',
        'routing_key': 'low_priority',
        'queue_arguments': {'priority': 100},
    },
    'monitor': {
        'exchange': 'monitor',
        'routing_key': 'monitor',
        'queue_arguments': {'monitor': 20},
    },
}
CELERY_AUTOSCALE = 'prefork'              # 启用自动扩展功能
CELERYD_PREFETCH_MULTIPLIER = 1           # 只取一个任务
CELERYD_AUTOSCALER = 'celery.worker.autoscale:Autoscaler'
CELERYD_AUTOSCALE_MAX_TASKS_PER_CHILD = 100  # 设置每个 Worker 进程最大处理的任务数
CELERYD_AUTOSCALE_INTERVAL = 10  # 设置自动扩展的检查间隔，单位为秒
CELERYD_AUTOSCALE_MULTIPLIER = 1.0  # 设置自动扩展的乘数因子
CELERYBEAT_SCHEDULER = 'django_celery_beat.schedulers.DatabaseScheduler'
DJANGO_CELERY_BEAT_TZ_AWARE = False
CELERY_TASK_RETRY_DELAY = 5
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'task.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'task.wsgi.application'


# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'task',
        'USER': 'root',
        'PASSWORD': '123456',
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

LANGUAGE_CODE = 'zh-Hans'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_URL = '/static/'