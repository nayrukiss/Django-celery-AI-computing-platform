# Generated by Django 3.1 on 2024-04-11 04:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0002_auto_20240411_1119'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='schedule',
            name='ga_parameters',
        ),
    ]
