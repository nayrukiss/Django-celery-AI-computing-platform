# Generated by Django 3.1 on 2024-04-23 07:41

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0006_auto_20240423_1540'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='schedule',
            name='find',
        ),
    ]
