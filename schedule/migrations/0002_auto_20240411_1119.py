# Generated by Django 3.1 on 2024-04-11 03:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='schedule',
            name='ga_function_code',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='schedule',
            name='ga_parameters',
            field=models.JSONField(default=dict),
        ),
    ]