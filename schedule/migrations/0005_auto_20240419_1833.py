# Generated by Django 3.1 on 2024-04-19 10:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('schedule', '0004_auto_20240412_1648'),
    ]

    operations = [
        migrations.AddField(
            model_name='schedule',
            name='result2',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='schedule',
            name='result3',
            field=models.TextField(default=''),
        ),
    ]
