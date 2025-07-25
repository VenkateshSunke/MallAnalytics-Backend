# Generated by Django 5.2.3 on 2025-06-27 06:32

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0016_remove_campaignstep_image_metadata'),
    ]

    operations = [
        migrations.CreateModel(
            name='CampaignStepImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('s3_key', models.CharField(max_length=500)),
                ('s3_url', models.URLField()),
                ('original_filename', models.CharField(max_length=255)),
                ('content_type', models.CharField(max_length=100)),
                ('file_size', models.IntegerField()),
                ('upload_order', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('step', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='images', to='core.campaignstep')),
            ],
            options={
                'ordering': ['upload_order', 'created_at'],
            },
        ),
    ]
