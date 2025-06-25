from django.core.management.base import BaseCommand
from core.models import EmailCampaign
from core.utils.sendgrid_service import create_list, SENDGRID_BASE_URL, HEADERS
import requests

def get_list_id_by_name(name):
    res = requests.get(f"{SENDGRID_BASE_URL}/marketing/lists", headers=HEADERS)
    res.raise_for_status()
    lists = res.json().get("result", [])
    for l in lists:
        if l.get("name") == name:
            return l.get("id")
    return None

class Command(BaseCommand):
    help = 'Fix EmailCampaigns missing a sendgrid_list_id by creating or reusing a SendGrid list and saving the ID.'

    def handle(self, *args, **options):
        broken = EmailCampaign.objects.filter(sendgrid_list_id__isnull=True)
        if not broken.exists():
            self.stdout.write(self.style.SUCCESS('No campaigns missing sendgrid_list_id.'))
            return
        for campaign in broken:
            try:
                list_id = get_list_id_by_name(campaign.name)
                if not list_id:
                    list_id = create_list(campaign.name)
                campaign.sendgrid_list_id = list_id
                campaign.save()
                self.stdout.write(self.style.SUCCESS(f'Fixed campaign {campaign.campaign_id} with list {list_id}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to fix campaign {campaign.campaign_id}: {e}')) 