import os
from dotenv import load_dotenv
load_dotenv()
import requests
from django.utils.timezone import now
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
if not SENDGRID_API_KEY:
    raise ValueError("SENDGRID_API_KEY environment variable is required")

SENDGRID_BASE_URL = "https://api.sendgrid.com/v3"
HEADERS = {
    "Authorization": f"Bearer {SENDGRID_API_KEY}",
    "Content-Type": "application/json"
}

class SendGridError(Exception):
    """Custom exception for SendGrid API errors"""
    pass

def _handle_response(response, operation_name):
    """Helper function to handle API responses consistently"""
    if response.status_code >= 400:
        logger.error(f"[SendGrid] {operation_name} failed: {response.status_code} - {response.text}")
        raise SendGridError(f"{operation_name} failed: {response.status_code} - {response.text}")
    return response

def create_list(name):
    """Create a new contact list in SendGrid"""
    try:
        res = requests.post(
            f"{SENDGRID_BASE_URL}/marketing/lists",
            headers=HEADERS,
            json={"name": name}
        )
        _handle_response(res, "Create list")
        list_id = res.json().get("id")
        if not list_id:
            raise SendGridError("No list ID returned from SendGrid")
        logger.info(f"[SendGrid] Created list '{name}' with ID: {list_id}")
        return list_id
    except requests.RequestException as e:
        logger.error(f"[SendGrid] Network error creating list: {e}")
        raise SendGridError(f"Network error creating list: {e}")

def sync_contacts_to_list(campaign, contacts):
    """Sync contacts to a SendGrid list"""
    try:
        data = {
            "list_ids": [campaign.sendgrid_list_id],
            "contacts": [
                {"email": c.user.email, "first_name": c.user.name}
                for c in contacts
            ]
        }
        res = requests.put(
            f"{SENDGRID_BASE_URL}/marketing/contacts", 
            headers=HEADERS, 
            json=data
        )
        
        logger.info(f"[SendGrid] Sync contacts status: {res.status_code}")
        logger.debug(f"[SendGrid] Sync contacts response: {res.text}")
        
        _handle_response(res, "Sync contacts")
        return res.status_code, res.json()
        
    except requests.RequestException as e:
        logger.error(f"[SendGrid] Network error syncing contacts: {e}")
        raise SendGridError(f"Network error syncing contacts: {e}")

def validate_html_content(html_content):
    """Validate and clean HTML content for SendGrid"""
    if not html_content or not html_content.strip():
        raise SendGridError("HTML content cannot be empty")
    
    # Basic validation - ensure it has some content
    if len(html_content.strip()) < 10:
        raise SendGridError("HTML content is too short")
    
    # If it's plain text, wrap it in basic HTML
    if not html_content.strip().startswith('<'):
        html_content = f"<html><body>{html_content}</body></html>"
    
    return html_content

def create_sendgrid_campaign(step, sender_id, suppression_group_id=None):
    """Create a new SendGrid campaign"""
    try:
        # Validate required fields
        if not step.subject or not step.subject.strip():
            raise SendGridError("Campaign subject cannot be empty")
        
        if not step.body or not step.body.strip():
            raise SendGridError("Campaign body cannot be empty")
        
        # Validate and clean HTML content
        validated_html = validate_html_content(step.body)
        
        # Create campaign name with timestamp to ensure uniqueness
        campaign_name = f"{step.campaign.name} - Step {step.step_order} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        data = {
            "name": campaign_name,
            "subject": step.subject.strip(),
            "html_content": validated_html,
            "sender_id": int(sender_id),  # Ensure sender_id is integer
            "list_ids": [step.campaign.sendgrid_list_id],
            "categories": [f"campaign_{getattr(step.campaign, 'campaign_id', 'unknown')}", f"step_{step.id}"]  # Add categories for tracking
        }

        # Only add suppression_group_id if it's provided and valid
        if suppression_group_id:
            try:
                data["suppression_group_id"] = int(suppression_group_id)
            except (ValueError, TypeError):
                logger.warning(f"[SendGrid] Invalid suppression_group_id: {suppression_group_id}")

        logger.info(f"[SendGrid] Creating campaign with data: {data}")
        
        res = requests.post(
            f"{SENDGRID_BASE_URL}/marketing/singlesends", 
            headers=HEADERS, 
            json=data
        )
        
        logger.info(f"[SendGrid] Campaign creation response: {res.status_code} - {res.text}")
        _handle_response(res, "Create campaign")
        
        response_data = res.json()
        campaign_id = response_data.get("id")
        if not campaign_id:
            raise SendGridError(f"No campaign ID returned from SendGrid. Response: {response_data}")
            
        logger.info(f"[SendGrid] Created campaign '{campaign_name}' with ID: {campaign_id}")
        return campaign_id
        
    except requests.RequestException as e:
        logger.error(f"[SendGrid] Network error creating campaign: {e}")
        raise SendGridError(f"Network error creating campaign: {e}")

def schedule_sendgrid_campaign(campaign_id, dt):
    """Schedule a SendGrid campaign for sending"""
    try:
        # Validate datetime
        if not dt:
            raise SendGridError("Send datetime is required")
        
        # Check if the datetime is in the future
        if dt <= datetime.now(dt.tzinfo):
            raise SendGridError("Send datetime must be in the future")
        
        # Ensure datetime is in UTC and properly formatted for SendGrid
        if dt.tzinfo is None:
            logger.warning("[SendGrid] Datetime provided without timezone info, assuming UTC")
        
        # SendGrid expects ISO format with timezone info
        iso_time = dt.isoformat()
        if dt.tzinfo is None:
            iso_time += 'Z'
        elif str(dt.tzinfo) == 'UTC':
            iso_time = iso_time.replace('+00:00', 'Z')
            
        data = {"send_at": iso_time}
        
        logger.info(f"[SendGrid] Scheduling campaign {campaign_id} with data: {data}")
        
        res = requests.put(
            f"{SENDGRID_BASE_URL}/marketing/singlesends/{campaign_id}/schedule", 
            headers=HEADERS, 
            json=data
        )
        
        logger.info(f"[SendGrid] Schedule response: {res.status_code} - {res.text}")
        _handle_response(res, "Schedule campaign")
        
        logger.info(f"[SendGrid] Scheduled campaign {campaign_id} for {iso_time}")
        return res.status_code
        
    except requests.RequestException as e:
        logger.error(f"[SendGrid] Network error scheduling campaign: {e}")
        raise SendGridError(f"Network error scheduling campaign: {e}")

def get_senders():
    """Get all verified senders from SendGrid"""
    try:
        res = requests.get(f"{SENDGRID_BASE_URL}/senders", headers=HEADERS)
        _handle_response(res, "Get senders")
        
        senders = res.json()
        logger.info(f"[SendGrid] Retrieved {len(senders)} senders")
        return senders
        
    except requests.RequestException as e:
        logger.error(f"[SendGrid] Network error getting senders: {e}")
        return []
    except SendGridError:
        logger.error("[SendGrid] API error getting senders")
        return []

def get_default_sender_id():
    """Get the ID of the first available verified sender"""
    senders = get_senders()
    if senders and len(senders) > 0:
        # Look for verified senders first
        for sender in senders:
            if sender.get('verified', {}).get('status') == True:
                sender_id = sender.get('id')
                logger.info(f"[SendGrid] Using verified sender ID: {sender_id}")
                return sender_id
        
        # If no verified sender found, use the first one
        sender_id = senders[0].get('id')
        logger.info(f"[SendGrid] Using default sender ID: {sender_id}")
        return sender_id
    
    logger.warning("[SendGrid] No senders available")
    return None

def get_campaign_details(campaign_id):
    """Get details of a specific SendGrid campaign for debugging"""
    try:
        res = requests.get(
            f"{SENDGRID_BASE_URL}/marketing/singlesends/{campaign_id}",
            headers=HEADERS
        )
        _handle_response(res, "Get campaign details")
        return res.json()
    except Exception as e:
        logger.error(f"[SendGrid] Error getting campaign details: {e}")
        return None