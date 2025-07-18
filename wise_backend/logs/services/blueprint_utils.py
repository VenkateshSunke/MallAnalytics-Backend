"""
Utility functions for working with Blueprint data
"""
import os
import sys
import logging
import django
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def setup_django():
    """Setup Django environment for database access"""
    try:
        # Add the Django project path to sys.path
        project_path = '/Users/sunkevenkateswarlu/Desktop/onelab_projects/wise_backend'
        if project_path not in sys.path:
            sys.path.append(project_path)
        
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wise_backend.settings')
        django.setup()
        return True
    except Exception as e:
        logger.error(f"Failed to setup Django: {e}")
        return False

def get_blueprint_mapping_by_camera_id(camera_id: str) -> Dict[str, Any]:
    """
    Retrieve blueprint mapping data for a specific camera ID
    
    Args:
        camera_id (str): The camera ID to search for
        
    Returns:
        dict: Blueprint mapping data or empty dict if not found
    """
    if not setup_django():
        logger.warning("Django not available, cannot retrieve blueprint data")
        return {}
    
    try:
        from core.models import Blueprint
        
        # Get the most recent active blueprint for this camera
        blueprint = Blueprint.objects.filter(
            camera_id=camera_id,
            is_active=True
        ).order_by('-created_at').first()
        
        if blueprint:
            logger.info(f"Found blueprint '{blueprint.name}' for camera {camera_id}")
            return blueprint.mapping_data
        else:
            logger.warning(f"No active blueprint found for camera {camera_id}")
            return {}
            
    except Exception as e:
        logger.error(f"Error retrieving blueprint for camera {camera_id}: {e}")
        return {}

def get_blueprint_info_by_camera_id(camera_id: str) -> Dict[str, Any]:
    """
    Get detailed blueprint information for a camera ID
    
    Args:
        camera_id (str): The camera ID to search for
        
    Returns:
        dict: Blueprint information including name, description, mapping data, etc.
    """
    if not setup_django():
        logger.warning("Django not available, cannot retrieve blueprint data")
        return {}
    
    try:
        from core.models import Blueprint
        
        blueprint = Blueprint.objects.filter(
            camera_id=camera_id,
            is_active=True
        ).order_by('-created_at').first()
        
        if blueprint:
            return {
                'blueprint_id': blueprint.blueprint_id,
                'name': blueprint.name,
                'description': blueprint.description,
                'camera_id': blueprint.camera_id,
                'mapping_data': blueprint.mapping_data,
                'created_at': blueprint.created_at.isoformat() if blueprint.created_at else None,
                'updated_at': blueprint.updated_at.isoformat() if blueprint.updated_at else None,
                'is_active': blueprint.is_active
            }
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error retrieving blueprint info for camera {camera_id}: {e}")
        return {}

def get_all_active_blueprints() -> Dict[str, Any]:
    """
    Get all active blueprints
    
    Returns:
        dict: Dictionary with camera_id as key and blueprint info as value
    """
    if not setup_django():
        logger.warning("Django not available, cannot retrieve blueprint data")
        return {}
    
    try:
        from core.models import Blueprint
        
        blueprints = Blueprint.objects.filter(is_active=True).order_by('-created_at')
        
        result = {}
        for blueprint in blueprints:
            result[blueprint.camera_id] = {
                'blueprint_id': blueprint.blueprint_id,
                'name': blueprint.name,
                'description': blueprint.description,
                'camera_id': blueprint.camera_id,
                'mapping_data': blueprint.mapping_data,
                'created_at': blueprint.created_at.isoformat() if blueprint.created_at else None,
                'updated_at': blueprint.updated_at.isoformat() if blueprint.updated_at else None,
                'is_active': blueprint.is_active
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving all blueprints: {e}")
        return {}

def test_blueprint_retrieval():
    """Test function to verify blueprint retrieval works"""
    # Test with a camera ID from cameras.py
    test_camera_id = "uuid:58112d3b-1023-0783-1f4e-810f3a8e1804:video"
    
    print(f"Testing blueprint retrieval for camera: {test_camera_id}")
    
    # Get blueprint info
    blueprint_info = get_blueprint_info_by_camera_id(test_camera_id)
    if blueprint_info:
        print(f"Found blueprint: {blueprint_info['name']}")
        print(f"Description: {blueprint_info['description']}")
        print(f"Mapping data keys: {list(blueprint_info['mapping_data'].keys()) if blueprint_info['mapping_data'] else 'None'}")
    else:
        print("No blueprint found for this camera")
    
    # Get all active blueprints
    all_blueprints = get_all_active_blueprints()
    print(f"\nTotal active blueprints: {len(all_blueprints)}")
    for camera_id, info in all_blueprints.items():
        print(f"  - {camera_id}: {info['name']}")

if __name__ == "__main__":
    test_blueprint_retrieval() 