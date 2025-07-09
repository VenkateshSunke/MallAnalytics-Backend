"""
Dashboard metrics utility functions.
"""
from django.db.models import Avg, Count, Sum
from django.utils import timezone as dj_timezone
from datetime import timedelta, datetime
from ..models import User, Visit, UserMovement
from django.db.models import Q


def get_current_month_start():
    """Get the first day of current month"""
    now = dj_timezone.now()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_previous_month_start():
    """Get the first day of previous month"""
    current_month_start = get_current_month_start()
    if current_month_start.month == 1:
        return current_month_start.replace(year=current_month_start.year - 1, month=12)
    else:
        return current_month_start.replace(month=current_month_start.month - 1)


def get_dashboard_metrics(days=30):
    """
    Calculate all dashboard metrics for the given time period.
    
    Args:
        days: Number of days to look back (default 30, but main metrics use current month)
    
    Returns:
        Dictionary containing all dashboard metrics
    """
    now = dj_timezone.now()
    current_month_start = get_current_month_start()
    previous_month_start = get_previous_month_start()
    
    # Main metrics (current month based)
    active_user_ids = User.objects.filter(
        visits__start_time__gte=current_month_start
    ).distinct().values_list("user_id", flat=True)
    total_visitors = len(active_user_ids)
    
    new_active_users = User.objects.filter(first_visit__gte=current_month_start).count()
    
    # Average visit duration (current month)
    avg_duration = Visit.objects.filter(
        start_time__gte=current_month_start
    ).exclude(duration__isnull=True).aggregate(
        avg=Avg("duration")
    )["avg"]
    
    if avg_duration:
        hours, remainder = divmod(avg_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        avg_visit_duration = f"{int(hours):02}:{int(minutes):02}"
    else:
        avg_visit_duration = "00:00"
    
    # Average stores visited (current month)
    distinct_stores = UserMovement.objects.filter(
        visit__start_time__gte=current_month_start,
        store__isnull=False
    ).values("store").distinct().count()
    total_users_count = User.objects.count()
    avg_stores_visited = distinct_stores / total_users_count if total_users_count else 0
    
    # NEW METRICS
    
    # 1. Total Registered Users (Lifetime)
    total_registered_users = User.objects.count()
    
    # 2. Current Month Registrations
    current_month_registrations = User.objects.filter(
        created_at__gte=current_month_start
    ).count()
    
    # 3. Registration Growth % (Month-over-month)
    previous_month_registrations = User.objects.filter(
        created_at__gte=previous_month_start,
        created_at__lt=current_month_start
    ).count()
    
    if previous_month_registrations > 0:
        registration_growth = ((current_month_registrations - previous_month_registrations) / previous_month_registrations) * 100
    else:
        registration_growth = 100.0 if current_month_registrations > 0 else 0.0
    
    # 4. Monthly Visits (Current month)
    monthly_visits = Visit.objects.filter(
        start_time__gte=current_month_start
    ).count()
    
    # 5. Monthly Visitors (Unique users who visited this month)
    monthly_visitors = User.objects.filter(
        visits__start_time__gte=current_month_start
    ).distinct().count()
    
    # 6. Monthly Store Visits (Total store interactions this month)
    monthly_store_visits = Visit.objects.filter(
        start_time__gte=current_month_start
    ).aggregate(
        total_stores=Sum('stores_visited')
    )['total_stores'] or 0
    
    # Additional calculated metrics
    
    # Monthly Visitor Engagement Rate (% of registered users active this month)
    monthly_engagement_rate = (monthly_visitors / total_registered_users) * 100 if total_registered_users > 0 else 0
    
    # Average Stores per Monthly Visitor
    avg_stores_per_monthly_visitor = monthly_store_visits / monthly_visitors if monthly_visitors > 0 else 0
    
    # Registration Rate (current month registrations / total registered users)
    registration_rate = (current_month_registrations / total_registered_users) * 100 if total_registered_users > 0 else 0
    
    return {
        # Main metrics (current month based)
        "total_visitors": total_visitors,
        "new_active_users": new_active_users,
        "avg_visit_duration": avg_visit_duration,
        "avg_stores_visited": round(avg_stores_visited),
        
        # New lifetime metrics
        "total_registered_users": total_registered_users,
        
        # New monthly metrics
        "current_month_registrations": current_month_registrations,
        "registration_growth_percent": round(registration_growth, 1),
        "monthly_visits": monthly_visits,
        "monthly_visitors": monthly_visitors,
        "monthly_store_visits": monthly_store_visits,
        
        # Calculated rates and percentages
        "monthly_engagement_rate": round(monthly_engagement_rate, 1),
        "avg_stores_per_monthly_visitor": round(avg_stores_per_monthly_visitor, 1),
        "registration_rate": round(registration_rate, 1),
        
        # Previous month data for comparison
        "previous_month_registrations": previous_month_registrations,
    } 