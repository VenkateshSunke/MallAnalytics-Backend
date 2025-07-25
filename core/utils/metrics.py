"""
Metrics utility functions for visitor analytics.
"""
from datetime import date, datetime
from typing import List, Optional
from collections import defaultdict

# Placeholder types for User, Visit, etc.
# In real usage, import your actual models.


def _get_visit_date(visit):
    """Helper function to extract visit date from visit object or dict."""
    vdate = visit.get('visit_date') if isinstance(visit, dict) else getattr(visit, 'visit_date', None)
    if isinstance(vdate, str):
        return datetime.strptime(vdate, "%Y-%m-%d").date()
    return vdate

def _get_stores_visited(visit):
    """Helper function to extract stores_visited from visit object or dict."""
    return visit.get('stores_visited') if isinstance(visit, dict) else getattr(visit, 'stores_visited', 0)


def average_monthly_visits(visits: List) -> float:
    """
    Calculate the average number of visits per month across all months with visits.
    
    Calculation:
    1. Group visits by month (year-month combination)
    2. Count visits in each month
    3. Calculate average: total_visits / number_of_unique_months
    
    Example: If user visited 3 times in Jan 2024, 5 times in Feb 2024:
    Average = (3 + 5) / 2 = 4.0 visits per month
    
    Returns 0.0 if no visits or no valid dates.
    """
    if not visits:
        return 0.0
    
    months = defaultdict(int)
    for v in visits:
        vdate = _get_visit_date(v)
        if vdate:
            months[(vdate.year, vdate.month)] += 1
    
    if not months:
        return 0.0
    
    return sum(months.values()) / len(months)


def average_yearly_visits(visits: List) -> float:
    """
    Calculate the average number of visits per year across all years with visits.
    
    Calculation:
    1. Group visits by year
    2. Count visits in each year
    3. Calculate average: total_visits / number_of_unique_years
    
    Example: If user visited 12 times in 2023, 18 times in 2024:
    Average = (12 + 18) / 2 = 15.0 visits per year
    
    Returns 0.0 if no visits or no valid dates.
    """
    if not visits:
        return 0.0
    
    years = defaultdict(int)
    for v in visits:
        vdate = _get_visit_date(v)
        if vdate:
            years[vdate.year] += 1
    
    if not years:
        return 0.0
    
    return sum(years.values()) / len(years)


def total_life_visits(visits: List) -> int:
    """
    Calculate the total number of visits across the user's entire history.
    
    Calculation:
    Simply counts the length of the visits list.
    
    Example: If visits list has 50 entries, returns 50.
    """
    return len(visits)


def average_stores_visited_per_month(visits: List) -> float:
    """
    Calculate the average number of stores visited per month across all months with visits.
    
    Calculation:
    1. Group visits by month (year-month combination)
    2. Sum stores_visited for each month
    3. Calculate average: total_stores_across_all_months / number_of_unique_months
    
    Example: 
    - Jan 2024: Visit 1 (2 stores) + Visit 2 (3 stores) = 5 stores
    - Feb 2024: Visit 3 (1 store) + Visit 4 (4 stores) = 5 stores
    Average = (5 + 5) / 2 = 5.0 stores per month
    
    Returns 0.0 if no visits or no valid dates.
    """
    if not visits:
        return 0.0
    
    stores_per_month = defaultdict(int)
    for v in visits:
        vdate = _get_visit_date(v)
        stores = _get_stores_visited(v) or 0
        if vdate:
            stores_per_month[(vdate.year, vdate.month)] += stores
    
    if not stores_per_month:
        return 0.0
    
    return sum(stores_per_month.values()) / len(stores_per_month)


def total_stores_visited_life(visits: List) -> int:
    """
    Calculate the total number of stores visited across the user's entire history.
    
    Calculation:
    Sums the stores_visited field from all visits.
    
    Example: If visits show [2, 3, 1, 4] stores respectively, returns 10.
    Treats None/missing values as 0.
    """
    return sum(_get_stores_visited(v) or 0 for v in visits)


def first_visit_date(visits: List) -> Optional[date]:
    """
    Return the date of the earliest visit.
    
    Calculation:
    1. Extract all valid visit dates
    2. Return the minimum (earliest) date
    
    Returns None if no visits have valid dates.
    """
    dates = [_get_visit_date(v) for v in visits if _get_visit_date(v)]
    return min(dates) if dates else None


def last_visit_date(visits: List) -> Optional[date]:
    """
    Return the date of the most recent visit.
    
    Calculation:
    1. Extract all valid visit dates
    2. Return the maximum (latest) date
    
    Returns None if no visits have valid dates.
    """
    dates = [_get_visit_date(v) for v in visits if _get_visit_date(v)]
    return max(dates) if dates else None


def recency(visits: List, reference_date: Optional[date] = None) -> Optional[int]:
    """
    Calculate recency as the number of days since the last visit.
    
    Calculation:
    1. Find the last visit date
    2. Calculate difference: reference_date - last_visit_date
    3. Return the difference in days
    
    Args:
        reference_date: Date to calculate from (defaults to today)
    
    Example: If last visit was 2024-01-15 and reference_date is 2024-01-20,
    returns 5 (days since last visit).
    
    Returns None if no visits have valid dates.
    """
    last_date = last_visit_date(visits)
    if not last_date:
        return None
    
    if reference_date is None:
        reference_date = date.today()
    
    return (reference_date - last_date).days


def monthly_frequency(visits: List) -> float:
    """
    Calculate the monthly frequency of visits (visits per month).
    
    This is an alias for average_monthly_visits() for semantic clarity.
    See average_monthly_visits() for detailed calculation explanation.
    """
    return average_monthly_visits(visits)


def visit_frequency_over_timespan(visits: List, reference_date: Optional[date] = None) -> Optional[float]:
    """
    Calculate visit frequency as visits per month over the entire active timespan.
    
    Calculation:
    1. Find first and last visit dates
    2. Calculate total months in timespan: (last_date - first_date) in months
    3. Calculate frequency: total_visits / total_months_in_timespan
    
    This differs from average_monthly_visits by considering the entire timespan,
    including months with zero visits.
    
    Example: 10 visits between Jan 2024 and Jun 2024 (6 months total)
    = 10 / 6 = 1.67 visits per month
    
    Returns None if timespan cannot be determined.
    """
    if not visits:
        return None
    
    first_date = first_visit_date(visits)
    last_date = last_visit_date(visits)
    
    if not first_date or not last_date:
        return None
    
    if reference_date is None:
        reference_date = date.today()
    
    # Use the later of last_date or reference_date for end calculation
    end_date = max(last_date, reference_date)
    
    # Calculate months between first visit and end date
    months_diff = (end_date.year - first_date.year) * 12 + (end_date.month - first_date.month)
    
    # Add 1 to include the first month
    total_months = months_diff + 1
    
    if total_months <= 0:
        return None
    
    return len(visits) / total_months


def print_user_metrics(user):
    """
    Print comprehensive metrics for a user's visit history.
    
    Displays all calculated metrics in a formatted output including:
    - Basic info (first/last visit, recency)
    - Visit counts (total, monthly/yearly averages)
    - Store visit metrics
    - Frequency calculations
    """
    # If user is a model instance, fetch visits using ORM
    visits = list(user.visits.all())
    
    print(f"User: {user.name} (ID: {user.user_id})")
    print(f"First Visit: {first_visit_date(visits)}")
    print(f"Last Visit: {last_visit_date(visits)}")
    print(f"Recency (days since last visit): {recency(visits)}")
    print(f"Total Life Visits: {total_life_visits(visits)}")
    print(f"Average Monthly Visits: {average_monthly_visits(visits):.2f}")
    print(f"Average Yearly Visits: {average_yearly_visits(visits):.2f}")
    print(f"Average Stores Visited per Month: {average_stores_visited_per_month(visits):.2f}")
    print(f"Total Stores Visited in Life: {total_stores_visited_life(visits)}")
    print(f"Monthly Frequency: {monthly_frequency(visits):.2f}")
    print(f"Visit Frequency Over Timespan: {visit_frequency_over_timespan(visits):.2f}" 
          if visit_frequency_over_timespan(visits) else "Visit Frequency Over Timespan: N/A")


def get_all_metrics(visits):
    """
    Returns a dictionary of all calculated metrics for a list of visits.
    """
    return {
        "first_visit": str(first_visit_date(visits)) if first_visit_date(visits) else None,
        "last_visit": str(last_visit_date(visits)) if last_visit_date(visits) else None,
        "recency": recency(visits),
        "total_life_visits": total_life_visits(visits),
        "average_monthly_visits": round(average_monthly_visits(visits), 2),
        "average_yearly_visits": round(average_yearly_visits(visits), 2),
        "average_stores_visited_per_month": round(average_stores_visited_per_month(visits), 2),
        "total_stores_visited_life": total_stores_visited_life(visits),
        "monthly_frequency": round(monthly_frequency(visits), 2),
        "visit_frequency_over_timespan": round(visit_frequency_over_timespan(visits), 2) if visit_frequency_over_timespan(visits) is not None else None,
    }