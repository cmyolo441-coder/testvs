"""Explicit campaign facade. Not imported by ordinary routing or tool startup."""
from .work_campaign_contract import (
    CampaignError, WORK_CAMPAIGN_SCHEMA_VERSION, build_work_campaign, kanban_task_manifests,
    leaf_contract_text,
)
from .work_campaign_controller import Campaign
from .work_campaign_metrics import campaign_canary_report
from .work_campaign_routes import resolve_campaign_routes

__all__ = ["Campaign", "CampaignError", "WORK_CAMPAIGN_SCHEMA_VERSION", "build_work_campaign",
           "kanban_task_manifests", "leaf_contract_text", "campaign_canary_report", "resolve_campaign_routes"]
