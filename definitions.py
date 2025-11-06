"""
Dagster Definitions
"""

from dagster import Definitions, load_assets_from_modules
import dagster_assets

# Load all assets
all_assets = load_assets_from_modules([dagster_assets])

# Create definitions
defs = Definitions(
    assets=all_assets,
)
