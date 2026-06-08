"""The HA AI conversation integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .api import detect_and_validate_client

PLATFORMS = (Platform.CONVERSATION,)

type HAAIConversationConfigEntry = ConfigEntry


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: HAAIConversationConfigEntry) -> bool:
    """Set up integration from config entry."""
    try:
        client = await detect_and_validate_client(hass, entry.data)
    except ValueError as err:
        raise ConfigEntryNotReady(str(err)) from err

    entry.runtime_data = client
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload integration."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)