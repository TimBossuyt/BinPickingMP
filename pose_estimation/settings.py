import json
import logging

logger = logging.getLogger("SettingsManager")


class SettingsManager:
    def __init__(self, sConfigFile: str):
        self.sPath = sConfigFile
        self.settings = self._loadSettings()

    def reload_settings(self):
        logger.info("Reloading settings")
        self.settings = self._loadSettings()

    def _loadSettings(self) -> dict:
        try:
            with open(self.sPath, 'r') as f:
                settings = json.load(f)
                logger.info("Settings loaded")

                return settings
        except FileNotFoundError:
            logger.error("Config file not found")

    def get(self, key: str):
        keys = key.split(".")
        value = self.settings
        try:
            for k in keys:
                value = value[k]

            return value
        except KeyError:
            logger.error(f"Settings key ({key}) not found:")
