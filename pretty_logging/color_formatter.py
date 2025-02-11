import logging
from colorama import Fore, Style, init

# Initialize colorama for cross-platform support
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Custom logging formatter with colored components and proper alignment"""

    LEVEL_COLORS = {
        "DEBUG": Fore.WHITE + Style.DIM,  # Grey (Dim White)
        "INFO": Fore.BLUE,  # Blue
        "WARNING": Fore.YELLOW,  # Yellow
        "ERROR": Fore.RED,  # Red
        "CRITICAL": Fore.RED + Style.BRIGHT  # Bright Red
    }

    DATE_COLOR = Fore.GREEN + Style.BRIGHT  # Bright Green

    def format(self, record):
        # Ensure the timestamp is formatted before applying colors
        record.asctime = self.formatTime(record, self.datefmt)

        # Get log level color
        level_color = self.LEVEL_COLORS.get(record.levelname, Fore.WHITE)

        # Construct the final log message with proper padding
        colored_message = (
            f"{self.DATE_COLOR}{record.asctime}{Style.RESET_ALL} | "
            f"{level_color}{record.levelname:<8}{Style.RESET_ALL} | "
            f"{level_color}{record.name:<15}{Style.RESET_ALL} | "
            f"{level_color}{record.getMessage()}{Style.RESET_ALL}"
        )

        return colored_message

