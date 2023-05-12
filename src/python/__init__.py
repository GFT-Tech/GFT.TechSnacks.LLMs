import logging

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()

# Set the logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)