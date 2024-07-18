import time
import requests
from embedchain.config.log_conf import logger
from typing import Callable, Any


def retry(retries: int, retry_delay: int):
    """
    Decorator to retry a function call with the specified number of retries and timeout.
    Args:
        retries (int): The number of retries.
        retry_delay (int): The delay between retries in seconds.
    Returns:
        Callable: The decorated function.
    """
    def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(retries):
                try:
                    response = func(*args, **kwargs)
                    response.raise_for_status()
                    return response
                except requests.exceptions.HTTPError as e:
                    logger.info(f"HTTP error occurred on attempt {attempt + 1}: {e}")
                    error = e
                except requests.exceptions.ConnectionError as e:
                    logger.info(f"Connection error occurred on attempt {attempt + 1}: {e}")
                    error = e
                except requests.exceptions.Timeout as e:
                    logger.info(f"Timeout error occurred on attempt {attempt + 1}: {e}")
                    error = e
                except requests.exceptions.RequestException as e:
                    logger.info(f"Request error occurred on attempt {attempt + 1}: {e}")
                    error = e
                except Exception as e:
                    logger.info(f"Unexpected error occurred on attempt {attempt + 1}: {e}")
                    error = e
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to execute function after {retries} attempts: {error}")
                    return False
        return wrapper
    return decorator