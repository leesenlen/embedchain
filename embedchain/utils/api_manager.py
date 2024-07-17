import time
import requests
from embedchain.config.log_conf import logger
from typing import Callable, Any

def request_retry(retries: int, timeout: int):
    """
     Decorator to retry a function call with the specified number of retries and timeout.

     Args:
         retries (int): The number of retries.
         timeout (int): The timeout for each retry in seconds.

     Returns:
         Callable: The decorated function.
     """

    def decorator(func: Callable[[], Any]) -> Callable[[], Any]:
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(retries):
                try:
                    response = func(*args, **kwargs)
                    response.raise_for_status()  # Check HTTP response status code
                    return response.json()
                except requests.exceptions.RequestException as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(timeout)
                    else:
                        raise
        return wrapper

    return decorator