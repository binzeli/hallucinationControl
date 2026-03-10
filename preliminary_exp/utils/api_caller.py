import os
import asyncio
import aiohttp
import json


async def get_response_async(api_url, payload, api_key, headers=None, include_log_prob=True, return_raw_response=False, max_retries=5):
    """
    Send a POST request to the specified API asynchronously and return the response.
    Uses /v1/chat/completions format (gpt-4o-mini). Includes retry logic for rate limiting (429 errors).
    Run scripts (run_PopQA, run_MMLU-pro) process raw responses themselves.

    :param api_url: URL of the API endpoint.
    :param payload: Data to send in the POST request.
    :param api_key: API key for authentication (passed from run script, typically from env).
    :param headers: Optional headers for the request. If not provided, uses Authorization: Bearer {api_key}.
    :param include_log_prob: Boolean indicating whether to include log-probabilities in the response.
    :param return_raw_response: Boolean indicating whether to return raw API responses without processing.
    :param max_retries: Maximum number of retry attempts for rate limiting errors.
    :return: Raw API response (parsed JSON).
    """
    headers = headers or {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Add model from prompt parameters if not already in payload
    if "model" not in payload:
        payload["model"] = os.getenv("PROMPT_MODEL", "gpt-4o-mini-2024-07-18")  # Default to gpt-4 if not set
    
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers) as response:
                    # Handle rate limiting (429) with retry
                    if response.status == 429:
                        if attempt < max_retries:
                            # Try to get Retry-After header, default to exponential backoff (seconds)
                            retry_after_str = response.headers.get('Retry-After', None)
                            base_wait = 5  # seconds
                            if retry_after_str:
                                try:
                                    wait_time = int(retry_after_str)
                                except ValueError:
                                    wait_time = base_wait * (2 ** attempt)  # Fallback to exponential backoff
                            else:
                                wait_time = base_wait * (2 ** attempt)  # Exponential backoff

                            print(f"Rate limit hit (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            response.raise_for_status()  # Raise error if max retries reached
                    
                    response.raise_for_status()
                    raw_response = await response.text()
                    parsed_response = json.loads(raw_response)
                    return parsed_response
        except aiohttp.ClientResponseError as e:
            if e.status == 429 and attempt < max_retries:
                # Fallback handling for rate limit errors when we don't have a response object
                base_wait = 5  # seconds
                wait_time = base_wait * (2 ** attempt)
                print(f"Rate limit error (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)
                continue
            else:
                # Print more detailed error information
                error_text = ""
                try:
                    if hasattr(e, 'message'):
                        error_text = e.message
                except:
                    pass
                print(f"ClientResponseError: {e.status}, message={error_text}, url={e.request_info.url}")
                if e.status == 404:
                    print(f"⚠️  404 Not Found - This usually means:")
                    print(f"   1. The model name '{payload.get('model', 'unknown')}' may not exist")
                    print(f"   2. The API endpoint '{api_url}' may be incorrect")
                    print(f"   3. Your API key may not have access to this model")
                raise
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff for other errors
                print(f"Error occurred: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise

async def get_responses_concurrently_with_ids(api_url, payloads, request_ids, api_key, headers=None, include_log_prob=True, return_raw_response=False, max_concurrent=5):
    """
    Send multiple POST requests to the specified API concurrently and return the responses with their request IDs.
    :param api_url: URL of the API endpoint.
    :param payloads: List of data to send in the POST requests.
    :param request_ids: List of unique request IDs corresponding to each payload.
    :param api_key: API key for authentication (passed from run script, typically from env).
    :param headers: Optional headers for the requests.
    :param include_log_prob: Boolean indicating whether to include log-probabilities in the responses.
    :param return_raw_response: Boolean indicating whether to return raw API responses without processing.
    :param max_concurrent: Maximum number of concurrent requests.
    :return: List of dicts with request_id and raw_response (or error). Run scripts process responses themselves.
    """
    if len(payloads) != len(request_ids):
        raise ValueError("The number of payloads must match the number of request IDs.")

    # Limit the number of concurrent requests with a semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    async def send_request_with_id(payload, request_id):
        async with semaphore:
            try:
                response = await get_response_async(api_url, payload, api_key, headers, include_log_prob, return_raw_response)
                if return_raw_response:
                    return {"request_id": request_id, "raw_response": response}
                if response is not None:
                    response["request_id"] = request_id  # Attach the request ID to the processed response
                return response
            except Exception as e:
                print(f"Error sending request with ID {request_id}: {e}")
                return {"request_id": request_id, "error": str(e)}

    tasks = [send_request_with_id(payload, request_id) for payload, request_id in zip(payloads, request_ids)]
    return await asyncio.gather(*tasks)