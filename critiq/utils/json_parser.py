import json
import json5
import logging
import traceback
import concurrent.futures
import re
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Type, Union

from ..agent import Agent

T = TypeVar('T')

class ResponseJSONParser:
    """A robust JSON parser that handles various error cases and supports multiple parsing strategies."""
    
    def __init__(
        self,
        model_config: Dict[str, Any] = None,
        max_retries: int = 5,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 5
    ):
        """Initialize the JSON parser.
        
        Args:
            model_config: Model configuration parameters
            max_retries: Maximum number of retry attempts
            logger: Logger instance
            max_workers: Maximum number of worker threads
        """
        # Default model configuration matching config_local.py
        self.model_config = {
            "model": "Qwen2.5-32B-Instruct",
            "api_keys": "EMPTY",
            "base_url": "http://10.130.1.235:30000/v1",
        }
        
        # Update model configuration
        if model_config:
            self.model_config.update(model_config)
            
        self.max_retries = max_retries
        self.max_workers = max_workers
        
        # Setup logger
        if logger:
            self.logger = logger
        else:
            import os
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.logger = logging.getLogger('ResponseJSONParser')
            handler = logging.FileHandler('json_parser.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text.
        
        Supports multiple formats:
        1. JSON in ```json ... ``` code blocks
        2. JSON objects in { ... } format
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON object or None if not found/parse failed
        """
        self.logger.info(f"Attempting to extract JSON from text: {text[:100]}...")
        
        # Try to extract JSON from markdown code blocks
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_block_pattern, text)
        
        for match in matches:
            try:
                result = json.loads(match)
                self.logger.info("Successfully extracted and parsed JSON from code block")
                return result
            except json.JSONDecodeError:
                try:
                    result = json5.loads(match)
                    self.logger.info("Successfully extracted and parsed JSON using json5")
                    return result
                except Exception as e:
                    self.logger.debug(f"Code block parsing failed: {str(e)}")
                    continue
        
        # Try to extract JSON directly from text
        brace_pattern = r'(\{[\s\S]*?\})'
        matches = re.findall(brace_pattern, text)
        
        for match in matches:
            try:
                result = json.loads(match)
                self.logger.info("Successfully extracted and parsed JSON from text")
                return result
            except json.JSONDecodeError:
                try:
                    result = json5.loads(match)
                    self.logger.info("Successfully extracted and parsed JSON using json5")
                    return result
                except Exception as e:
                    self.logger.debug(f"Direct extraction parsing failed: {str(e)}")
                    continue
        
        self.logger.warning("Failed to extract valid JSON from text")
        return None
    
    def parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response with multiple fallback strategies.
        
        Args:
            response: Response string containing JSON
            
        Returns:
            Parsed JSON dictionary or None if all attempts fail
        """
        self.logger.info(f"Starting to parse response: {response[:100]}...")
        
        # Strategy 1: Direct JSON parsing
        try:
            result = json.loads(response)
            self.logger.info("Direct parsing successful")
            return result
        except json.JSONDecodeError:
            self.logger.info("Direct parsing failed, trying json5")
        
        # Strategy 2: JSON5 parsing
        try:
            result = json5.loads(response)
            self.logger.info("JSON5 parsing successful")
            return result
        except Exception:
            self.logger.info("JSON5 parsing failed, trying text extraction")
        
        # Strategy 3: Extract JSON from text
        result = self.extract_json_from_text(response)
        if result:
            return result
        
        # Strategy 4: Retry with Agent
        return self._retry_with_agent(response)
    
    def _retry_with_agent(self, response: str) -> Optional[Dict[str, Any]]:
        """Use Agent to fix and parse JSON with multiple retries.
        
        Args:
            response: Response string containing JSON
            
        Returns:
            Parsed JSON dictionary or None if all attempts fail
        """
        for attempt in range(self.max_retries):
            self.logger.info(f"Starting retry {attempt+1}/{self.max_retries}")
            try:
                retry_worker = Agent(**self.model_config)
                
                fix_prompt = f"""
                The following text contains a JSON object that needs to be extracted and fixed.
                Extract the valid JSON object and fix any syntax errors.
                Return only the fixed JSON, nothing else.
                
                Text:
                {response}
                """
                
                fixed_json_text = retry_worker(fix_prompt, stream=False)
                
                if not fixed_json_text:
                    self.logger.warning(f"Retry {attempt+1} returned empty response")
                    continue
                
                try:
                    result = json.loads(fixed_json_text)
                    self.logger.info(f"Retry {attempt+1} successfully parsed JSON")
                    return result
                except json.JSONDecodeError:
                    try:
                        result = json5.loads(fixed_json_text)
                        self.logger.info(f"Retry {attempt+1} successfully parsed with json5")
                        return result
                    except Exception as parse_err:
                        self.logger.warning(f"Retry {attempt+1} parsing failed: {str(parse_err)}")
            except Exception as retry_err:
                self.logger.warning(f"Retry {attempt+1} failed: {str(retry_err)}")
        
        self.logger.error("All parsing attempts failed")
        return None
    
    def _parse_response_with_index(self, indexed_response: Tuple[int, str]) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Parse indexed response for maintaining order in multi-threaded processing.
        
        Args:
            indexed_response: Tuple of (index, response string)
            
        Returns:
            Tuple of (index, parse result)
        """
        index, response = indexed_response
        result = self.parse_response(response)
        return index, result
    
    def parse_responses(self, responses: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Parse multiple responses in parallel while maintaining order.
        
        Args:
            responses: List of response strings
            
        Returns:
            List of parsed JSON dictionaries in original order
        """
        indexed_responses = list(enumerate(responses))
        results = [None] * len(responses)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._parse_response_with_index, item): item[0] 
                for item in indexed_responses
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    index, result = future.result()
                    results[index] = result
                    
                    print(f"\nProcessing response {index+1}/{len(responses)}:")
                    if result:
                        print(f"Successfully parsed JSON: {result}")
                    else:
                        print(f"Failed to parse response {index+1}")
                        
                except Exception as e:
                    index = future_to_index[future]
                    print(f"Error processing response {index+1}: {str(e)}")
                    self.logger.error(f"Error processing response {index+1}: {str(e)}")
        
        return results
    
    @staticmethod
    def format(data: Any, indent: int = 4) -> str:
        """Format data as JSON string.
        
        Args:
            data: Data to format
            indent: Number of spaces for indentation
            
        Returns:
            Formatted JSON string
        """
        return json.dumps(data, ensure_ascii=False, indent=indent) 

def parse_json(text: str) -> dict[str, Any]:
    parser = ResponseJSONParser()
    return parser.parse_response(text)
