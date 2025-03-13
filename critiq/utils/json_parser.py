import json
from typing import Any, TypeVar, Type, Optional

T = TypeVar('T')

class JSONParser:
    """A class for handling JSON parsing with error handling and type conversion."""
    
    @staticmethod
    def parse(text: str) -> dict[str, Any]:
        """Parse JSON text into a dictionary.
        
        Args:
            text: The JSON text to parse
            
        Returns:
            A dictionary containing the parsed JSON data
            
        Raises:
            ValueError: If the JSON text cannot be parsed
        """
        try:
            # Extract JSON object from text
            text = "{" + text.split("{", 1)[-1].strip().rsplit("}", 1)[0].strip() + "}"
            return json.loads(text, strict=False)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {text}") from e
    
    @staticmethod
    def parse_typed(text: str, target_type: Type[T]) -> Optional[T]:
        """Parse JSON text into a specific type.
        
        Args:
            text: The JSON text to parse
            target_type: The type to convert the JSON data into
            
        Returns:
            An instance of the target type, or None if parsing fails
        """
        try:
            data = JSONParser.parse(text)
            return target_type(**data)
        except Exception:
            return None
    
    @staticmethod
    def format(data: Any, indent: int = 4) -> str:
        """Format data as JSON string.
        
        Args:
            data: The data to format
            indent: Number of spaces for indentation
            
        Returns:
            A formatted JSON string
        """
        return json.dumps(data, ensure_ascii=False, indent=indent) 