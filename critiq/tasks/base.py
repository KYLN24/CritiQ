from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Type

from ..types import DataType, EvaluationOutput, PredictionOutput, PredictionOutputWithAnswer

InputT = TypeVar('InputT', bound=DataType)
OutputT = TypeVar('OutputT', bound=Any)

class Task(ABC, Generic[InputT, OutputT]):
    """Base class for all tasks in the system.
    
    This class defines the common interface and functionality for all tasks:
    1. Input processing
    2. Output generation
    3. Judgment based on criteria
    4. Metric calculation
    5. Voting function
    """
    
    def __init__(self, data_type: Type[InputT]):
        self.data_type = data_type
        
    @abstractmethod
    def process_input(self, data: InputT) -> Any:
        """Process the input data into a format suitable for judgment.
        
        Args:
            data: The input data to process
            
        Returns:
            Processed data in a format suitable for judgment
        """
        pass
    
    @abstractmethod
    def generate_output(self, judgment: Any) -> OutputT:
        """Generate the final output from the judgment.
        
        Args:
            judgment: The judgment result
            
        Returns:
            The final output in the expected format
        """
        pass
    
    @abstractmethod
    def judge(self, data: InputT, criterion: str) -> Any:
        """Judge the input data based on the given criterion.
        
        Args:
            data: The input data to judge
            criterion: The criterion to judge against
            
        Returns:
            The judgment result
        """
        pass
    
    @abstractmethod
    def metric(self, judgment: Any, output: OutputT) -> float:
        """Calculate a metric between the judgment and expected output.
        
        Args:
            judgment: The judgment result
            output: The expected output
            
        Returns:
            A float metric value
        """
        pass
    
    @abstractmethod
    def voting_fn(self, judgments: list[Any]) -> OutputT:
        """Combine multiple judgments into a final output.
        
        Args:
            judgments: List of judgment results
            
        Returns:
            The final combined output
        """
        pass
    
    def evaluate(self, data: InputT, criteria: list[str]) -> EvaluationOutput:
        """Evaluate the input data against multiple criteria.
        
        Args:
            data: The input data to evaluate
            criteria: List of criteria to evaluate against
            
        Returns:
            EvaluationOutput containing the results
        """
        predictions = []
        thoughts = []
        
        for criterion in criteria:
            judgment = self.judge(data, criterion)
            predictions.append(judgment)
            # Note: thoughts are optional and can be added by subclasses
            
        final_output = self.voting_fn(predictions)
        accuracy = self.metric(final_output, data.get("output"))
        
        return EvaluationOutput(
            prediction=predictions,
            is_correct=[accuracy > 0.5],  # Simple threshold, can be customized
            per_criterion_acc={c: self.metric(p, data.get("output")) for c, p in zip(criteria, predictions)},
            accuracy=accuracy,
            thoughts=thoughts if thoughts else None
        ) 