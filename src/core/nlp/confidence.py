"""
Confidence evaluation for query responses
"""
from typing import Dict


class ConfidenceEvaluator:
    """Evaluate confidence levels and determine response strategy"""

    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }

    def evaluate_confidence(self, vector_score: float, nlp_score: float,
                            exact_match: bool = False) -> Dict:
        """Evaluate confidence and determine response type"""

        # Combine scores with weights
        if vector_score > 0:
            combined_score = (vector_score * 0.7) + (nlp_score * 0.3)
        else:
            combined_score = nlp_score

        # Boost for exact matches
        if exact_match:
            combined_score = min(combined_score + 0.2, 1.0)

        # Determine response type
        if combined_score >= self.confidence_thresholds['high']:
            response_type = 'confident'
        elif combined_score >= self.confidence_thresholds['medium']:
            response_type = 'clarification'
        else:
            response_type = 'not_found'

        return {
            'score': combined_score,
            'type': response_type,
            'vector_score': vector_score,
            'nlp_score': nlp_score
        }