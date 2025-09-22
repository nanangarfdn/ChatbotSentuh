"""
Response generation based on confidence and context
"""
import random
from typing import Dict, List

from src.core.nlp.conversation import ConversationHandler


class ResponseGenerator:
    """Generate appropriate responses based on confidence and context"""

    def __init__(self):
        self.conversation_handler = ConversationHandler()

    def generate_response(self, query: str, matches: List[Dict],
                          confidence_data: Dict, conversation_context: Dict = None) -> Dict:
        """Generate appropriate response based on confidence and matches"""

        confidence_score = confidence_data['score']
        response_type = confidence_data['type']

        if response_type == 'confident':
            return self._generate_confident_response(matches[0], confidence_score)
        elif response_type == 'clarification':
            return self._generate_clarification_response(query, matches, confidence_score)
        else:
            return self._generate_not_found_response(query, confidence_score)

    def _generate_confident_response(self, best_match: Dict, confidence: float) -> Dict:
        """Generate confident response with best match"""
        return {
            'response': best_match['answer'],
            'response_type': 'confident',
            'confidence_score': confidence,
            'sources': [best_match],
            'suggestions': []
        }

    def _generate_clarification_response(self, query: str, matches: List[Dict],
                                         confidence: float) -> Dict:
        """Generate clarification request with suggestions using top 3 similarity scores"""

        # Sort matches by similarity score and get top 3
        sorted_matches = sorted(matches, key=lambda x: x.get('similarity_score', 0), reverse=True)
        top_matches = sorted_matches[:3]

        # Get unique categories from top matches
        categories = list(set([match.get('category', 'Umum') for match in top_matches]))
        categories = [cat for cat in categories if cat and cat.strip() and cat != 'Uncategorized']

        if categories:
            category_text = ", ".join(categories[:3])
            response = f"Saya menemukan beberapa informasi terkait. Apakah Anda bertanya tentang: {category_text}? Bisa lebih spesifik?"
        else:
            response = "Saya menemukan beberapa informasi terkait, tapi bisa lebih spesifik pertanyaannya? Misalnya tentang pendaftaran, sertifikat, atau fitur tertentu."

        # Generate suggestions from top 3 similarity scores (full questions for frontend)
        suggestions = []
        for match in top_matches:
            # Use full question without truncation and without scores for better frontend UX
            suggestions.append(match['question'])

        return {
            'response': response,
            'response_type': 'clarification',
            'confidence_score': confidence,
            'sources': top_matches,
            'suggestions': suggestions
        }

    def _generate_not_found_response(self, query: str, confidence: float, knowledge_data: List[Dict] = None) -> Dict:
        """Generate helpful not found response"""

        responses = [
            "Maaf, saya tidak menemukan informasi spesifik tentang itu. Bisa coba pertanyaan yang lebih spesifik tentang Sentuh Tanahku?",
            "Saya belum menemukan jawaban yang tepat untuk pertanyaan tersebut. Coba tanyakan tentang fitur, cara pendaftaran, atau masalah teknis Sentuh Tanahku.",
            "Informasi tentang itu belum tersedia. Silakan tanyakan hal lain seputar aplikasi Sentuh Tanahku yang bisa saya bantu."
        ]

        response = random.choice(responses)

        # Add helpful suggestions from random questions in the sheet
        suggestions = []
        if knowledge_data:
            # Get random questions from the knowledge base
            random_items = random.sample(knowledge_data, min(4, len(knowledge_data)))
            suggestions = [item['question'] for item in random_items]
        else:
            # Fallback if no data available
            suggestions = [
                "Cara mendaftar di Sentuh Tanahku",
                "Bagaimana cek sertifikat tanah",
                "Fitur utama aplikasi Sentuh Tanahku",
                "Cara menggunakan antrian online"
            ]

        return {
            'response': response,
            'response_type': 'not_found',
            'confidence_score': confidence,
            'sources': [],
            'suggestions': suggestions
        }