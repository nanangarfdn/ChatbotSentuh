"""
Conversation handling for greetings, clarifications, and goodbyes
"""
import random


class ConversationHandler:
    """Handle conversational aspects like greetings and clarifications"""

    def __init__(self):
        self.greetings = [
            'halo', 'hai', 'hello', 'selamat', 'assalamualaikum', 'salam',
            'pagi', 'siang', 'sore', 'malam', 'good', 'morning'
        ]

        self.clarification_keywords = [
            'lebih detail', 'spesifik', 'jelaskan', 'maksudnya', 'artinya',
            'contoh', 'bagaimana', 'gimana', 'caranya', 'tolong'
        ]

        self.goodbye_keywords = [
            'terima kasih', 'makasih', 'thanks', 'bye', 'sampai jumpa', 'selesai'
        ]

    def is_greeting(self, text: str) -> bool:
        """Detect if text is a greeting"""
        text_lower = text.lower()
        return any(greeting in text_lower for greeting in self.greetings)

    def is_clarification_request(self, text: str) -> bool:
        """Detect if text is asking for clarification"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.clarification_keywords)

    def is_goodbye(self, text: str) -> bool:
        """Detect if text is a goodbye"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.goodbye_keywords)

    def generate_greeting_response(self) -> str:
        """Generate friendly greeting response"""
        responses = [
            "Halo! Saya asisten Sentuh Tanahku. Ada yang bisa saya bantu tentang aplikasi Sentuh Tanahku?",
            "Hai! Selamat datang di layanan bantuan Sentuh Tanahku. Silakan tanyakan apa saja tentang aplikasi ini.",
            "Halo! Saya siap membantu Anda dengan pertanyaan seputar Sentuh Tanahku. Ada yang ingin ditanyakan?"
        ]
        return random.choice(responses)

    def generate_goodbye_response(self) -> str:
        """Generate friendly goodbye response"""
        responses = [
            "Terima kasih telah menggunakan layanan Sentuh Tanahku! Semoga informasinya bermanfaat.",
            "Sama-sama! Jangan ragu untuk bertanya lagi kapan saja tentang Sentuh Tanahku.",
            "Senang bisa membantu! Sampai jumpa dan semoga sukses dengan urusan pertanahan Anda."
        ]
        return random.choice(responses)