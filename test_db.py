from database import db

def test_connection():
    try:
        print("Testing database connection...")
        
        # Test basic connection
        faqs = db.get_all_faqs()
        print(f"✅ Connection successful! Found {len(faqs)} FAQs")
        
        # Test search
        if faqs:
            search_result = db.search_faqs("chatbot")
            print(f"✅ Search test: Found {len(search_result)} results for 'chatbot'")

        
        print("All tests passed!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Check your .env file and PostgreSQL service")

if __name__ == "__main__":
    test_connection()