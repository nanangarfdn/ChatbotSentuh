import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.host = os.getenv('DB_HOST')
        self.port = int(os.getenv('DB_PORT'))
        self.database = os.getenv('DB_NAME')
        self.user = os.getenv('DB_USER',)
        self.password = os.getenv('DB_PASSWORD')
        self.ssl_mode = os.getenv('DB_SSL_MODE')
        self.connection_timeout = int(os.getenv('DB_CONNECTION_TIMEOUT'))
        
        if not self.password:
            raise ValueError("Database password is required. Please set DB_PASSWORD in .env file")
        
        self.connection_pool = None
        self._create_pool()
    
    def _create_pool(self):
        try:
            connection_params = {
                'host': self.host,
                'port': self.port,
                'database': self.database,
                'user': self.user,
                'password': self.password,
                'sslmode': self.ssl_mode,
                'connect_timeout': self.connection_timeout,
                'application_name': 'chatbot_app'
            }
            
            self.connection_pool = SimpleConnectionPool(
                minconn=2,
                maxconn=20,  # Increased for bulk operations
                **connection_params
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Error creating connection pool: {e}")
            raise e
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        self.connection_pool.putconn(conn)
    
    def execute_query(self, query, params=None, fetch=False):
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Input sanitization - basic SQL injection prevention
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                # For RETURNING queries, fetch first then commit
                result = cursor.fetchall()
                conn.commit()
                cursor.close()
                self.return_connection(conn)
                return result
            else:
                # For non-returning queries, commit and return rowcount
                conn.commit()
                rowcount = cursor.rowcount
                cursor.close()
                self.return_connection(conn)
                return rowcount
        
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)
            raise e
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Unexpected error: {e}")
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)
            raise e
    
    def get_all_faqs(self):
        query = "SELECT * FROM faqs ORDER BY id"
        return self.execute_query(query, fetch=True)
    
    def search_faqs(self, keyword):
        if not keyword or len(keyword.strip()) < 2:
            raise ValueError("Search keyword must be at least 2 characters")
        
        # Sanitize keyword input
        sanitized_keyword = keyword.strip()[:100]  # Limit length
        
        query = """
        SELECT * FROM faqs 
        WHERE question ILIKE %s OR answer ILIKE %s 
        ORDER BY id
        """
        return self.execute_query(query, (f'%{sanitized_keyword}%', f'%{sanitized_keyword}%'), fetch=True)
    
    def add_faq(self, question, answer, category=None):
        if not question or not answer:
            raise ValueError("Question and answer are required")
        
        # Sanitize inputs
        question = question.strip()[:500]  # Limit length
        answer = answer.strip()[:2000]     # Limit length
        if category:
            category = category.strip()[:100]  # Limit length
        
        query = """
        INSERT INTO faqs (question, answer, category) 
        VALUES (%s, %s, %s) RETURNING id
        """
        result = self.execute_query(query, (question, answer, category), fetch=True)
        return result[0]['id'] if result else None
    
    def bulk_add_faqs(self, faqs_data):
        """Optimized bulk insert FAQs using batch processing"""
        if not faqs_data:
            return {"successful": 0, "failed": 0, "errors": []}
        
        successful = 0
        failed = 0
        errors = []
        batch_size = 100  # Process in batches of 100
        
        print(f"Starting optimized bulk insert for {len(faqs_data)} items in batches of {batch_size}...")
        
        # Process in batches for better performance
        for batch_start in range(0, len(faqs_data), batch_size):
            batch_end = min(batch_start + batch_size, len(faqs_data))
            batch = faqs_data[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: items {batch_start+1}-{batch_end}")
            
            # Use single transaction for each batch
            conn = None
            cursor = None
            batch_successful = 0
            batch_failed = 0
            
            try:
                conn = self.get_connection()
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                for i, faq in enumerate(batch):
                    try:
                        question = faq.get('question', '').strip()
                        answer = faq.get('answer', '').strip()
                        category = faq.get('category', 'Umum').strip()
                        
                        if not question or not answer:
                            batch_failed += 1
                            errors.append(f"Empty question or answer: {question[:50]}...")
                            continue
                        
                        # Sanitize inputs
                        question = question[:500]  # Limit length
                        answer = answer[:2000]     # Limit length
                        category = category[:100] if category else 'Umum'
                        
                        # Insert within the same transaction
                        cursor.execute("""
                            INSERT INTO faqs (question, answer, category) 
                            VALUES (%s, %s, %s)
                        """, (question, answer, category))
                        
                        batch_successful += 1
                        
                    except Exception as e:
                        batch_failed += 1
                        errors.append(f"Error inserting FAQ in batch: {str(e)}")
                
                # Commit the entire batch
                conn.commit()
                successful += batch_successful
                failed += batch_failed
                
                print(f"Batch completed: {batch_successful} successful, {batch_failed} failed")
                
            except Exception as e:
                # Rollback batch on error
                if conn:
                    conn.rollback()
                failed += len(batch) - batch_successful
                errors.append(f"Batch {batch_start//batch_size + 1} failed: {str(e)}")
                print(f"Batch {batch_start//batch_size + 1} failed: {str(e)}")
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    self.return_connection(conn)
        
        print(f"Optimized bulk insert completed: {successful} successful, {failed} failed")
        
        return {
            "successful": successful,
            "failed": failed,
            "errors": errors[:20]  # Limit errors to first 20 for response size
        }
    
    def bulk_add_faqs_single_transaction(self, faqs_data):
        """Ultra-fast bulk insert using single transaction with COPY or executemany"""
        if not faqs_data:
            return {"successful": 0, "failed": 0, "errors": []}
        
        successful = 0
        failed = 0
        errors = []
        
        print(f"Starting ultra-fast bulk insert for {len(faqs_data)} items...")
        
        # Prepare and validate data first
        valid_faqs = []
        for i, faq in enumerate(faqs_data):
            try:
                question = faq.get('question', '').strip()
                answer = faq.get('answer', '').strip()
                category = faq.get('category', 'Umum').strip()
                
                if not question or not answer:
                    failed += 1
                    errors.append(f"Empty question or answer: {question[:50]}...")
                    continue
                
                # Sanitize inputs
                question = question[:500]
                answer = answer[:2000]
                category = category[:100] if category else 'Umum'
                
                valid_faqs.append((question, answer, category))
                
            except Exception as e:
                failed += 1
                errors.append(f"Error validating FAQ #{i+1}: {str(e)}")
        
        if not valid_faqs:
            return {"successful": 0, "failed": failed, "errors": errors}
        
        # Single transaction bulk insert
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Use executemany for bulk insert (faster than individual inserts)
            cursor.executemany("""
                INSERT INTO faqs (question, answer, category) 
                VALUES (%s, %s, %s)
            """, valid_faqs)
            
            conn.commit()
            successful = len(valid_faqs)
            
            print(f"Ultra-fast bulk insert completed: {successful} items inserted in single transaction")
            
        except Exception as e:
            if conn:
                conn.rollback()
            failed = len(valid_faqs)
            successful = 0
            errors.append(f"Bulk transaction failed: {str(e)}")
            print(f"Bulk transaction failed: {str(e)}")
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)
        
        return {
            "successful": successful,
            "failed": failed,
            "errors": errors[:10]
        }
    
    def close_pool(self):
        if self.connection_pool:
            self.connection_pool.closeall()

db = DatabaseConnection()