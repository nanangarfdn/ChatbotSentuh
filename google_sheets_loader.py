#!/usr/bin/env python3
"""
Google Sheets Loader for Sentuh Tanahku LLM
Handles real-time data loading from Google Sheets using API Key
"""

import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleSheetsLoader:
    """
    Loads data from Google Sheets using API Key authentication
    Supports real-time data fetching and caching
    """
    
    def __init__(self):
        """
        Initialize Google Sheets Loader with API Key authentication
        """
        self.sheets_id = os.getenv('GOOGLE_SHEETS_ID')
        self.sheets_range = os.getenv('GOOGLE_SHEETS_RANGE', 'Sheet1!A:C')
        self.api_key = os.getenv('GOOGLE_API')
        
        self.service = None
        self.last_sync = None
        self.cache_duration = int(os.getenv('SHEETS_CACHE_DURATION', 300))  # 5 minutes default
        
        # Cache for storing fetched data
        self._cached_data = None
        self._cache_timestamp = None
        
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Sheets API using API Key"""
        try:
            if not self.api_key:
                raise ValueError("Google API key not found. Please set GOOGLE_API in .env file")
            
            logger.info("Using Google API Key authentication")
            self.service = build('sheets', 'v4', developerKey=self.api_key)
            logger.info("Google Sheets API authenticated successfully with API Key")
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if not self._cached_data or not self._cache_timestamp:
            return False
        
        cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
        return cache_age < self.cache_duration
    
    def fetch_data(self, use_cache: bool = True) -> List[List]:
        """
        Fetch data from Google Sheets
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List of rows from Google Sheets
        """
        try:
            # Check cache first
            if use_cache and self._is_cache_valid():
                logger.info("Using cached Google Sheets data")
                return self._cached_data
            
            if not self.sheets_id:
                raise ValueError("Google Sheets ID not configured. Please set GOOGLE_SHEETS_ID in .env")
            
            logger.info(f"Fetching data from Google Sheets: {self.sheets_id}")
            
            # Call Google Sheets API
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=self.sheets_id,
                range=self.sheets_range
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                logger.warning("No data found in Google Sheets")
                return []
            
            # Update cache
            self._cached_data = values
            self._cache_timestamp = datetime.now()
            self.last_sync = datetime.now()
            
            logger.info(f"Successfully fetched {len(values)} rows from Google Sheets")
            return values
            
        except HttpError as e:
            logger.error(f"Google Sheets API error: {str(e)}")
            if e.resp.status == 403:
                logger.error("Permission denied. Make sure the Google Sheet is public and API key has proper restrictions.")
            raise
        except Exception as e:
            logger.error(f"Error fetching Google Sheets data: {str(e)}")
            raise
    
    def convert_to_dataframe(self, data: List[List] = None) -> pd.DataFrame:
        """
        Convert Google Sheets data to pandas DataFrame
        
        Args:
            data: Raw data from Google Sheets. If None, fetches fresh data
            
        Returns:
            pandas DataFrame with proper column names
        """
        if data is None:
            data = self.fetch_data()
        
        if not data:
            return pd.DataFrame()
        
        # Use first row as headers
        if len(data) > 1:
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
        else:
            # If only one row, treat as data without headers
            df = pd.DataFrame(data)
        
        # Clean column names to match expected format
        if 'Kategori' in df.columns or 'kategori' in df.columns:
            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'kategori' in col_lower:
                    column_mapping[col] = 'Kategori'
                elif 'pertanyaan' in col_lower or 'question' in col_lower:
                    column_mapping[col] = 'Pertanyaan'
                elif 'jawaban' in col_lower or 'answer' in col_lower:
                    column_mapping[col] = 'Jawaban'
            
            df = df.rename(columns=column_mapping)
        
        logger.info(f"Converted to DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        return df
    
    def get_knowledge_data(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get knowledge data in the format expected by the system
        
        Args:
            force_refresh: Force refresh from Google Sheets
            
        Returns:
            List of dictionaries with category, question, and answer
        """
        try:
            # Fetch data
            raw_data = self.fetch_data(use_cache=not force_refresh)
            df = self.convert_to_dataframe(raw_data)
            
            if df.empty:
                logger.warning("No data available from Google Sheets")
                return []
            
            # Convert to expected format
            knowledge_data = []
            required_columns = ['Kategori', 'Pertanyaan', 'Jawaban']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.info(f"Available columns: {list(df.columns)}")
                raise ValueError(f"Google Sheets must contain columns: {required_columns}")
            
            for _, row in df.iterrows():
                if pd.notna(row['Pertanyaan']) and pd.notna(row['Jawaban']):
                    knowledge_data.append({
                        'category': str(row['Kategori']) if pd.notna(row['Kategori']) else 'General',
                        'question': str(row['Pertanyaan']).strip(),
                        'answer': str(row['Jawaban']).strip()
                    })
            
            logger.info(f"Processed {len(knowledge_data)} valid knowledge items")
            return knowledge_data
            
        except Exception as e:
            logger.error(f"Error processing knowledge data: {str(e)}")
            raise
    
    def test_connection(self) -> Dict[str, any]:
        """
        Test connection to Google Sheets
        
        Returns:
            Dictionary with connection status and info
        """
        try:
            # Try to fetch just the first few rows
            test_range = self.sheets_range.split('!')[0] + '!A1:C3'  # Get first 3 rows
            
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=self.sheets_id,
                range=test_range
            ).execute()
            
            values = result.get('values', [])
            
            return {
                'status': 'connected',
                'sheets_id': self.sheets_id,
                'range': self.sheets_range,
                'sample_rows': len(values),
                'last_sync': self.last_sync.isoformat() if self.last_sync else None,
                'cache_valid': self._is_cache_valid(),
                'auth_method': 'API Key'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'sheets_id': self.sheets_id,
                'range': self.sheets_range
            }
    
    def _check_write_permissions(self) -> bool:
        """
        Check if current authentication supports write operations
        
        Returns:
            True if write operations are supported
        """
        try:
            # Try a simple operation that requires write permissions
            # This will fail gracefully if only API key is used
            sheet = self.service.spreadsheets()
            
            # Try to get sheet metadata first
            spreadsheet = sheet.get(spreadsheetId=self.sheets_id).execute()
            
            # If we can read, try a test write operation
            # We'll attempt to update a cell with the same value (no real change)
            test_range = 'Sheet1!A1'
            result = sheet.values().get(
                spreadsheetId=self.sheets_id,
                range=test_range
            ).execute()
            
            # Try to write back the same value
            values = result.get('values', [['']])
            body = {'values': values}
            
            sheet.values().update(
                spreadsheetId=self.sheets_id,
                range=test_range,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info("Write permissions verified - can log user questions")
            return True
            
        except Exception as e:
            if "API keys are not supported" in str(e) or "OAuth2" in str(e):
                logger.warning("Write operations not supported with current authentication (API key only)")
                logger.info("User question logging will be disabled. Use Service Account for write operations.")
                return False
            else:
                logger.error(f"Error checking write permissions: {str(e)}")
                return False
    
    def _ensure_user_question_tab(self) -> bool:
        """
        Ensure USER_QUESTION tab exists, create if not
        Only works with proper authentication (not API key only)
        
        Returns:
            True if tab exists or created successfully
        """
        try:
            # Check write permissions first
            if not self._check_write_permissions():
                logger.warning("Cannot ensure USER_QUESTION tab - insufficient permissions")
                return False
            
            # Try to get the sheet metadata to check if USER_QUESTION tab exists
            sheet = self.service.spreadsheets()
            spreadsheet = sheet.get(spreadsheetId=self.sheets_id).execute()
            
            # Check if USER_QUESTION tab exists
            sheets = spreadsheet.get('sheets', [])
            user_question_exists = any(
                s.get('properties', {}).get('title') == 'USER_QUESTION' 
                for s in sheets
            )
            
            if user_question_exists:
                logger.info("USER_QUESTION tab already exists")
                return True
            
            # Create USER_QUESTION tab if it doesn't exist
            logger.info("Creating USER_QUESTION tab...")
            request_body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': 'USER_QUESTION'
                        }
                    }
                }]
            }
            
            response = sheet.batchUpdate(
                spreadsheetId=self.sheets_id,
                body=request_body
            ).execute()
            
            # Add headers
            header_values = [['Pertanyaan', 'Timestamp']]
            header_body = {'values': header_values}
            
            sheet.values().update(
                spreadsheetId=self.sheets_id,
                range='USER_QUESTION!A1:B1',
                valueInputOption='RAW',
                body=header_body
            ).execute()
            
            logger.info("USER_QUESTION tab created successfully with headers")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring USER_QUESTION tab: {str(e)}")
            return False
    
    def log_user_question(self, question: str) -> bool:
        """
        Log user question to USER_QUESTION tab column A
        Only works with proper write permissions (Service Account/OAuth2)
        
        Args:
            question: User's question to log
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.sheets_id:
                logger.warning("Google Sheets ID not configured - cannot log questions")
                return False
            
            # Check if we have write permissions first
            if not self._check_write_permissions():
                logger.info("Skipping user question logging - insufficient permissions (read-only API key)")
                return False
            
            # Ensure USER_QUESTION tab exists
            if not self._ensure_user_question_tab():
                logger.error("Could not ensure USER_QUESTION tab exists")
                return False
            
            # Prepare data to append
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            values = [[question, timestamp]]
            
            # Append to USER_QUESTION tab
            body = {
                'values': values
            }
            
            sheet = self.service.spreadsheets()
            result = sheet.values().append(
                spreadsheetId=self.sheets_id,
                range='USER_QUESTION!A:B',
                valueInputOption='RAW',
                body=body
            ).execute()
            
            logger.info(f"Successfully logged user question: {question[:50]}...")
            return True
            
        except Exception as e:
            if "API keys are not supported" in str(e):
                logger.info("User question logging disabled - API key doesn't support write operations")
            else:
                logger.error(f"Error logging user question: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear cached data to force fresh fetch"""
        self._cached_data = None
        self._cache_timestamp = None
        logger.info("Google Sheets cache cleared")

if __name__ == "__main__":
    # Test the Google Sheets loader
    try:
        loader = GoogleSheetsLoader()
        
        # Test connection
        print("Testing Google Sheets connection...")
        connection_status = loader.test_connection()
        print(f"Connection Status: {json.dumps(connection_status, indent=2)}")
        
        if connection_status['status'] == 'connected':
            # Test data loading
            print("\nFetching knowledge data...")
            knowledge_data = loader.get_knowledge_data()
            print(f"Loaded {len(knowledge_data)} knowledge items")
            
            # Show sample data
            if knowledge_data:
                print("\nSample data:")
                for i, item in enumerate(knowledge_data[:3]):
                    print(f"{i+1}. Category: {item['category']}")
                    print(f"   Question: {item['question'][:100]}...")
                    print(f"   Answer: {item['answer'][:100]}...")
                    print()
        
    except Exception as e:
        print(f"Error: {str(e)}")