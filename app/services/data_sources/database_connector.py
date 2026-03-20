"""
Database Connector Service
Handles data extraction from MySQL, PostgreSQL, MSSQL, MongoDB
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Base class for database connections"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None

    def connect(self) -> bool:
        """Establish database connection"""
        raise NotImplementedError("Subclasses must implement connect()")

    def disconnect(self):
        """Close database connection"""
        raise NotImplementedError("Subclasses must implement disconnect()")

    def execute_query(self, query: str) -> Tuple[bool, Any, str]:
        """Execute the exact query provided and return results"""
        raise NotImplementedError("Subclasses must implement execute_query()")

    def test_connection(self) -> Tuple[bool, str]:
        """Test database connection"""
        raise NotImplementedError("Subclasses must implement test_connection()")

    def _cursor_to_dataframe(self, cursor) -> pd.DataFrame:
        """
        Convert a DBAPI2 cursor (post-execute) to a DataFrame.
        Works for pymysql, psycopg2, and pyodbc cursors.
        Reads column names from cursor.description and rows from fetchall(),
        which is reliable across all supported drivers.
        """
        if cursor.description is None:
            return pd.DataFrame()
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)


class MySQLConnector(DatabaseConnector):
    """MySQL database connector"""

    def connect(self) -> bool:
        try:
            import pymysql
            # FIX: explicitly use the plain Cursor class so rows come back as
            # tuples and column names are read from cursor.description —
            # consistent with _cursor_to_dataframe expectations.
            self.connection = pymysql.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['username'],
                password=self.config['password'],
                database=self.config['database_name'],
                connect_timeout=self.config.get('connection_timeout', 30),
                cursorclass=pymysql.cursors.Cursor,
            )
            logger.info(f"✓ Connected to MySQL: {self.config['host']}:{self.config['port']}")
            return True
        except Exception as e:
            logger.error(f"❌ MySQL connection failed: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> Tuple[bool, Any, str]:
        """
        FIX: Execute the exact query using a cursor instead of pd.read_sql.

        pd.read_sql() on a raw pymysql connection is deprecated and unreliable.
        Using cursor.execute() + _cursor_to_dataframe() is the correct approach.
        The query is passed through unmodified as required.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                df = self._cursor_to_dataframe(cursor)
            return True, df, ""
        except Exception as e:
            return False, None, str(e)

    def test_connection(self) -> Tuple[bool, str]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector"""

    def connect(self) -> bool:
        try:
            import psycopg2
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['username'],
                password=self.config['password'],
                database=self.config['database_name'],
                connect_timeout=self.config.get('connection_timeout', 30),
            )
            logger.info(f"✓ Connected to PostgreSQL: {self.config['host']}:{self.config['port']}")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> Tuple[bool, Any, str]:
        """
        FIX: Execute the exact query using a cursor instead of pd.read_sql.

        pd.read_sql() on a raw psycopg2 connection is deprecated and unreliable.
        Using cursor.execute() + _cursor_to_dataframe() is the correct approach.
        The query is passed through unmodified as required.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                df = self._cursor_to_dataframe(cursor)
            return True, df, ""
        except Exception as e:
            return False, None, str(e)

    def test_connection(self) -> Tuple[bool, str]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)


class MSSQLConnector(DatabaseConnector):
    """Microsoft SQL Server database connector"""

    def connect(self) -> bool:
        try:
            import pyodbc

            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config['host']},{self.config['port']};"
                f"DATABASE={self.config['database_name']};"
                f"UID={self.config['username']};"
                f"PWD={self.config['password']}"
            )

            if self.config.get('ssl_enabled', False):
                conn_str += ";Encrypt=yes;TrustServerCertificate=no"

            self.connection = pyodbc.connect(conn_str)
            logger.info(f"✓ Connected to MSSQL: {self.config['host']}:{self.config['port']}")
            return True
        except Exception as e:
            logger.error(f"❌ MSSQL connection failed: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> Tuple[bool, Any, str]:
        """
        FIX: Execute the exact query using a cursor instead of pd.read_sql.

        pd.read_sql() on a raw pyodbc connection is deprecated and unreliable.
        Note: pyodbc cursors do NOT support the context-manager protocol, so
        we call cursor() / close() explicitly.
        The query is passed through unmodified as required.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            df = self._cursor_to_dataframe(cursor)
            cursor.close()
            return True, df, ""
        except Exception as e:
            return False, None, str(e)

    def test_connection(self) -> Tuple[bool, str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)


class MongoDBConnector(DatabaseConnector):
    """MongoDB database connector"""

    def connect(self) -> bool:
        try:
            from pymongo import MongoClient

            if self.config.get('username') and self.config.get('password'):
                uri = (
                    f"mongodb://{self.config['username']}:{self.config['password']}"
                    f"@{self.config['host']}:{self.config['port']}/{self.config['database_name']}"
                )
            else:
                uri = f"mongodb://{self.config['host']}:{self.config['port']}"

            if self.config.get('ssl_enabled', False):
                uri += "?ssl=true"

            self.connection = MongoClient(
                uri,
                serverSelectionTimeoutMS=self.config.get('connection_timeout', 30) * 1000,
            )

            # Test connection eagerly
            self.connection.server_info()

            self.db = self.connection[self.config['database_name']]
            logger.info(f"✓ Connected to MongoDB: {self.config['host']}:{self.config['port']}")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False

    def disconnect(self):
        if self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> Tuple[bool, Any, str]:
        """
        Execute a MongoDB query.

        Query format — JSON string with keys:
            collection (required): collection name
            filter      (optional): MongoDB filter dict, default {}
            limit       (optional): max documents to return, default 1000

        FIX: BSON types such as ObjectId are not JSON-serialisable and cause
        crashes in _clean_json downstream.  Each document value is probed with
        json.dumps(); non-serialisable values are coerced to str().
        """
        try:
            import json

            query_dict = json.loads(query)
            collection_name = query_dict.get('collection')
            filter_query = query_dict.get('filter', {})
            limit = query_dict.get('limit', 1000)

            if not collection_name:
                return False, None, "Missing 'collection' in query"

            collection = self.db[collection_name]
            mongo_cursor = collection.find(filter_query).limit(limit)

            rows = []
            for doc in mongo_cursor:
                sanitised = {}
                for k, v in doc.items():
                    try:
                        json.dumps(v)       # probe serializability
                        sanitised[k] = v
                    except (TypeError, ValueError):
                        sanitised[k] = str(v)   # coerce ObjectId / Binary / etc.
                rows.append(sanitised)

            df = pd.DataFrame(rows)
            return True, df, ""
        except Exception as e:
            return False, None, str(e)

    def test_connection(self) -> Tuple[bool, str]:
        try:
            self.connection.server_info()
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)


class DatabaseConnectorFactory:
    """Factory for creating database connectors"""

    @staticmethod
    def get_connector(db_type: str, config: Dict[str, Any]) -> DatabaseConnector:
        """
        Get appropriate connector for database type.

        FIX: db_type may arrive as a DatabaseType enum (e.g. DatabaseType.MYSQL)
        or as a plain string.  Calling .lower() directly on an enum raises
        AttributeError.  We normalise to a lowercase string first.
        """
        # Normalise: extract .value if this is a DatabaseType enum
        if hasattr(db_type, 'value'):
            db_type = db_type.value
        db_type = str(db_type).lower()

        connectors = {
            'mysql': MySQLConnector,
            'postgresql': PostgreSQLConnector,
            'mssql': MSSQLConnector,
            'mongodb': MongoDBConnector,
        }
        connector_class = connectors.get(db_type)
        if not connector_class:
            raise ValueError(
                f"Unsupported database type: '{db_type}'. "
                f"Supported types: {', '.join(connectors.keys())}"
            )
        return connector_class(config)

    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported database types"""
        return ['mysql', 'postgresql', 'mssql', 'mongodb']
