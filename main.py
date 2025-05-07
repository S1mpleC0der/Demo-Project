import pandas as pd
import json
import os
import pyodbc
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('banking_etl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('banking_etl')

# Load environment variables
load_dotenv()

# SQL Server connection string
conn_str = (
    f"Driver={os.getenv('SQL_DRIVER')};"
    f"Server={os.getenv('SQL_SERVER')};"
    f"Database={os.getenv('SQL_DATABASE')};"
    f"Trusted_Connection={os.getenv('SQL_TRUSTED_CONNECTION')};"
)

# Load mapping
def load_mapping(file_path="column_table_map.json"):
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load mapping file: {e}")
        raise

# Verify table exists in database
def verify_table_exists(table_name):
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                      WHERE TABLE_NAME = '{table_name}')
                SELECT 1
            ELSE
                SELECT 0
        """)
        
        exists = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        return exists == 1
    except Exception as e:
        logger.error(f"Error verifying table {table_name}: {e}")
        return False

# Get table schema from database
def get_table_schema(table_name):
    """Retrieve the actual column names from the SQL database table"""
    try:
        columns = []
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}'
        """)
        
        for row in cursor.fetchall():
            columns.append(row[0])
        
        cursor.close()
        conn.close()
        
        logger.info(f"Retrieved schema for {table_name}: {', '.join(columns)}")
        return columns
    except Exception as e:
        logger.error(f"Error retrieving schema for {table_name}: {e}")
        return []

# Decode columns using mapping
def decode_columns(df, table_id, table_name):
    try:
        column_map = load_mapping()
        col_map = column_map[table_id]["columns"]
        rename_dict = {}
        
        # Track renamed and ignored columns for logging
        renamed = []
        ignored = []
        
        # Rename using JSON map first
        for code, readable in col_map.items():
            # Handle both numeric codes and string codes correctly
            if code.isdigit():
                encoded_col = f"{table_id}-{code}"
            else:
                encoded_col = code
                
            if encoded_col in df.columns:
                rename_dict[encoded_col] = readable
                renamed.append(f"{encoded_col} → {readable}")
            else:
                ignored.append(encoded_col)

        df.rename(columns=rename_dict, inplace=True)
        
        # Log rename operations
        logger.info(f"Renamed {len(renamed)} columns in {table_name}: {', '.join(renamed)}")
        if ignored:
            logger.warning(f"Ignored {len(ignored)} columns not found in dataset: {', '.join(ignored)}")

        # Apply schema adjustments to match SQL table definitions
        schema_adjustments = {}
        
        if table_name == "users":
            schema_adjustments = {
                "id": "user_id",
                "name": "full_name",
                "phone_number": "phone"
            }

        elif table_name == "cards":
            schema_adjustments = {
                "id": "card_id"
            }

        elif table_name == "transactions":
            schema_adjustments = {
                "id": "transaction_id",
                "from_card_id": "card_id",  # Rename from_card_id to card_id
                "created_at": "transaction_date"  # Rename created_at to transaction_date
            }
            # Drop columns not in SQL schema
            drop_cols = [col for col in ["to_card_id"] if col in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                logger.info(f"Dropped columns not in schema: {', '.join(drop_cols)}")

        elif table_name == "scheduled_payments":
            schema_adjustments = {
                "id": "schedule_id",
                "amount": "payment_amount",
                "payment_date": "schedule_date",  # Map payment_date to schedule_date
                "from_card_id": "card_id",  # If from_card_id exists
                "card_id": "card_id"  # Keep card_id as is
            }
            # Remove columns not in SQL schema
            drop_cols = []
            for col in ["user_id", "created_at"]:
                if col in df.columns:
                    drop_cols.append(col)
            
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                logger.info(f"Dropped columns not in schema: {', '.join(drop_cols)}")

        elif table_name == "reports":
            schema_adjustments = {
                "id": "report_id",
                "created_at": "generated_date"  # Rename created_at to generated_date
            }

        elif table_name == "logs":
            schema_adjustments = {
                "id": "log_id",
                "created_at": "event_timestamp"  # Rename created_at to event_timestamp
            }
        
        # Apply schema adjustments
        schema_renamed = []
        for old_col, new_col in schema_adjustments.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
                schema_renamed.append(f"{old_col} → {new_col}")
        
        if schema_renamed:
            logger.info(f"Schema adjustments: {', '.join(schema_renamed)}")

        # Final validation - ensure all columns exist in the database
        db_columns = get_table_schema(table_name)
        if db_columns:
            invalid_cols = [col for col in df.columns if col not in db_columns]
            if invalid_cols:
                logger.warning(f"Dropping columns not in database schema: {', '.join(invalid_cols)}")
                df.drop(columns=invalid_cols, inplace=True)

        return df
    except Exception as e:
        logger.error(f"Error decoding columns: {e}")
        raise

# Clean and validate data
def clean_data(df, table_name):
    issues = []
    validation_stats = {
        "total": len(df),
        "date_corrections": 0,
        "email_corrections": 0,
        "phone_corrections": 0,
        "flags_added": 0
    }
    
    try:
        # Date handling
        for col in df.columns:
            if any(date_term in col.lower() for date_term in ["date", "created_at", "timestamp"]):
                # Save original values to count changes
                invalid_dates = pd.to_datetime(df[col], errors='coerce').isna() & df[col].notna()
                invalid_count = invalid_dates.sum()
                if invalid_count > 0:
                    issues.append(f"Found {invalid_count} invalid dates in {col}")
                    validation_stats["date_corrections"] += invalid_count
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted {col} to datetime, fixed {invalid_count} invalid values")

        # Email validation and normalization
        if "email" in df.columns:
            # Convert to string first
            df["email"] = df["email"].astype(str).str.strip().str.lower()
            
            # Regular email pattern validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_emails = ~df["email"].str.match(email_pattern) & (df["email"] != "nan")
            invalid_count = invalid_emails.sum()
            
            if invalid_count > 0:
                issues.append(f"Found {invalid_count} invalid email formats")
                validation_stats["email_corrections"] += invalid_count
                logger.warning(f"Email validation: {invalid_count} invalid formats")

        # Phone number cleaning
        if any(phone_col in df.columns for phone_col in ["phone", "phone_number"]):
            phone_col = "phone" if "phone" in df.columns else "phone_number"
            # Save original values to count changes
            original_phones = df[phone_col].copy()
            # Clean phone numbers - remove non-digit chars except + and spaces
            df[phone_col] = df[phone_col].astype(str).str.replace(r"[^\d+\s-()]", "", regex=True)
            # Count changes
            changed_phones = (original_phones != df[phone_col]).sum()
            validation_stats["phone_corrections"] += changed_phones
            if changed_phones > 0:
                logger.info(f"Cleaned {changed_phones} phone numbers")

        # Flag large transactions
        if table_name == "transactions" and "amount" in df.columns:
            large_amount_threshold = 10000
            large_transactions = df["amount"] > large_amount_threshold
            df["flag_large_amount"] = large_transactions
            large_count = large_transactions.sum()
            validation_stats["flags_added"] += large_count
            if large_count > 0:
                logger.info(f"Flagged {large_count} large transactions (>{large_amount_threshold})")

        # Flag over limit cards
        if table_name == "cards" and "limit_amount" in df.columns and "balance" in df.columns:
            over_limit = df["balance"] > df["limit_amount"]
            df["exceeds_limit"] = over_limit
            over_count = over_limit.sum()
            validation_stats["flags_added"] += over_count
            if over_count > 0:
                logger.info(f"Flagged {over_count} cards exceeding their limit")

        # Check for duplicates in primary keys
        primary_key_map = {
            "users": "user_id",
            "cards": "card_id",
            "transactions": "transaction_id",
            "logs": "log_id",
            "reports": "report_id",
            "scheduled_payments": "schedule_id"
        }
        
        if table_name in primary_key_map and primary_key_map[table_name] in df.columns:
            pk_col = primary_key_map[table_name]
            duplicates = df[pk_col].duplicated()
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                issues.append(f"Found {duplicate_count} duplicate values in primary key {pk_col}")
                logger.warning(f"Removed {duplicate_count} duplicate primary keys from {table_name}")
                # Keep first occurrence of each primary key
                df = df[~duplicates]

        # Report validation results
        logger.info(f"Data validation for {table_name}: {validation_stats}")
        if issues:
            logger.warning(f"Validation issues: {'; '.join(issues)}")

        return df, issues
    except Exception as e:
        logger.error(f"Error cleaning data for {table_name}: {e}")
        issues.append(f"Cleaning error: {str(e)}")
        return df, issues

# Check for existing keys in the database
def filter_existing_keys(df, table_name):
    primary_key_map = {
        "users": "user_id",
        "cards": "card_id",
        "transactions": "transaction_id",
        "logs": "log_id",
        "reports": "report_id",
        "scheduled_payments": "schedule_id"
    }
    
    if table_name not in primary_key_map or primary_key_map[table_name] not in df.columns:
        return df, 0
        
    pk_col = primary_key_map[table_name]
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Get all IDs that already exist in the database
        cursor.execute(f"SELECT {pk_col} FROM {table_name}")
        existing_ids = set([row[0] for row in cursor.fetchall()])
        
        # Filter out rows with IDs that already exist
        if existing_ids:
            initial_count = len(df)
            df = df[~df[pk_col].isin(existing_ids)]
            filtered_count = initial_count - len(df)
            
            if filtered_count > 0:
                logger.warning(f"Filtered out {filtered_count} rows with existing primary keys in {table_name}")
                
        cursor.close()
        conn.close()
        
        return df, filtered_count
    except Exception as e:
        logger.error(f"Error checking existing primary keys: {e}")
        return df, 0

# Upload to SQL Server
def upload_to_sql(df, table_name):
    if not verify_table_exists(table_name):
        logger.error(f"Table {table_name} does not exist in the database")
        return 0, len(df)
    
    # Get actual column names from the database
    db_columns = get_table_schema(table_name)
    
    # Only keep columns that exist in the database
    valid_columns = [col for col in df.columns if col in db_columns]
    if len(valid_columns) < len(df.columns):
        dropped = set(df.columns) - set(valid_columns)
        logger.warning(f"Dropping columns not in database: {', '.join(dropped)}")
        df = df[valid_columns]
    
    # If no valid columns left, return error
    if not valid_columns:
        logger.error(f"No valid columns for table {table_name}")
        return 0, len(df)
    
    # Check for existing records based on primary key
    df, filtered_count = filter_existing_keys(df, table_name)
    
    # If no rows left after filtering, return early
    if len(df) == 0:
        logger.warning(f"No rows to insert after filtering for {table_name}")
        return 0, filtered_count
    
    processed, errors = 0, 0
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        conn.autocommit = False  # Start transaction
        
        # Try fast batch insert first
        try:
            start_time = time.time()
            # Create placeholder string like (?,?,?)
            placeholders = ','.join(['?'] * len(df.columns))
            cols = ','.join([f"[{col}]" for col in df.columns])
            
            # Prepare data as list of tuples
            data = [tuple(None if pd.isna(x) else x for x in row) for row in df.values]
            
            # Execute batch insert
            cursor.fast_executemany = True
            cursor.executemany(
                f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})", 
                data
            )
            conn.commit()
            
            end_time = time.time()
            processed = len(df)
            logger.info(f"Batch inserted {processed} rows in {end_time - start_time:.2f} seconds")
            return processed, 0
            
        except Exception as e:
            logger.warning(f"Batch insert failed, falling back to row-by-row: {e}")
            conn.rollback()  # Rollback the failed batch
            
            # Fall back to row-by-row insert
            start_time = time.time()
            conn.autocommit = False
            
            # Get insert mode from environment
            insert_mode = os.getenv('INSERT_MODE', 'SKIP_EXISTING')
            
            for _, row in df.iterrows():
                try:
                    # Proceed with insert
                    cols = ','.join([f"[{col}]" for col in row.index])
                    placeholders = ','.join(['?'] * len(row))
                    values = [None if pd.isna(x) else x for x in row]
                    
                    cursor.execute(f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})", values)
                    processed += 1
                except Exception as row_error:
                    logger.error(f"Error inserting row: {row_error}")
                    errors += 1
            
            conn.commit()
            end_time = time.time()
            logger.info(f"Row-by-row inserted {processed} rows in {end_time - start_time:.2f} seconds")
    
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error in SQL upload: {e}")
        errors = len(df) - processed
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return processed, errors

# Insert into retrieveinfo
def insert_metadata(file, total, processed, errors, issues=None):
    try:
        # Prepare notes
        notes = f"Ingested from {file}"
        if issues:
            notes += f"; Issues: {'; '.join(issues)}"
            
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Check if retrieveinfo table exists
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'retrieveinfo')
            CREATE TABLE retrieveinfo (
                retrieve_id INT IDENTITY(1,1) PRIMARY KEY,
                source_file VARCHAR(255),
                retrieved_at DATETIME,
                total_rows INT,
                processed_rows INT,
                errors INT,
                notes TEXT
            )
        """)
        conn.commit()
        
        # Insert metadata
        cursor.execute("""
            INSERT INTO retrieveinfo (source_file, retrieved_at, total_rows, processed_rows, errors, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, file, datetime.now(), total, processed, errors, notes)
        conn.commit()
        
        logger.info(f"Recorded metadata for {file}: {processed}/{total} rows, {errors} errors")
        
    except Exception as e:
        logger.error(f"Error inserting metadata: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Main ETL process
def main():
    overall_start = time.time()
    column_map = load_mapping()
    
    # Option for handling existing records
    INSERT_MODE = os.getenv('INSERT_MODE', 'SKIP_EXISTING')  # SKIP_EXISTING or REPLACE_EXISTING or FAIL_ON_EXISTING
    logger.info(f"Running ETL with insert mode: {INSERT_MODE}")
    
    # Process each table
    for table_id, info in column_map.items():
        file = info["file"]
        table_name = info["table"]
        
        if not os.path.exists(file):
            logger.warning(f"Missing file: {file}")
            continue
        
        try:
            logger.info(f"Processing {file} → {table_name}")
            start_time = time.time()
            
            # Read CSV
            df = pd.read_csv(file)
            logger.info(f"Loaded {len(df)} rows from {file}")
            
            # Decode columns
            df = decode_columns(df, table_id, table_name)
            
            # Clean data
            df, issues = clean_data(df, table_name)
            
            # Upload to SQL
            total = len(df)
            processed, errors = upload_to_sql(df, table_name)
            
            # Record metadata
            insert_metadata(file, total, processed, errors, issues)
            
            end_time = time.time()
            logger.info(f"Completed {file} to {table_name}: {processed}/{total} rows, "
                        f"{errors} errors in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")
    
    overall_end = time.time()
    logger.info(f"ETL process completed in {overall_end - overall_start:.2f} seconds")

if __name__ == "__main__":
    main()