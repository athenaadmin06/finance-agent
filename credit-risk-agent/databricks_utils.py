from databricks import sql
import os
from dotenv import load_dotenv

load_dotenv()

def get_customer_history_from_databricks(customer_id):
    """
    Fetches structured payment history from Databricks SQL Warehouse,
    including column names for better context.
    """
    try:
        connection = sql.connect(
            server_hostname=os.getenv("DATABRICKS_HOST").replace("https://", ""),
            http_path=os.getenv("DATABRICKS_HTTP_PATH"),
            access_token=os.getenv("DATABRICKS_TOKEN")
        )

        cursor = connection.cursor()

        # Define table path (Modify these as needed)
        catalog = "vistora_db" 
        schema = "vistora_schema"
        table = "CUSTOMER_PAYMENT_HISTORY"
        full_table_path = f"{catalog}.{schema}.{table}"

        # Execute Query
        query = f"SELECT * FROM {full_table_path} WHERE customer_id = :cid"
        cursor.execute(query, {"cid": customer_id})
        
        # 1. Fetch Data
        row = cursor.fetchone()
        
        if row:
            # 2. Fetch Column Names from cursor.description
            # description is a list of tuples, the first element is the column name
            col_names = [desc[0] for desc in cursor.description]
            
            # 3. Combine Names with Values
            # zip() pairs them up: ('credit_score', 720), ('loan_status', 'active')
            history_dict = dict(zip(col_names, row))
            
            # 4. Format nicely for the LLM
            # Excluding 'customer_id' since we already know it
            formatted_history = ", ".join(
                [f"{k}: {v}" for k, v in history_dict.items() if k != 'customer_id']
            )
            
            return f"Customer History: [{formatted_history}]"
        else:
            return "No prior customer history found in Databricks."

    except Exception as e:
        print(f"❌ Databricks SQL Error: {e}")
        return "Error fetching history."
    
    finally:
        if 'cursor' in locals() and cursor: cursor.close()
        if 'connection' in locals() and connection: connection.close()
