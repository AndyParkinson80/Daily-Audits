import io
import json
import logging
import time
import warnings
import os
import sys

from collections import defaultdict
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import default
from google.cloud import bigquery

import pandas as pd

from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def google_auth():
    """
    Authenticate with Google Cloud and return credentials and project ID.
    
    First tries to use Application Default Credentials (ADC).
    If that fails, uses service account credentials from GOOGLE_CLOUD_SECRET environment variable.
    
    Returns:
        tuple: (credentials, project_id)
        
    Raises:
        Exception: If both authentication methods fail
    """
    try:
        # Try Application Default Credentials first
        credentials, project_id = default()
        print("Successfully authenticated using Application Default Credentials")
        return credentials, project_id
         
    except DefaultCredentialsError:
        print("Application Default Credentials not available, trying service account...")
        
        # Try service account from environment variable
        secret_json = os.getenv('GOOGLE_CLOUD_SECRET')
        if not secret_json:
            raise Exception("GOOGLE_CLOUD_SECRET environment variable not found")
        
        try:
            # Parse the JSON credentials
            service_account_info = json.loads(secret_json)
            
            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info
            )
            
            # Extract project ID from service account info
            project_id = service_account_info.get('project_id')
            if not project_id:
                raise Exception("project_id not found in service account credentials")
            
            print("Successfully authenticated using service account credentials")
            return credentials, project_id
            
        except json.JSONDecodeError:
            raise Exception("Invalid JSON in GOOGLE_CLOUD_SECRET environment variable")
        except Exception as e:
            raise Exception(f"Failed to create service account credentials: {str(e)}")
    
    except Exception as e:
        raise Exception(f"Authentication failed: {str(e)}")

#--------------------------------Download the files from google drive-------------------------------

def get_folder_id_by_name(service, folder_name):
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    response = service.files().list(q=query, fields="files(id, name)").execute()
    folders = response.get('files', [])
    if not folders:
        raise Exception(f"Folder named '{folder_name}' not found.")
    return folders[0]['id']

def find_file_id_by_name_keyword(service, folder_id, keyword):
    # Search for a file with keyword in name inside the folder
    query = f"'{folder_id}' in parents and name contains '{keyword}' and trashed = false"
    response = service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get('files', [])
    if not files:
        raise Exception(f"No files containing '{keyword}' found in folder ID {folder_id}")
    return files[0]['id'], files[0]['name']

def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")
    file_buffer.seek(0)
    return file_buffer

def load_excel_from_drive(folder_name, keyword , creds):

    drive_service = build('drive', 'v3', credentials=creds)

    folder_id = get_folder_id_by_name(drive_service, folder_name)
    print(f"Found folder '{folder_name}' with ID: {folder_id}")

    file_id, file_name = find_file_id_by_name_keyword(drive_service, folder_id, keyword)
    print(f"Found file containing '{keyword}': {file_name} (ID: {file_id})")

    file_buffer = download_file(drive_service, file_id)
    df = pd.read_excel(file_buffer)
    return df

# --------------------------------Clean the data and combine into folder-----------------------------

def deduplicate_table(project: str, dataset: str, table: str):
    """
    Deduplicates a BigQuery table by all columns using TO_JSON_STRING-based row hash.
    Replaces the original table with the deduplicated result.
    Also prints row counts before and after deduplication.

    Args:
        project (str): GCP project ID.
        dataset (str): BigQuery dataset name.
        table (str): BigQuery table name.
    """
    client = bigquery.Client(credentials=creds,project=project)
    table_ref = f"{project}.{dataset}.{table}"

    # Step 1: Count rows before deduplication
    count_before = client.query(f"SELECT COUNT(*) AS count FROM `{table_ref}`").result().to_dataframe().iloc[0]["count"]
    print(f"Rows before deduplication: {count_before}")

    # Step 2: Remove 'row_num' column if it exists
    table_obj = client.get_table(table_ref)
    cleaned_schema = [field for field in table_obj.schema if field.name != 'row_num']

    if len(cleaned_schema) != len(table_obj.schema):
        print(f"Removing 'row_num' column from: {table_ref}")
        
        columns = ', '.join([f"`{field.name}`" for field in cleaned_schema])
        remove_rownum_sql = f"""
        CREATE OR REPLACE TABLE `{table_ref}` AS
        SELECT {columns}
        FROM `{table_ref}`;
        """
        client.query(remove_rownum_sql).result()

    # Step 3: Deduplicate
    dedup_sql = f"""
    CREATE OR REPLACE TABLE `{table_ref}` AS
    WITH ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY TO_JSON_STRING(t)) AS row_num
      FROM `{table_ref}` AS t
    )
    SELECT * EXCEPT(row_num)
    FROM ranked
    WHERE row_num = 1;
    """
    client.query(dedup_sql).result()

    # Step 4: Count rows after deduplication
    count_after = client.query(f"SELECT COUNT(*) AS count FROM `{table_ref}`").result().to_dataframe().iloc[0]["count"]
    print(f"Rows after deduplication: {count_after}")
    print(f"Rows removed: {count_before - count_after}")

def audits():
    def dates_times(given_date=None):
        if given_date is None:
            return
        
        day = int(given_date.day)
        month = int(given_date.month)
        year = int(given_date.year)

        def get_week_number_after_week_1(given_date, week_1_start):
            """
            Calculate the week number after Week 1.
            
            :param given_date: The date for which the week number is calculated.
            :param week_1_start: The Monday start date of Week 1.
            :return: The week number after Week 1 (1-based).
            """
            if given_date < week_1_start:
                raise ValueError("The given date is before Week 1 starts.")
            
            days_difference = (given_date - week_1_start).days
            week_number = (days_difference // 7)+1
            return week_number
        
        if month > 4 or (month == 4 and day > 5):
            year = year
        else:
            year = year - 1

        specific_date = datetime(year, 4, 5)

        days_until_friday = (4 - specific_date.weekday() + 7) % 7
        if days_until_friday == 0:
            days_until_friday = 7
        friday_date = specific_date + timedelta(days=days_until_friday)

        week_1_office = friday_date - timedelta(days=friday_date.weekday() + 7)
        week_1_field = week_1_office - timedelta(days=2)

        week_number = get_week_number_after_week_1(given_date, week_1_office)
        year_range = f"{year}-{year+1}"

        return year_range, week_number
    
    df = audit_daily_df
    df = df.dropna(subset=['Item Date'])
    df['Item Date'] = df['Item Date'].str.replace(r'^[^\d]+', '', regex=True)
    df['Item Date'] = pd.to_datetime(df['Item Date'], format='%d/%m/%y')
    df = df[df['Item Date'] >= pd.Timestamp('2024-04-05')]
    df[['Financial Year', 'Week Number']] = df['Item Date'].apply(lambda x: pd.Series(dates_times(x)))
    df[['SubjectName', 'CascadeId']] = df['Subject'].str.extract(r'^(.*) \[(\d+)\]$')
    df.columns = df.columns.str.replace(' ', '')
    df = df.drop(columns=['Subject'])                       
    df = df.drop_duplicates()

    df['ActionTime'] = pd.to_datetime(df['ActionTime'], format='%d %b %Y, %H:%M:%S', errors='coerce').dt.date
    df = df.where(df.notna(), None)

    unique_days = (
        pd.Series(df['ActionTime'].unique())  # convert to date only
        .dropna()                                     # remove NaT
        .tolist()
    )

    query = f"""
        SELECT *
        FROM `{projectId}.{dataset}.Audits`
        WHERE DATE(ActionTime) IN UNNEST(@days)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("days", "DATE", unique_days)
        ]
    )

    df_bq = client.query(query, job_config=job_config).to_dataframe()

    df['ItemDate'] = pd.to_datetime(df['ItemDate']).dt.date
    df_bq['ItemDate'] = pd.to_datetime(df_bq['ItemDate']).dt.date
    df['CascadeId'] = pd.to_numeric(df['CascadeId'], errors='coerce', downcast='integer')
    df_bq['CascadeId'] = pd.to_numeric(df_bq['CascadeId'], errors='coerce', downcast='integer')

    filtered_df = df.merge(df_bq, how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')

    table_id = f"{projectId}.{dataset}.Audits"
    job = client.load_table_from_dataframe(filtered_df, table_id)

    job.result()
    print(f"\n{job.output_rows} rows were uploaded to {table_id}.")

    time.sleep(1)

    deduplicate_table(
    project="api-integrations-412107",
    dataset="Imperago_downloads",
    table="Audits")

def clockings():
    df = clock_daily_df
    df['Clockings'] = df['Clockings'].str.replace('Â', '', regex=False)
    df['Clockings'] = df['Clockings'].apply(lambda x: ' '.join([time.replace(':', '') for time in str(x).split()]) if pd.notnull(x) else x)

    df['Date'] = df['Date'].str.replace(r'^[^\d]+', '', regex=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    df = df[df['Date'] >= pd.Timestamp('2024-04-05')]

    df.columns = df.columns.str.replace(' ', '_')
    unique_days = df['Date'].dt.date.unique().tolist()
    unique_days_str = ", ".join([f"DATE '{day}'" for day in unique_days])

    query = f"""
        SELECT *
        FROM `{projectId}.{dataset}.Clockings`
        WHERE DATE(Date) IN UNNEST([{unique_days_str}])
    """

    df_bq = client.query(query).to_dataframe()

    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df_bq['Date'] = pd.to_datetime(df_bq['Date']).dt.date
    df['Staff_No'] = pd.to_numeric(df['Staff_No'], errors='coerce', downcast='integer')
    df_bq['Staff_No'] = pd.to_numeric(df_bq['Staff_No'], errors='coerce', downcast='integer')

    filtered_df = df.merge(df_bq, how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')

    table_id = f"{projectId}.{dataset}.Clockings"
    job = client.load_table_from_dataframe(filtered_df, table_id)

    job.result()
    print(f"\n{job.output_rows} rows were uploaded to {table_id}.")

    time.sleep(1)

    deduplicate_table(
    project="api-integrations-412107",
    dataset="Imperago_downloads",
    table="Clockings")

def clean_clockings():
    query = f"""
    SELECT *
    FROM `{projectId}.{dataset}.Clockings`
    """

    print("\nRemoving Duplicates from Clockings...")
    # Run the query
    query_job = client.query(query)
    results = query_job.result()

    # Convert to pandas DataFrame for easier manipulation
    df = results.to_dataframe()
    print(f"Retrieved {len(df)} rows from BigQuery.")

    # Define a dictionary to hold the combined data
    combined_data = defaultdict(lambda: {
        'Name': '',
        'Staff_No': 0,
        'Organisation_Group': '',
        'Payroll_Group': '',
        'Date': None,
        'Clockings': [],
        'Comments': []
    })

    print("Combining rows with the same Staff_No and Date...")
    # Iterate through the results to combine rows
    for row in df.itertuples():
        # Create a key using Staff_No and Date
        key = (row.Staff_No, row.Date)
        
        # Store the non-list information (only need to do this once per key)
        if not combined_data[key]['Name']:
            combined_data[key]['Name'] = row.Name
            combined_data[key]['Staff_No'] = row.Staff_No
            combined_data[key]['Organisation_Group'] = row.Organisation_Group
            combined_data[key]['Payroll_Group'] = row.Payroll_Group
            combined_data[key]['Date'] = row.Date
            
        # Append clockings only if it's not already in the list
        if row.Clockings and pd.notna(row.Clockings):
            clocking_str = str(row.Clockings)
            if clocking_str not in combined_data[key]['Clockings']:
                combined_data[key]['Clockings'].append(clocking_str)
        
        # Always append comments
        if row.Comments and pd.notna(row.Comments):
            combined_data[key]['Comments'].append(str(row.Comments))

    # Create a list of dictionaries for the cleaned data
    cleaned_data = []
    for key, data in combined_data.items():
        cleaned_data.append({
            'Name': data['Name'],
            'Staff_No': data['Staff_No'],
            'Organisation_Group': data['Organisation_Group'],
            'Payroll_Group': data['Payroll_Group'],
            'Date': data['Date'],
            'Clockings': '\n'.join(data['Clockings']),
            'Comments': '\n'.join(data['Comments'])
        })

    print(f"Created {len(cleaned_data)} combined records.")

    cleaned_df = pd.DataFrame(cleaned_data)

    # Convert date objects to ISO format for JSON serialization
    json_data = cleaned_df.copy()
    json_data['Date'] = json_data['Date'].apply(lambda x: x.isoformat() if x else None)

    print(f"Total records: {len(json_data)}")

    # Create a schema for the new table
    schema = [
        bigquery.SchemaField("Name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Staff_No", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("Organisation_Group", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Payroll_Group", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Date", "DATE", mode="NULLABLE"),
        bigquery.SchemaField("Clockings", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("Comments", "STRING", mode="NULLABLE")
    ]

    # Create a table reference for the new table
    table_id = f"{projectId}.{dataset}.clockings_cleaned"
    table = bigquery.Table(table_id, schema=schema)

    # Create the table
    try:
        table = client.create_table(table, exists_ok=True)
        print(f"Created table {table_id}")
    except Exception as e:
        print(f"Error creating table: {e}")

    print("Uploading cleaned data to BigQuery")

    # Using load_table_from_dataframe for the upload
    try:
        # Ensure dates are in datetime format for BigQuery loading
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
        
        # Create a job config
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE",
        )
        
        # Load the dataframe directly to BigQuery
        job = client.load_table_from_dataframe(
            cleaned_df, table_id, job_config=job_config
        )
        
        # Wait for the job to complete
        job.result()
        
        print(f"Successfully loaded {len(cleaned_df)} rows using load_table_from_dataframe")
    except Exception as e:
        print(f"Error using load_table_from_dataframe: {e}")
        raise e

    print("Process completed successfully!")

def clean_audits():
    query = f"""
        SELECT *
        FROM `{projectId}.{dataset}.Audits`
        """

    print("\nSimplifying Audits")
    
    # Run the query
    query_job = client.query(query)
    results = query_job.result()

    # Convert to pandas DataFrame
    df = results.to_dataframe()
    print(f"Retrieved {len(df)} rows from BigQuery.")


    print("\nStep 1: Removing general duplicates and keeping only the most recent record...")
    
    # Ensure ActionTime is datetime
    if 'ActionTime' in df.columns:
        try:
            df['ActionTime'] = pd.to_datetime(df['ActionTime'])
            df_sorted = df.sort_values('ActionTime', ascending=False)
        except:
            print("Warning: Could not convert ActionTime to datetime. Using original order.")
            df_sorted = df
    else:
        print("Warning: ActionTime column not found. Using original order.")
        df_sorted = df
    
    # Drop duplicates
    duplicate_columns = [col for col in df.columns if col != 'ActionTime']
    df_deduped = df_sorted.drop_duplicates(subset=duplicate_columns)
    print(f"After removing general duplicates: {len(df_deduped)} rows remain (from original {len(df)} rows).")

    # Write deduped data to BigQuery
    destination_table = f"{projectId}.{dataset}.Audits_cleaned"
    print(f"Writing cleaned data to {destination_table}...")
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    
    job = client.load_table_from_dataframe(
        df_deduped, destination_table, job_config=job_config
    )
    job.result()

    print("Data successfully written to Audits_cleaned table.")
    
    print("\nStep 2: Consolidating related records...")
    
    # Ensure we have an ID column
    if 'ID' not in df_deduped.columns:
        df_deduped = df_deduped.reset_index().rename(columns={'index': 'ID'})
    
    # Replace nulls in non-datetime columns
    for col in df_deduped.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_deduped[col]):
            try:
                df_deduped[col] = df_deduped[col].fillna('null')
            except Exception as e:
                print(f"Skipping column {col} due to error: {e}")

    # Define grouping columns
    group_columns = []
    possible_group_columns = ['ItemDate', 'Actor', 'SubjectName', 'CascadeId']
    for col in possible_group_columns:
        if col in df_deduped.columns:
            group_columns.append(col)

    if not group_columns:
        print("Warning: No standard grouping columns found. Using all columns except ID and Action.")
        group_columns = [col for col in df_deduped.columns if col not in ['ID', 'Action']]
    
    print(f"Grouping by columns: {group_columns}")
    
    # Group and consolidate
    if group_columns:
        consolidated_records = []

        for _, group in df_deduped.groupby(group_columns):
            base_row = group.iloc[0].copy()

            # Consolidate Action
            if 'Action' in group.columns:
                actions = '\n'.join(str(action) for action in group['Action'].unique() if action != 'null')
                base_row['Action'] = actions if actions else 'null'

            # Consolidate From and To
            for field in ['From', 'To']:
                if field in group.columns:
                    values = '\n'.join(str(val) for val in group[field].unique() if val != 'null')
                    base_row[field] = values if values else 'null'

            # Use lowest ID
            if 'ID' in group.columns:
                base_row['ID'] = group['ID'].min()

            consolidated_records.append(base_row)

        consolidated_df = pd.DataFrame(consolidated_records)

        if 'ID' in consolidated_df.columns:
            consolidated_df = consolidated_df.sort_values('ID')
    else:
        print("Warning: No grouping possible. Using original deduplicated data.")
        consolidated_df = df_deduped

    # Drop ID before export
    if 'ID' in consolidated_df.columns:
        consolidated_df = consolidated_df.drop(columns=['ID'])

    print(f"After consolidation: {len(consolidated_df)} rows (from deduped {len(df_deduped)} rows)")

    # Export to BigQuery
    consolidated_table = f"{projectId}.{dataset}.Audits_cleaned"
    print(f"Writing consolidated data to {consolidated_table}...")
    
    job = client.load_table_from_dataframe(
        consolidated_df, consolidated_table, job_config=job_config
    )
    job.result()

    print("Data successfully written to Audits_cleaned table.")
    
    return consolidated_df

# --------------------------------Join cleaned Data for looker --------------------------------------

def create_joined_table():
    client = bigquery.Client(credentials=creds,project=projectId)

    query = f"""
    CREATE OR REPLACE TABLE `{projectId}.{dataset}.Joined` AS
    SELECT DISTINCT 
      c.*,
      a.ActionTime, 
      a.Actor, 
      a.Action,
      a.From,
      a.To,
      a.FinancialYear,
      a.WeekNumber,
      lm.LM_Path AS LM_Path
    FROM `{projectId}.{dataset}.Audits_cleaned` AS a
    JOIN `{projectId}.{dataset}.clockings_cleaned` AS c
      ON a.ItemDate = c.Date 
      AND a.CascadeId = c.Staff_No
    LEFT JOIN `universal_lookup_tables.id_to_lm_path` AS lm
      ON CAST(c.Staff_No AS STRING) = lm.EmployeeId
    """

    query_job = client.query(query)
    query_job.result()  # Wait for job to complete

    print(f"\nTable `{projectId}.{dataset}.Joined` created successfully.")

# ------------------------------------------------------------ --------------------------------------

if __name__ == "__main__":
    try:
        creds, projectId = google_auth()
        print(f"Authenticated successfully! Project ID: {projectId}")
    except Exception as e:
        print(f"Authentication error: {e}")


    folder = "Imperago Spreadsheets"

    audit_daily_df = load_excel_from_drive(folder, "__audit__", creds)
    print("\nAudit DataFrame loaded:")

    print (audit_daily_df)

    clock_daily_df = load_excel_from_drive(folder, "__clock__", creds)
    print("\nClock DataFrame loaded:")

    print (clock_daily_df)

    client = bigquery.Client(credentials=creds,project=projectId)

    dataset = "Imperago_downloads"

    audits()
    clockings()

    time.sleep(1)

    clean_clockings()
    clean_audits()

    time.sleep(1)

    create_joined_table()

    print ("\nC'est fini!!!")