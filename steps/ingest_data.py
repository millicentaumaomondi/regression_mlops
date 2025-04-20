import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        """Load data from the specified path."""
        try:
            data = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully from {self.data_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
        return None
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """Ingest data from a CSV file.
    
    Args:
        data_path (str): Path to the CSV file.
        
        Returns:    
        pd.DataFrame: Loaded data as a DataFrame."""
    try:

        ingest_data_instance = IngestData(data_path)
        data = ingest_data_instance.get_data()
        return data
    except Exception as e:
        logging.error(f"Error in ingest_data step: {e}")
        raise
    return None


   