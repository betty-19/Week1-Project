import pandas as pd

def load_data(file_path):
 
    required_columns = ['headline', 'url', 'publisher', 'date', 'stock']
    
    try:
  
        data = pd.read_csv(file_path)
        
       
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Dataset is missing required columns: {', '.join(required_columns)}")
        
        
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
