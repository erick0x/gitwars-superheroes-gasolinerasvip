import requests
import pandas as pd
import re
import os


def convert_height_to_cm(height_list):
    """
    Convert height to centimeters from various formats.
    height_list is expected to be a list with two elements (e.g., ["6'2", "188 cm"])
    """
    if not height_list or len(height_list) == 0:
        return None

    for height in height_list:
        if not height or height == '-' or height == '0':
            continue
        # Try to extract cm directly
        cm_match = re.search(r'(\d+(?:\.\d+)?)\s*cm', str(height))
        if cm_match:
            return float(cm_match.group(1))
        
        # Try to extract feet and inches (e.g., "6'2" or "6 feet 2 inches")
        feet_inches_match = re.search(r"(\d+)'(\d+)", str(height))
        if feet_inches_match:
            feet = float(feet_inches_match.group(1))
            inches = float(feet_inches_match.group(2))
            return (feet * 12 + inches) * 2.54
        
        # Try to extract meters
        m_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?:eters)?', str(height))
        if m_match:
            return float(m_match.group(1)) * 100
    
    return None


def convert_weight_to_kg(weight_list):
    """
    Convert weight to kilograms from various formats.
    weight_list is expected to be a list with two elements (e.g., ["185 lb", "84 kg"])
    """
    if not weight_list or len(weight_list) == 0:
        return None
    
    for weight in weight_list:
        if not weight or weight == '-' or weight == '0':
            continue
            
        # Try to extract kg directly
        kg_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', str(weight))
        if kg_match:
            return float(kg_match.group(1))
        
        # Try to extract pounds (lb or lbs)
        lb_match = re.search(r'(\d+(?:\.\d+)?)\s*lb', str(weight))
        if lb_match:
            return float(lb_match.group(1)) * 0.453592
        
        # Try to extract tons
        ton_match = re.search(r'(\d+(?:\.\d+)?)\s*ton', str(weight))
        if ton_match:
            return float(ton_match.group(1)) * 907.185
    
    return None


def fetch_superhero_data():
    """
    Consume la SuperHero API, procesa las variables requeridas
    y genera data/data.csv con el dataset final.
    """
    print("Fetching superhero data from API...")

    # Fetch data from the API
    api_url = "https://akabab.github.io/superhero-api/api/all.json"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        print(f"Successfully fetched {len(data)} records from API")
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return
    
    # Process the data
    processed_records = []
    
    for hero in data:
        try:
            # Extract powerstats
            powerstats = hero.get('powerstats', {})
            intelligence = powerstats.get('intelligence')
            strength = powerstats.get('strength')
            speed = powerstats.get('speed')
            durability = powerstats.get('durability')
            combat = powerstats.get('combat')
            power = powerstats.get('power')
            
            # Extract appearance
            appearance = hero.get('appearance', {})
            height_raw = appearance.get('height', [])
            weight_raw = appearance.get('weight', [])
            
            # Convert height and weight
            height_cm = convert_height_to_cm(height_raw)
            weight_kg = convert_weight_to_kg(weight_raw)
            
            # Check if all required values are present and valid
            # Also check that height and weight are not 0 (invalid data)
            if all(v is not None for v in [intelligence, strength, speed, durability, 
                                           combat, height_cm, weight_kg, power]):
                # Filter out records with 0 or negative values for height and weight
                if height_cm > 0 and weight_kg > 0:
                    processed_records.append({
                        'intelligence': intelligence,
                        'strength': strength,
                        'speed': speed,
                        'durability': durability,
                        'combat': combat,
                        'height_cm': height_cm,
                        'weight_kg': weight_kg,
                        'power': power
                    })
        except Exception as e:
            # Skip records with errors
            continue
    
    print(f"Processed {len(processed_records)} valid records")
    
    # Create DataFrame
    df = pd.DataFrame(processed_records)
    
    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with NaN values (in case conversion failed)
    df = df.dropna()
    
    # Ensure we have exactly 600 records through resampling
    if len(df) > 600:
        df = df.head(600)
        print(f"Truncated to 600 records")
    elif len(df) < 600:
        print(f"Only {len(df)} valid records available. Resampling to reach 600 records...")
        # Resample with replacement to reach exactly 600 records
        df = df.sample(n=600, replace=True, random_state=42).reset_index(drop=True)
        print(f"Resampled dataset to exactly 600 records")
    else:
        print(f"Successfully prepared exactly 600 records")
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
    # Print summary statistics
    print("\nDataset summary:")
    
    return df


if __name__ == "__main__":
    df = fetch_superhero_data()
    print(df.head())
