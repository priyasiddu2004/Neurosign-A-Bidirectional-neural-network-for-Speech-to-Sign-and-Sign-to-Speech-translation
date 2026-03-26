import os
import json
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def convert_json_to_h5():
    """Convert JSON model files to H5 format to avoid compatibility issues."""
    
    # Set up the directory
    directory = 'model'
    
    # List of model files to convert
    model_files = [
        'model-bw.json',
        'model-bw_dru.json', 
        'model-bw_tkdi.json',
        'model-bw_smn.json'
    ]
    
    for model_file in model_files:
        json_path = os.path.join(directory, model_file)
        h5_path = os.path.join(directory, model_file.replace('.json', '.h5'))
        
        if os.path.exists(json_path):
            print(f"Converting {model_file} to H5 format...")
            
            try:
                # Read JSON file
                with open(json_path, 'r') as f:
                    model_json = f.read()
                
                # Load model from JSON
                model = keras.models.model_from_json(model_json)
                
                # Save as H5
                model.save(h5_path)
                print(f"✓ Successfully converted {model_file} to {model_file.replace('.json', '.h5')}")
                
            except Exception as e:
                print(f"✗ Error converting {model_file}: {e}")
        else:
            print(f"✗ File not found: {json_path}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    convert_json_to_h5()
