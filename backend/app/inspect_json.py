# # inspect_json.py
# import json
# import os

# def inspect_json_file(file_path):
#     print(f" Inspecting JSON file: {file_path}")
    
#     if not os.path.exists(file_path):
#         print(f" File does not exist: {file_path}")
#         return False
    
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         print(f" JSON loaded successfully")
#         print(f"Data type: {type(data)}")
        
#         if isinstance(data, list):
#             print(f" Number of items: {len(data)}")
#             if len(data) > 0:
#                 first_item = data[0]
#                 print(f" First item type: {type(first_item)}")
#                 if isinstance(first_item, dict):
#                     print(f"first item keys: {list(first_item.keys())}")
#                     for key in first_item.keys():
#                         value = first_item[key]
#                         print(f"   {key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
#         else:
#             print(f" Data structure: {type(data)}")
#             if isinstance(data, dict):
#                 print(f"Top-level keys: {list(data.keys())}")
#                 for key in data.keys():
#                     value = data[key]
#                     print(f"   {key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            
#         return True
        
#     except Exception as e:
#         print(f"Error loading JSON: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# if __name__ == "__main__":
#     inspect_json_file("data/Preprocessed_Medical_Book.json")