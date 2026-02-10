import requests
import os

url = 'http://127.0.0.1:5001/predict'


image_paths = [
    "./dataset2-master/dataset2-master/images/TEST/EOSINOPHIL/_0_852.jpeg",
    "./dataset2-master/dataset2-master/images/TEST/NEUTROPHIL/_0_208.jpeg",
    "./dataset2-master/dataset2-master/images/TEST/LYMPHOCYTE/_0_1072.jpeg",
    "./dataset2-master/dataset2-master/images/TEST/MONOCYTE/_0_1673.jpeg"
]

print(f"--- Starting Bulk Test on {len(image_paths)} images ---\n")

for i, img_path in enumerate(image_paths):
    print(f"Test #{i+1}: Sending {os.path.basename(img_path)}...")
    
    try:
        if not os.path.exists(img_path):
            print(f"Error: File not found at {img_path}")
            print("-" * 30)
            continue

        with open(img_path, 'rb') as img_file:
            files = {'file': img_file}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data['class']}")
            print(f"Confidence: {data['confidence']}")
            print(f"Full Probs: {data['all_probabilities']}")
        else:
            print(f"Server Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Connection Error: {e}")

    print("-" * 30)

print("\n--- Testing Complete ---")