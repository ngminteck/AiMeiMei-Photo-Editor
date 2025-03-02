import requests
import base64


def get_api_key(filename="../../keys/googlevision.txt"):
    """Reads the API key from a text file."""
    try:
        with open(filename, "r") as file:
            api_key = file.read().strip()
            print(f"[DEBUG] Successfully read API key from {filename}")
            return api_key
    except Exception as e:
        print(f"[ERROR] Unable to read API key file: {e}")
        return None


# Read API Key
GOOGLE_API_KEY = get_api_key()


def encode_image(image_path):
    """Encodes image as base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            print(f"[DEBUG] Successfully encoded image {image_path} as base64.")
            return encoded_image
    except Exception as e:
        print(f"[ERROR] Failed to encode image {image_path}: {e}")
        return None


def recognize_landmarks(image_path):
    """Detects multiple landmarks in an image and returns a list with confidence percentages."""
    base64_image = encode_image(image_path)

    if not base64_image:
        print("[ERROR] Failed to encode image. Skipping landmark recognition.")
        return []

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
    request_body = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "LANDMARK_DETECTION"}]
            }
        ]
    }

    print(f"[DEBUG] Sending landmark recognition request to Google Vision API.")

    try:
        response = requests.post(url, json=request_body)
        result = response.json()
        print(f"[DEBUG] Google Vision API Response: {result}")

        if "landmarkAnnotations" in result["responses"][0]:
            landmarks = []
            for landmark in result["responses"][0]["landmarkAnnotations"]:
                name = landmark["description"]
                confidence = landmark["score"] * 100  # Convert to percentage
                lat_lng = landmark["locations"][0]["latLng"]
                landmarks.append({
                    "name": name,
                    "confidence": round(confidence, 2),
                    "latitude": lat_lng["latitude"],
                    "longitude": lat_lng["longitude"]
                })

            landmarks = sorted(landmarks, key=lambda x: x["confidence"], reverse=True)
            return landmarks
    except Exception as e:
        print(f"[ERROR] Landmark recognition failed: {e}")

    return []


# Example usage
landmarks = recognize_landmarks("images/test/2_people_together.jpg")

# Final output
print("\n=== FINAL OUTPUT ===")
for landmark in landmarks:
    print(f"Place: {landmark['name']} ({landmark['confidence']}%)")
    print(f"Location: ({landmark['latitude']}, {landmark['longitude']})")
