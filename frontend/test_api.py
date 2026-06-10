import requests

url = "http://127.0.0.1:5000/predict"

files = {
    "image": open(r"C:\EyeProject\combined_dir\moderate_gla\4055_right.jpeg", "rb")
}

response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.text)