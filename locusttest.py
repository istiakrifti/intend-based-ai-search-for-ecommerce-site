from locust import HttpUser, task, between
import random
import string
import time

QUERIES = [
    "gaming laptop with good graphics",
    "cheap phone under 20000",
    "smartwatch with heart rate sensor",
    "4K television 55 inch",
    "wireless headphones",
    "budget ultrabook",
    "noise cancelling earbuds",
    "Android phone with best camera",
    "iPhone latest model",
    "laptop for video editing",
    "fitness tracker with GPS",
    "Bluetooth speaker waterproof",
    "DSLR camera under 50000",
    "home theater system",
    "tablet with stylus support",
    "portable hard drive 1TB",
    "best budget smartphone",
    "gaming mouse with RGB",
    "mechanical keyboard",
    "smart home assistant"
]

class SearchUser(HttpUser):
    wait_time = between(0.5, 2)  # simulate real user pacing

    @task(3)
    def perform_valid_search(self):
        query = random.choice(QUERIES)
        with self.client.get(f"/search?query={query}", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # if "products" in data and isinstance(data["products"], list) and data["products"]:
                    response.success()
                    # else:
                        # response.failure("Missing or empty 'products' key in response")
                except ValueError:
                    response.failure("Response is not valid JSON")
            else:
                response.failure(f"Failed with status code {response.status_code}")

    # @task(1)
    # def perform_random_garbage_search(self):
    #     # Generate a random string of 10 characters
    #     query = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    #     with self.client.get(f"/search?query={query}", catch_response=True) as response:
    #         if response.status_code == 200:
    #             try:
    #                 data = response.json()
                    
    #                 response.success()
                    
    #             except ValueError:
    #                 response.failure("Response is not valid JSON")
    #         else:
    #             response.failure(f"Failed with status code {response.status_code}")

    @task(1)
    def simulate_burst_load(self):
        for _ in range(5):  # simulate 5 rapid requests
            query = random.choice(QUERIES)
            with self.client.get(f"/search?query={query}", catch_response=True) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "products" in data and isinstance(data["products"], list) and data["products"]:
                            response.success()
                        else:
                            response.failure("Missing or empty 'products' key in response")
                    except ValueError:
                        response.failure("Response is not valid JSON")
                else:
                    response.failure(f"Failed with status code {response.status_code}")
            time.sleep(0.1)  # short delay between burst requests
