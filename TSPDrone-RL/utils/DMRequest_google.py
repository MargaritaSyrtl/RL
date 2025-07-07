import requests
from itertools import combinations


class DMRequest(object):
    def __init__(self, places):
        self.places = places
        self.api_key = ""
        self.distance_base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        self.directions_base_url = "https://maps.googleapis.com/maps/api/directions/json"

    def get_response_data_ga(self):
        """
        Generates data for the GA algorithm.
        Gets distances, durations and route geometry from the Google API.
        """
        data = {
            'waypoints_distances': {},
            'waypoints_durations': {},
            'waypoints_geometries': {},
            'waypoints_traffic': {}
        }

        for wp1, wp2 in combinations(self.places, 2):
            origin = f"{wp1[0]},{wp1[1]}"
            destination = f"{wp2[0]},{wp2[1]}"
            key = frozenset([wp1, wp2])

            # Distance Matrix API
            try:
                params = {
                "origins": origin,
                "destinations": destination,
                "mode": "driving",
                "departure_time": "now",
                "key": self.api_key
                }
                response = requests.get(self.distance_base_url, params=params).json()
                if response["status"] == "OK":
                    element = response["rows"][0]["elements"][0]
                    if element["status"] == "OK":
                        dist = element["distance"]["value"]
                        dur = element["duration"]["value"]
                        data['waypoints_distances'][key] = dist
                        data['waypoints_durations'][key] = dur
                        # get duration in traffic
                        dur_traffic = element.get("duration_in_traffic", {}).get("value", dur)
                        # calculate the traffic jam coefficient
                        ratio = dur_traffic / dur if dur > 0 else 1.0
                        data['waypoints_traffic'][key] = ratio
                    else:
                        print(f"Distance Matrix API error: {element['status']}")
                else:
                    print(f"Google API general error: {response['status']}")
            except Exception as e:
                print(f"Failed to get distance for {wp1} -> {wp2}: {e}")

            # Directions API (for geometry)
            try:
                params = {
                "origin": origin,
                "destination": destination,
                "mode": "driving",
                "key": self.api_key
                }
                response = requests.get(self.directions_base_url, params=params).json()
                if response.get("status") == "OK":
                    steps = response["routes"][0]["legs"][0]["steps"]
                    geometry = []
                    for step in steps:
                        start = step["start_location"]
                        geometry.append((start["lat"], start["lng"]))
                    end = steps[-1]["end_location"]
                    geometry.append((end["lat"], end["lng"]))
                    data["waypoints_geometries"][key] = geometry
                else:
                    print(f"Directions API error: {response.get('status')}")
            except Exception as e:
                print(f"Failed to get geometry for {wp1} -> {wp2}: {e}")

        return data
