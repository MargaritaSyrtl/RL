import matplotlib.pyplot as plt
import json


def visualize_instance(idx,
                       paths_file="results/test_paths.json",
                       data_file="data/DroneTruck-size-100-len-11.txt",
                       html_out="route_map.html"):
    with open(data_file, 'r') as file:
        first_line = file.readline().strip()
    print(first_line)

    # Преобразуем строку в список float-значений
    coords_data = list(map(float, first_line.split()))
    print(coords_data)
    coordinates = []
    for i in range(0, len(coords_data), 3):
        x = coords_data[i]
        y = coords_data[i + 1]
        coordinates.append((x, y))
    print(coordinates)
    # todo
    # [(40.0, 30.5), (36.5, 37.0), (27.5, 36.5), (24.5, 47.5), (24.0, 43.5), (7.0, 45.5), (36.5, 35.0), (41.5, 43.0), (41.0, 25.5), (28.0, 13.0), (0.4091140241497346, 0.37472207480314396)]
    # Load paths (truck + drone)
    with open(paths_file) as f:
        paths = json.load(f)
    truck_raw = paths["truck"][idx]
    drone_raw = paths["drone"][idx]
    print(f"truck: {truck_raw}")
    print(f"drone: {drone_raw}")

    def extract_coords(route, coords):
        return [coords[i] for i in route]

    truck_coords = extract_coords(truck_raw, coordinates)
    drone_coords = extract_coords(drone_raw, coordinates)
    print(f"truck_coords: {truck_coords}")
    print(f"drone_coords: {drone_coords}")

    tx, ty = zip(*truck_coords)
    dx, dy = zip(*drone_coords)
    print(tx)
    print(ty)

    plt.figure(figsize=(10, 8))
    plt.plot(tx, ty, '-o', label='Truck', linewidth=2)
    plt.plot(dx, dy, '-s', label='Drone', linewidth=2)

    # Подписи всех точек
    for i, (x, y) in enumerate(coordinates):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=9)

    # Подсветим депо
    depot_x, depot_y = coordinates[-1]
    plt.plot(depot_x, depot_y, 'r*', markersize=12, label='Depot')

    plt.title('Truck and Drone Routes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def test(idx, data_file="data/DroneTruck-size-100-len-10.txt",):
    with open(data_file, 'r') as file:
        first_line = file.readline().strip()

    # Преобразуем строку в список float-значений
    coords_data = list(map(float, first_line.split()))

    coordinates = []
    for i in range(0, len(coords_data), 3):
        x = coords_data[i]
        y = coords_data[i + 1]
        coordinates.append((x, y))

    # Разделим X и Y координаты для построения
    xs, ys = zip(*coordinates)

    plt.figure(figsize=(10, 8))

    # Рисуем все точки (кроме депо)
    plt.scatter(xs[:-1], ys[:-1], c='blue', label='Customers', s=50)

    # Рисуем депо отдельно
    depot_x, depot_y = coordinates[-1]
    plt.scatter(depot_x, depot_y, c='red', marker='*', s=150, label='Depot')

    # Подписи всех точек
    for i, (x, y) in enumerate(coordinates):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=9)

    plt.title('Truck and Drone Routes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()



visualize_instance(3)
#test(0)


