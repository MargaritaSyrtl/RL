import torch
import torch.optim as optim
import copy
import time
import matplotlib.pyplot as plt
import folium
from folium.features import DivIcon
from branca.element import Template, MacroElement

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class A2CAgent(object):
    
    def __init__(self, actor, critic, args, env, dataGen):
        self.actor = actor
        self.critic = critic 
        self.args = args 
        self.env = env 
        self.dataGen = dataGen
        print("agent is initialized")
        
    def train(self):
        args = self.args 
        env = self. env 
        dataGen = self.dataGen
        actor = self.actor
        critic = self.critic 
        prt = self.prt 
        actor.train()
        critic.train()
        max_epochs = args['n_train']

        actor_optim = optim.Adam(actor.parameters(), lr=args['actor_net_lr'])
        critic_optim = optim.Adam(critic.parameters(), lr=args['critic_net_lr'])
        
        best_model = 1000
        val_model = 1000
        r_test = []
        r_val = []
        s_t = time.time()
        print("training started")
        for i in range(max_epochs):
            
            data = dataGen.get_train_next()
            env.input_data = data 
            state, avail_actions = env.reset()
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
            # [b_s, hidden_dim, n_nodes]
            static_hidden = actor.emd_stat(data).permute(0, 2, 1)
            # critic inputs 
            static = torch.from_numpy(env.input_data[:, :, :2].astype(np.float32)).permute(0, 2, 1).to(device)
            w = torch.from_numpy(env.input_data[:, :, 2].reshape(env.batch_size, env.n_nodes, 1).astype(np.float32)).to(device)
            
            # lstm initial states 
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
            last_hh = (hx, cx)
       
            # prepare input 
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
       
            #[n_nodes, rem_time]
            time_vec_truck = np.zeros([env.batch_size, 2])
            # [n_nodes, rem_time, weigth]
            time_vec_drone = np.zeros([env.batch_size, 3])
            
            # storage containers 
            logs = []
            actions = []
            probs = []
            time_step = 0
            
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(device)
                for j in range(2):
                    # truck takes action 
                    if j == 0:
                        avail_actions_truck = torch.from_numpy(avail_actions[:, :, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(device)
                        dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, 0], 2)).to(device)
                        idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input, last_hh, 
                                                                     terminated, avail_actions_truck)
                        b_s = np.where(np.logical_and(avail_actions[:, :, 1].sum(axis=1)>1, env.sortie==0))[0]
                        avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                        avail_actions_drone = torch.from_numpy(avail_actions[:, :, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(device)
                        idx = idx_truck 
                    else:
                        dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, 1], 2)).to(device)
                        idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input, last_hh, 
                                                                     terminated, avail_actions_drone)
                        idx = idx_drone 
                
                    decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                    logs.append(logp.unsqueeze(1))
                    actions.append(idx.unsqueeze(1))
                    probs.append(prob.unsqueeze(1))
               
                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck.cpu().numpy(), idx_drone.cpu().numpy(), time_vec_truck, time_vec_drone, ter)
                
             
                time_step += 1
         
            print("epochs: ", i)
            actions = torch.cat(actions, dim=1)  # (batch_size, seq_len)
            logs = torch.cat(logs, dim=1)  # (batch_size, seq_len)
            # Query the critic for an estimate of the reward
            critic_est = critic(static, w).view(-1)
            R = env.current_time.astype(np.float32) 
            R = torch.from_numpy(R).to(device)
            advantage = (R - critic_est)
            actor_loss = torch.mean(advantage.detach() * logs.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
           
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args['max_grad_norm'])
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args['max_grad_norm'])
            critic_optim.step()
        
            e_t = time.time() - s_t
            print("e_t: ", e_t)
            if i % args['test_interval'] == 0:
        
                R = self.test()
                r_test.append(R)
                np.savetxt("trained_models/test_rewards.txt", r_test)
            
                print("testing average rewards: ", R)
                if R < best_model:
                 #   R_val = self.test(inference=False, val=False)
                    best_model = R
                    num = str(i // args['save_interval'])
                    torch.save(actor.state_dict(), 'trained_models/' + '/' + 'best_model' + '_actor_truck_params.pkl')
                    torch.save(critic.state_dict(), 'trained_models/' + '/' + 'best_model' + '_critic_params.pkl')
                   
            if i % args['save_interval'] ==0:
                num = str(i // args['save_interval'])
                torch.save(actor.state_dict(), 'trained_models/' + '/' + num + '_actor_truck_params.pkl')
                torch.save(critic.state_dict(), 'trained_models/' + '/' + num + '_critic_params.pkl')

    def test(self):
        args = self.args 
        env = self.env 
        dataGen = self.dataGen
        actor = self.actor
        prt = self.prt 
        actor.eval()
        
        data = dataGen.get_test_all()
        env.input_data = data 
        state, avail_actions = env.reset()

        time_vec_truck = np.zeros([env.batch_size, 2])
        time_vec_drone = np.zeros([env.batch_size, 3])
        # sols = []
        # costs = []
        with torch.no_grad():
            # coordinates
            data = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
            # print(data)
            static_hidden = actor.emd_stat(data).permute(0, 2, 1)
          
            # lstm initial states 
            hx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
            cx = torch.zeros(1, env.batch_size, args['hidden_dim']).to(device)
            last_hh = (hx, cx)
       
            # prepare input 
            ter = np.zeros(env.batch_size).astype(np.float32)
            decoder_input = static_hidden[:, :, env.n_nodes-1].unsqueeze(2)
            time_step = 0
            sols = [[] for _ in range(env.batch_size)]
            prev_truck_pos = [env.n_nodes - 1] * env.batch_size  # все стартуют из депо
            while time_step < args['decode_len']:
                terminated = torch.from_numpy(ter.astype(np.float32)).to(device)
                for j in range(2):
                    # truck takes action 
                    if j == 0:
                        avail_actions_truck = torch.from_numpy(avail_actions[:, :, 0].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(device)
                        dynamic_truck = torch.from_numpy(np.expand_dims(state[:, :, 0], 2)).to(device)
                        idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dynamic_truck, decoder_input, last_hh, 
                                                                     terminated, avail_actions_truck)
                        b_s = np.where(np.logical_and(avail_actions[:, :, 1].sum(axis=1)>1, env.sortie==0))[0]
                        avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                        avail_actions_drone = torch.from_numpy(avail_actions[:, :, 1].reshape([env.batch_size, env.n_nodes]).astype(np.float32)).to(device)
                        idx = idx_truck
                    else:
                        dynamic_drone = torch.from_numpy(np.expand_dims(state[:, :, 1], 2)).to(device)
                        idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dynamic_drone, decoder_input, last_hh, 
                                                                     terminated, avail_actions_drone)
                        idx = idx_drone
                    decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(env.batch_size, args['hidden_dim'], 1)).detach()
                
                state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(idx_truck.cpu().numpy(), idx_drone.cpu().numpy(), time_vec_truck, time_vec_drone, ter)
                time_step += 1
                # sequence of nodes that the truck and drone choose
                # sequence of action pairs [truck_node, drone_node] chosen by the agent at each time step for a particular task with index n from the batch.
                for b in range(env.batch_size):
                    sols[b].append((idx_truck[b], idx_drone[b], prev_truck_pos[b]))
                    prev_truck_pos[b] = idx_truck[b].item()

                # sols.append([idx_truck[n], idx_drone[n]])  # idx of nodes served by the truck and the drone

        # print(f"sols: {sols}")

        R = copy.copy(env.current_time)
        # costs.append(env.current_time[n])
        print("finished: ", sum(terminated))

        fname = 'test_results-{}-len-{}.txt'.format(args['test_size'], args['n_nodes'])
        fname = 'results/' + fname
        # R is saved - an array with the total time costs for each task in the test set (one number per task)
        np.savetxt(fname, R)
        actor.train()

        ###
        # find the best route and visualise
        # best_idx = np.argmin(R)
        best_idx = 0  # for the first set of nodes
        best_reward = R[best_idx]
        print(f"Best route: task №{best_idx}, time: {best_reward:.2f}")
        route_truck = [env.n_nodes - 1]
        for idx in sols[best_idx]:
            route_truck.append(idx[0].item())

        route_drone = [env.n_nodes - 1]
        for idx in sols[best_idx]:
            route_drone.append(idx[1].item())

        #for idx in sols[best_idx]:
        #    print(idx)
        drone_starts = [idx[2] for idx in sols[best_idx]]

        print(f"truck: {route_truck}")
        print(f"drone: {route_drone}")
        print(f"drone starts: {drone_starts}")
        coords = env.input_data[best_idx, :, :2]

        plot_route(
            coords=np.array(coords),
            route_truck=route_truck,
            route_drone=route_drone,
            depot_idx=env.n_nodes - 1,
            title=f"Route Example {best_idx}",
            save_path=f"results/route_{best_idx}.png")
        ###

        return R.mean()  # average service time for the test set. This is a metric of the quality of the model during validation.

    def sampling_batch(self, sample_size):
        self.use_google_for_first = False  # default False

        data_all = self.dataGen.get_test_all()
        num_graphs = data_all.shape[0]
        args = self.args
        env = self.env
        actor = self.actor

        actor.eval()
        actor.set_sample_mode(True)

        best_rewards_list = []
        times = []
        initial_t = time.time()

        for graph_idx in range(num_graphs):
            graph = data_all[graph_idx:graph_idx + 1]  # [1, n_nodes, 3]
            if graph_idx == 0:
                coords = graph[0, :, :2]
                rescaled_coords = scaled_to_latlon(coords)
                np.savetxt("data/rescaled_coords.txt", rescaled_coords, fmt="%.10f", delimiter=",",
                           comments='')
            env.use_google_for_first = (graph_idx == 0)
            # print(env.use_google_for_first)

            data = np.repeat(graph, sample_size, axis=0)  # [sample_size, n_nodes, 3]
            env.input_data = data
            state, avail_actions = env.reset()

            time_vec_truck = np.zeros([sample_size, 2])
            time_vec_drone = np.zeros([sample_size, 3])
            sols = [[] for _ in range(sample_size)]
            prev_truck_pos = [env.n_nodes - 1] * sample_size  # reset for this graph

            with torch.no_grad():
                coords = torch.from_numpy(data[:, :, :2].astype(np.float32)).to(device)
                static_hidden = actor.emd_stat(coords).permute(0, 2, 1)

                hx = torch.zeros(1, sample_size, args['hidden_dim']).to(device)
                cx = torch.zeros(1, sample_size, args['hidden_dim']).to(device)
                last_hh = (hx, cx)

                ter = np.zeros(sample_size).astype(np.float32)
                decoder_input = static_hidden[:, :, env.n_nodes - 1].unsqueeze(2)
                time_step = 0

                while time_step < args['decode_len']:
                    terminated = torch.from_numpy(ter).to(device)
                    for j in range(2):
                        if j == 0:
                            avail_truck = torch.from_numpy(
                                avail_actions[:, :, 0].reshape([sample_size, env.n_nodes]).astype(np.float32)).to(
                                device)
                            dyn_truck = torch.from_numpy(np.expand_dims(state[:, :, 0], 2)).to(device)
                            idx_truck, prob, logp, last_hh = actor.forward(static_hidden, dyn_truck, decoder_input,
                                                                           last_hh, terminated, avail_truck)
                            b_s = np.where(np.logical_and(avail_actions[:, :, 1].sum(axis=1) > 1, env.sortie == 0))[0]
                            avail_actions[b_s, idx_truck[b_s].cpu(), 1] = 0
                            avail_drone = torch.from_numpy(
                                avail_actions[:, :, 1].reshape([sample_size, env.n_nodes]).astype(np.float32)).to(
                                device)
                            idx = idx_truck
                        else:
                            dyn_drone = torch.from_numpy(np.expand_dims(state[:, :, 1], 2)).to(device)
                            idx_drone, prob, logp, last_hh = actor.forward(static_hidden, dyn_drone, decoder_input,
                                                                           last_hh, terminated, avail_drone)
                            idx = idx_drone

                        decoder_input = torch.gather(static_hidden, 2,
                                                     idx.view(-1, 1, 1).expand(sample_size, args['hidden_dim'],
                                                                               1)).detach()

                    state, avail_actions, ter, time_vec_truck, time_vec_drone = env.step(
                        idx_truck.cpu().numpy(), idx_drone.cpu().numpy(),
                        time_vec_truck, time_vec_drone, ter
                    )
                    time_step += 1

                    for b in range(sample_size):
                        sols[b].append((idx_truck[b], idx_drone[b], prev_truck_pos[b]))
                        prev_truck_pos[b] = idx_truck[b].item()

            R = env.current_time  # shape: [sample_size]
            best_i = np.argmin(R)
            best_cost = R[best_i]
            best_rewards_list.append(best_cost)

            # visualize best route for graph with idx
            if graph_idx == 0:
                geom_dict = env.geom_dict
                traffic = env.traffic
                best_sol = sols[best_i]
                route_truck = [env.n_nodes - 1] + [step[0].item() for step in best_sol]
                route_drone = [env.n_nodes - 1] + [step[1].item() for step in best_sol]
                print(f"truck: {route_truck}")
                print(f"drone: {route_drone}")

                coords = graph[0, :, :2]

                # folium visualisation
                rescaled_coords = scaled_to_latlon(coords)
                print(f"rescaled coords: {rescaled_coords}")
                places = env.places
                plot_folium(places,
                            route_truck=route_truck,
                            route_drone=route_drone,
                            geom_dict=geom_dict,
                            traffic=traffic,
                            makespan=best_rewards_list[0],
                            depot_idx=env.n_nodes - 1
                            )

                # matplot visualisation
                #plot_route(
                #    coords=np.array(coords),
                #    route_truck=route_truck,
                #    route_drone=route_drone,
                #    depot_idx=env.n_nodes - 1,
                #    title=f"Best Route for Graph {graph_idx}",
                #    save_path=f"results/best_route_graph_{graph_idx}.png"
                #)

        # Save all best costs
        np.savetxt(f'results/best_rewards_list_{sample_size}_samples.txt', best_rewards_list)
        print(f"ended")
        return best_rewards_list, times


def plot_folium(places, route_truck, route_drone, geom_dict, traffic, makespan, depot_idx=None):
    """Visualises truck's and drone's optimal routes in real coordinates via folium."""

    print(f"makespan={makespan}")
    # time_sec = makespan * const_res / 10  # todo scale??
    time_sec = makespan
    print(f"makespan in seconds={time_sec}")
    hours, remainder = divmod(time_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        res_time = f"{int(hours)}h:{int(minutes)}min"
    else:
        res_time = f"{int(minutes)}min"
    print(f"resulting time={res_time}")

    if depot_idx is None:
        depot_idx = len(places) - 1
    depot = places[depot_idx]
    m = folium.Map(depot, zoom_start=14)

    # marker for each node (as given at the beginning)
    for i, (lat, lon) in enumerate(places):
        folium.Marker(
            [lat, lon],
            tooltip=f"node {i}",
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f"""
                                        <div style="background:blue; 
                                                   color:white; 
                                                   text-align:center; 
                                                   border-radius:10px; 
                                                   width:20px; height:20px;
                                                   line-height:20px;">
                                                   {0 if i == depot_idx else i+1}
                                        </div>"""
            )
        ).add_to(m)

    def compress_route(route):
        """remove duplicate nodes """
        return [route[i] for i in range(len(route)) if i == 0 or route[i] != route[i - 1]]

    cleaned_truck = compress_route(route_truck)
    cleaned_drone = compress_route(route_drone)

    # truck with geometry
    for a, b in zip(cleaned_truck, cleaned_truck[1:]):
        key = frozenset([places[a], places[b]])
        path = geom_dict.get(key, [places[a], places[b]])
        ratio = traffic.get(key, 1.0)
        if ratio < 1.1:
            color = "green"  # green
        elif ratio < 1.5:
            color = "#FFC107"  # orange
        else:
            color = "#E74C3C"  # red
        folium.PolyLine(
            path,
            color=color,
            weight=5,
            tooltip=f"Truck {a+1}→{b+1}, traffic ratio: {ratio:.2f}"  # human readable
        ).add_to(m)

    # drone
    for i in range(len(cleaned_drone) - 1):
        a, b = cleaned_drone[i], cleaned_drone[i + 1]
        folium.PolyLine(
            [places[a], places[b]],
            color="#1E90FF",
            weight=3,
            dash_array="6 8",
            tooltip=f"Drone {a+1}→{b+1}"  # human readable
        ).add_to(m)

    truck_str = " → ".join(str(0 if i == depot_idx else i + 1) for i in cleaned_truck)
    drone_str = " → ".join(str(0 if i == depot_idx else i + 1) for i in cleaned_drone)

    legend_html = f"""
            <div style="
                 position: fixed;
                 bottom: 30px; right: 30px;
                 z-index: 9999;
                 background: rgba(255,255,255,0.9);
                 padding: 10px 14px;
                 border: 2px solid #999;
                 border-radius: 6px;
                 box-shadow: 3px 3px 6px rgba(0,0,0,0.25);
                 font-size: 14px; line-height: 1.5;">
              <b>Optimal route:&nbsp;</b><br>
              <span style="color:#000000; font-weight:600;">
                Truck route: &nbsp;{truck_str}
              </span><br>
              <span style="color:#000000; font-weight:600;">
                Drone route: &nbsp;{drone_str}
              </span><br>
              <span style="color:#000000; font-weight:600;">
                Travel time: &nbsp;{res_time}
              </span>
            </div>"""

    macro = MacroElement()
    macro._template = Template(f"{{% macro html(this, kwargs) %}}{legend_html}{{% endmacro %}}")
    m.get_root().add_child(macro)
    # save
    filename_to_save = "results/opt_route.html"
    m.save(filename_to_save)
    print(f"saved in {filename_to_save}")


def plot_route(coords, route_truck, route_drone, depot_idx=None, title="Optimal Route", save_path=None):
    """Visualises truck's and drone's optimal routes in conventional units via matplotlib."""
    plt.figure(figsize=(8, 8))

    x, y = coords[:, 0], coords[:, 1]

    if depot_idx is None:
        depot_idx = coords.shape[0] - 1
    depot = coords[depot_idx]
    plt.scatter(x, y, c='gray', label='Nodes')
    plt.scatter(depot[0], depot[1], c='red', marker='*', s=200, label='Depot')

    # Truck route
    truck_path = coords[route_truck]
    plt.plot(truck_path[:, 0], truck_path[:, 1], '-o', label='Truck')

    # Drone route
    drone_path = coords[route_drone]
    plt.plot(drone_path[:, 0], drone_path[:, 1], '--x', label='Drone')

    for i, (xi, yi) in enumerate(coords):
        label = "0" if i == depot_idx else str(i + 1 if i < depot_idx else i)
        plt.text(
            xi + 1, yi + 1,  # небольшое смещение
            label,
            fontsize=10,  # увеличить шрифт
            fontweight='bold',  # жирный
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')  # фон
        )
        # plt.text(xi + 0.01, yi + 0.01, str(i), fontsize=8)

    plt.legend()
    plt.title(title)
    plt.grid(True)

    max_range = max(coords[:, 0].max(), coords[:, 1].max())
    padding = max_range * 0.05
    plt.xlim(0, max_range + padding)
    plt.ylim(0, max_range + padding)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


#### rescale
import json
from pyproj import Proj, transform
import numpy as np

# Define WGS84 (lat/lon) and UTM projection
utm = Proj(proj='utm', zone=32, datum='WGS84')
wgs84 = Proj(proj='latlong', datum='WGS84')


def xy_to_latlon(x, y):
    """
    UTM (x, y) → (lat, lon)
    """
    lon, lat = transform(utm, wgs84, x, y)
    return np.array([lat, lon])


# const
with open("data/DroneTruck-meta.json") as f:
    meta = json.load(f)

x_min = meta["x_min"]
x_max = meta["x_max"]
y_min = meta["y_min"]
y_max = meta["y_max"]

scale_const = max(x_max - x_min, y_max - y_min)
const_res = scale_const / 100
print(f"const: {const_res}")


def scaled_to_xy(scaled):
    """[1, 100] → meters (UTM)"""
    x_s, y_s = scaled.T
    x = x_min + x_s * const_res
    y = y_min + y_s * const_res
    return np.vstack([x, y]).T


def scaled_to_latlon(scaled_coords):
    """
    [1, 100] -> (lat, lon)
    """
    # [1, 100] -> meters (UTM)
    xy_coords = scaled_to_xy(scaled_coords)
    # -> lat,lon
    return np.array([xy_to_latlon(x, y) for x, y in xy_coords])
