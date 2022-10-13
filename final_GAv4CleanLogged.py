
# import modules and packages

import numpy as np
import random
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
import pandas as pd
import warnings
import time
import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gnn', default='GCN')
parser.add_argument('--location', default='2022-08-31_12:33:27-GCN_32_1')
parser.add_argument('--mol_len', default=4)
parser.add_argument('--num_layers', default = 1)
parser.add_argument('--num_nodes', default = 32)
parser.add_argument('--best_known_val', default = 3.0730981546883953)

args = parser.parse_args()

gnn = str(args.gnn)
directory = str(args.location)
n = int(args.mol_len)
num_nodes = int(args.num_nodes)
num_layers = int(args.num_layers)
best_known_val = float(args.best_known_val)

loc = f'trained_models_{gnn}/{directory}/{gnn}_{num_nodes}_{num_layers}.tar'
PATH = loc


gnn_layer_by_name = {
    "GCN": GCNConv,
    "GraphSAGE": SAGEConv
}


def extract_matrix_string(A_sparse):
    if isinstance(A_sparse, (np.ndarray, np.generic) ):
        A = A_sparse
    else:
        A = A_sparse.todense()
        
    n = A.shape[0]
    mat_string = []
    for i in range(n-1):
        for j in range(i+1,n):
            mat_string.append(A[i,j])
    
    return np.array(mat_string)


def build_matrix_from_mat_string(mat_string, n):
    A = np.zeros((n, n))
    pos_counter = 0
    for i in range(n-1):
        for j in range(i+1,n):
            A[i,j] = mat_string[pos_counter]
            pos_counter += 1
            A[j,i] = A[i,j]
    
    return(A)


def generate_atom_types(A, degree_sequence):
    atom_types = []
    
    for degree in degree_sequence:
        if degree == 1:
            atom_type = random.randint(0,3)
        elif degree == 2:
            atom_type = random.randint(0,1)
        else:
            atom_type = 0
            
        atom_types.append(atom_type)
        
    return atom_types


def generate_graphs(num_graphs, n):
    num_generated_graphs = 0
    degree_sequence_list = []
    gene_list = []
    atom_types = []
    
    while num_generated_graphs < num_graphs:
        try:
            degree_sequence = [random.randint(1,4) for i in range(n)]
            G = nx.random_degree_sequence_graph(degree_sequence)
            A = nx.adjacency_matrix(G)
            if nx.is_connected(G):
                gene_list.append(extract_matrix_string(A))
                atom_types.append(generate_atom_types(A, degree_sequence))
                num_generated_graphs += 1
        except:
            pass
        
        
    return list(zip(gene_list, atom_types))


class Net(torch.nn.Module):
    def __init__(self, layer_name = gnn, c_in = 14, c_hidden = num_nodes, c_out = 32, num_layers = num_layers):
        super(Net, self).__init__()

        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden

        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, out_channels, bias = False, aggr = 'add')
            ]
            in_channels = c_hidden

        layers += [
                gnn_layer(in_channels, c_out, bias = False, aggr = 'add')
            ]

        self.layers = torch.nn.ModuleList(layers)
        
        self.fc11 = torch.nn.Linear(32, 16)
        self.fc12 = torch.nn.Linear(16, 1)

    def forward(self, data, print_intermediaries = False):
        
        for i, l in enumerate(self.layers):
            if i == 0:
                output_GNN = F.relu(l(data.x, data.edge_index))
            else:
                output_GNN = F.relu(l(output_GNN, data.edge_index))

        
        # print(f'>this is what we are looking for?: {self.training}')

        out_scatter = output_GNN.sum(axis=0)
     
        out_ANN1 = F.relu(self.fc11(out_scatter))
        
        out_final = self.fc12(out_ANN1)
        
        return out_final
    

# loc = 'trained_models/2022-09-16_21:28:10-GraphSAGE_64_3/GraphSAGE_64_3.tar'
# PATH = f'{loc}'
model = Net()
print(model)
state_dict = torch.load(PATH)
print(f'{state_dict.keys()=}')
model.load_state_dict(torch.load(PATH))
model.eval()

def atom_and_neighbour_to_feature_vec(atom_number, neighbours):
    num_atoms = 4
    atom_covalence = [4, 2, 1, 1]
    num_neighbours = 5
    num_h_neighbours = 5
    neighbours = int(neighbours)
    
    if neighbours > atom_covalence[atom_number]:
        raise Exception('this is not possible')
    calculated_hs = atom_covalence[atom_number] - neighbours
    
    feature_vec = [0] * (num_atoms + num_neighbours + num_h_neighbours)
    
    feature_vec[atom_number] = 1
    feature_vec[num_atoms + neighbours] = 1
    feature_vec[num_atoms + num_neighbours + calculated_hs] = 1
        
    return(feature_vec)


def prep_gene_for_nn(A, atom_types):
    degrees = np.sum(A, axis = 0)
    n = A.shape[0]
    A_diag = A + np.identity(n)
    A_diag = A 

    # print(A_diag)

    # create edge_index
    G = nx.from_numpy_array(A_diag)
    adj = nx.to_scipy_sparse_matrix(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    
    # create feat_vecs
    feat_vecs = []
    for i in range(len(degrees)):
        feat_vecs.append(atom_and_neighbour_to_feature_vec(atom_types[i], degrees[i]))
        
    x_data = torch.tensor(feat_vecs).to(torch.float)
    data = Data(x = x_data, edge_index = edge_index)

    return data


def fitness_func(gene, atom_types):
    n = len(atom_types)

    A = build_matrix_from_mat_string(gene, n)
    fitness = 0
    
    
    if not(nx.is_connected(nx.from_numpy_array(A))):
        fitness = -1000
        
    else:
        n = A.shape[0]
        
        data = prep_gene_for_nn(A, atom_types)
        fitness = float(model(data))
    
    return fitness


def create_molecule_from_matrix_and_atom_types_3(A, atom_types):
    G = nx.from_numpy_array(A)
    degrees = [int(i) for i in np.sum(A, axis=0)]
    atom_type_to_valence_dict = {0: 4, 1: 2, 2: 1, 3: 1}
    allowed_degrees = [atom_type_to_valence_dict[atom] for atom in atom_types]

    counter = 0
    # create connected graph with degree <= 4 from 
    while nx.number_connected_components(G) > 1:
        A = nx.to_numpy_array(G)
        degrees = [int(i) for i in np.sum(A, axis=0)]
        nodes_in_component = list(nx.node_connected_component(G, 0))
        available_nodes = [item for item in list(np.arange(n)) if item not in nodes_in_component]
        node_to_be_added = random.choice(available_nodes)

        link_list = [i for i in range(n) if (degrees[i] < 4) and i in nodes_in_component]

        if link_list:
            rand_node_in_component = random.choice(link_list)
        else:
            rand_node_in_component = random.choice(nodes_in_component)
        
        G.add_edge(rand_node_in_component,node_to_be_added)

        A = nx.to_numpy_array(G)
        degrees = [int(i) for i in np.sum(A, axis=0)]

        for z in range(n):
            while degrees[z] > 4:
                neighbors = list(nx.all_neighbors(G,z))
                remove_this_neighbor = random.choice(neighbors)
                G.remove_edge(z,remove_this_neighbor)
                A = nx.to_numpy_array(G)
                degrees = [int(i) for i in np.sum(A, axis=0)]
        
        counter += 1

    for z in range(n):
            while degrees[z] > 4:
                neighbors = list(nx.all_neighbors(G,z))
                remove_this_neighbor = random.choice(neighbors)
                G.remove_edge(z,remove_this_neighbor)
                A = nx.to_numpy_array(G)
                degrees = [int(i) for i in np.sum(A, axis=0)]

    degrees = [int(i) for i in np.sum(A, axis=0)]
    allowed_degrees = [atom_type_to_valence_dict[atom] for atom in atom_types]


    
    A = nx.to_numpy_array(G)
    degrees = [int(i) for i in np.sum(A, axis=0)]
    atom_type_to_valence_dict = {0: 4, 1: 2, 2: 1, 3: 1}
    allowed_degrees = [atom_type_to_valence_dict[atom] for atom in atom_types]
    
    for i in range(n):
        if degrees[i] > allowed_degrees[i]:
            if degrees[i] > 2:
                atom_types[i] = 0
            else:
                atom_types[i] = 1
    

    degrees = [int(i) for i in np.sum(A, axis=0)]
    allowed_degrees = [atom_type_to_valence_dict[atom] for atom in atom_types]
    
    for i in range(n):
        if degrees[i] > allowed_degrees[i]:
            print('something is wrong!')


    return A, atom_types


def find_cut_node(n, cross_position):
    vec_length = ((n - 1)*n)/2
    lenghts = [n - 1 - i for i in range(n-1)]
    indices = np.cumsum(lenghts)
    position = None
    
    if cross_position <= indices[0]:
        position = 0
    else:
        for i in range(len(indices) - 1):
            if indices[i] < cross_position <= indices[i + 1]:
                position = i + 1
            
    return position


def cross_over(parent_1, parent_2, n, cross_position):
    parent_1_string = parent_1[0]
    parent_1_atom_types = parent_1[1]
    
    parent_2_string = parent_2[0]
    parent_2_atom_types = parent_2[1]
    
    # child creation
    child_1_string = np.concatenate([parent_1_string[:cross_position], parent_2_string[cross_position:]])
    child_2_string = np.concatenate([parent_2_string[:cross_position], parent_1_string[cross_position:]])
    
    child_1_mat = build_matrix_from_mat_string(child_1_string, n)
    child_2_mat = build_matrix_from_mat_string(child_2_string, n)
    
    cross_node = find_cut_node(n, cross_position) + 1
    
    child_1_atom_types =  np.concatenate([parent_1_atom_types[:cross_node], parent_2_atom_types[cross_node:]])
    child_2_atom_types =  np.concatenate([parent_2_atom_types[:cross_node], parent_1_atom_types[cross_node:]])
    
    #fixing the child lol and returning degree and 
    new_A_1, child_1_atom_types = create_molecule_from_matrix_and_atom_types_3(child_1_mat, child_1_atom_types)
    new_A_2, child_2_atom_types = create_molecule_from_matrix_and_atom_types_3(child_2_mat, child_2_atom_types)
    
    child_1_string = [int(x) for x in list(extract_matrix_string(new_A_1))]
    child_2_string = [int(x) for x in list(extract_matrix_string(new_A_2))]
        
    return child_1_string, child_1_atom_types, child_2_string, child_2_atom_types


def random_non_repeating_pairs(num):
    num_list = list(range(num))

    assert(num % 2 == 0)

    pair_list = []
    while num_list:
        random1 = random.choice(num_list)
        num_list.remove(random1)
        random2 = random.choice(num_list)
        num_list.remove(random2)

        pair = [random1, random2]
        pair_list.append(pair)

    return pair_list


def selection3(pop, num_elites, pop_size):
    #input of this function is [string, atom_types, fitness score]
    current_pop = pd.DataFrame(pop)
    pop_highest_val = current_pop.nlargest(1, 2)
    
    elites = current_pop.nlargest(num_elites, 2)
    remove_list = list(elites.index.values)
    remaining_pop = current_pop.drop(current_pop.index[remove_list])

    rem_pop_len = len(remaining_pop)
    remaining_pop = remaining_pop.sort_values(by=2)
    
    #selection probability
    ranked_probability_list = [(1/r) for r in range(2,rem_pop_len + 2)]
    cumsum = np.cumsum(ranked_probability_list)
    sum = cumsum[-1]
    remaining_pop[3] = cumsum 
    remaining_pop[4] = remaining_pop[3].shift(1)
    remaining_pop[4].iloc[0] = 0
    
    # select them
    evo_step_mols = []
    while len(evo_step_mols) < pop_size - num_elites:
        rand_val = random.random() *  sum
        the_chosen_one = remaining_pop[(rand_val < remaining_pop[3]) & (rand_val > remaining_pop[4])]
        chosen_as_list = the_chosen_one[[0,1]].values.tolist()[0]
        evo_step_mols.append(chosen_as_list)

    elites = elites[[0,1]].values.tolist()
    pairs = random_non_repeating_pairs(int((pop_size - num_elites)))

    return evo_step_mols, elites, pairs, pop_highest_val


def mutate_string(string):
    mutated_string = []
    mutated = 0
    for bit in string:
        random_val_string = random.uniform(0,1)
        if random_val_string < 4/len(string):
            mutated_string.append((bit + 1) % 2)
            mutated = 1
        else:
            mutated_string.append(bit)
    
    return mutated_string, mutated


def mutation_step(string, atom_types, n):
    # keeps the score whether there was a mutation or not
    atom_mutated = 0
    
    
    for i, atom in enumerate(atom_types):
        random_val = random.uniform(0,1)
        if random_val < 1/n:
            # if an atom is mutated, counter goes to one
            atom_mutated = 1
            
            available_mols = list(range(4))
            available_mols.remove(atom)

            atom_types[i] = random.choice(available_mols)

    mutated_string, string_mutated = mutate_string(string)
    A = build_matrix_from_mat_string(mutated_string, n)
    # A = build_matrix_from_mat_string(string, n)

    if string_mutated == 1 or atom_mutated == 1:
        A_mutated, atom_types = create_molecule_from_matrix_and_atom_types_3(A, atom_types)
        mutated_string = extract_matrix_string(A_mutated)
        int_string = [int(x) for x in list(mutated_string)]
    else:
        int_string = string
        

    return int_string, atom_types



def run_GA(n, num_iterations, num_elites, pop_size, time_lim, known_best_val):
    # initialisation
    start = time.time()
    
    initial_pop = generate_graphs(pop_size, n)
    
    length_chromosome = ((n-1)*n)/2

    # this is of the form [string, atom_types]
    pop = initial_pop
    
    best_mol = [None, None, 0]
    
    saved_data = []
    saved_data.append(['STARTED', 'STARTED', 'STARTED'])

    # for \tau iterations
    for i in range(num_iterations):

        # fitness
        current_pop = [[pop[i][0], pop[i][1], fitness_func(pop[i][0], pop[i][1])] for i in range(len(pop))]

        # selection
        selection, elites, pairs, pop_highest_val = selection3(current_pop, num_elites, pop_size)
        print(i)
        print(list(pop_highest_val.iloc[0])[2])
        
        

        if list(pop_highest_val.iloc[0])[2] > best_mol[2]:
            best_mol = list(pop_highest_val.iloc[0])
            saved_data.append([i] + best_mol + ['improved!'])
        else:
            saved_data.append([i] + best_mol)
            

        if known_best_val is not None:
            if np.round(best_mol[2], 6) >= np.round(known_best_val, 6):
                stopping_time = time.time() - start
                print(f'{stopping_time=}')
                saved_data.append([i] + best_mol + [stopping_time])
                break


        # crossover & mutation
        offspring = []
        for pair in pairs:
            # crossover
            cross_position = random.randint(0, length_chromosome - 1)
            chl_1_str, chl_1_at, chl_2_str, chl_2_at = cross_over(selection[pair[0]], 
                                                                  selection[pair[1]], 
                                                                  n, cross_position)

            # mutation
            offspring_1 = mutation_step(chl_1_str, chl_1_at, n)
            offspring_2 = mutation_step(chl_2_str, chl_2_at, n)

            offspring.append([np.array(offspring_1[0]), list(offspring_1[1])])
            offspring.append([np.array(offspring_2[0]), list(offspring_2[1])])
        
        pop = elites + offspring
        
        now = time.time()

        if now - start > time_lim:
            break
        #output: 100 graphs

    # after \tau iterations, return best graph


    winner = best_mol
    winner_string = winner[0]
    winner_atom_types = winner[1]
    winner_obj_val = winner[2]

    winner_adjacency = build_matrix_from_mat_string(winner_string, n)
    winner_data = prep_gene_for_nn(winner_adjacency, winner_atom_types)
    winner_feat_vecs = winner_data.x

    file_name = f'{gnn}_{num_nodes}_{num_layers}_mol_len_{n}.csv'
    with open(file_name, 'a+') as f:
        write = csv.writer(f)
        write.writerows(saved_data)

    return winner_adjacency, winner_feat_vecs, winner_obj_val


def main():    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pop = run_GA(n, num_iterations = 100000000000, num_elites = 0, pop_size = 10, time_lim = 36000, known_best_val = best_known_val)
        print(final_pop)

if __name__ == '__main__':
    main()
