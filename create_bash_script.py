import pandas as pd

df_GCN = pd.read_csv('GCN_results_cleaned_10_hours.csv')
df_GS = pd.read_csv('GraphSAGE_results_cleaned_10_hours.csv')

def create_bash_script():
    print('SECONDS=0')
    print('echo \"started!\"')
    print(f'now=$(date +"%T")')
    print(f'echo \"Started at : $now\"')

    for df in [df_GS, df_GCN]:
        for i in range(21):
            data = list(df.iloc[i])
            print(f'python final_GAv4CleanLogged.py --gnn \'{str(data[1])}\' --location \'{str(data[0])}\' --mol_len {int(data[4])} --num_layers {int(data[2])} --num_nodes {int(data[3])} --best_known_val {data[7]}')
            # print(f'echo \"progress: {(j)*len(location_list) + i + 1}/{(len(location_list))*(len(mol_len_list))} after: $SECONDS s\"')
            print(f'SECONDS=0')
            print(f'now=$(date +"%T")')
            print(f'echo \"Started at : $now\"')

create_bash_script()

