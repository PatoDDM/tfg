import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/drngb/Desktop/tfg/dataset/dataset_attack.csv', sep=',')

attack_df = df[df['ataque'] == 'attack']

tcp_attacks = attack_df[attack_df['frame.protocols'].str.contains('tcp')].shape[0]
udp_attacks = attack_df[attack_df['frame.protocols'].str.contains('udp')].shape[0]
attack_counts = {'TCP Normal': tcp_attacks, 'UDP Normal': udp_attacks}

plt.bar(attack_counts.keys(), attack_counts.values(), color=['blue', 'orange'])
plt.xlabel('Tipo de Paquete')
plt.ylabel('Cantidad')
plt.title('Cantidad de Ataques TCP y UDP')

plt.show()
