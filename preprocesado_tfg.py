import pandas as pan
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


conjunto1 = pan.read_csv("C:/Users/drngb/Desktop/tfg/dataset/dataset_attack.csv")
conjunto2 = pan.read_csv("C:/Users/drngb/Desktop/tfg/dataset/dataset_normal.csv")

conjunto1 = conjunto1.iloc[1500000:]
conjunto2 = conjunto2.iloc[1500000:]

conjunto = pan.concat([conjunto1, conjunto2])

conjunto = conjunto.drop('frame.encap_type', axis=1)

ordinal_features = ['tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta','ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'ip.frag_offset', 'ip.ttl', 'ip.proto']

frame_protocol_split = conjunto["frame.protocols"].str.split(":", expand=True)
max_protocol_columns = 5
frame_protocol_split = frame_protocol_split.iloc[:, :max_protocol_columns]

frame_protocol_split.columns = [f"frame.protocols_{i}" for i in range(frame_protocol_split.shape[1])]

conjunto = pan.concat([conjunto, frame_protocol_split], axis=1)


conjunto = pan.get_dummies(conjunto, columns=['frame.protocols_2', 'frame.protocols_3', 'frame.protocols_4'])
conjunto_frame0 = pan.get_dummies(conjunto["frame.protocols_0"], prefix="frame.protocols_0")
conjunto_frame1 = pan.get_dummies(conjunto["frame.protocols_1"], prefix="frame.protocols_1")
conjunto_ataque = pan.get_dummies(conjunto["ataque"], prefix="ataque")
conjunto = pan.concat([conjunto, conjunto_frame0, conjunto_frame1, conjunto_ataque], axis=1)
conjunto = conjunto.drop(columns=['frame.protocols_0', 'frame.protocols_1', 'ataque'])

for feature in ordinal_features:
    conjunto[feature] = conjunto[feature].astype('category')
    conjunto[feature] = conjunto[feature].cat.codes

conjunto = conjunto.drop(columns=["frame.protocols"])

numeric_cols = conjunto.select_dtypes(include=['float64', 'int64']) 
scaler = MinMaxScaler()
conjunto[numeric_cols.columns] = scaler.fit_transform(numeric_cols)


conjunto.to_csv('C:/Users/drngb/Desktop/tfg/prueba/prueba.csv',index=False)

#En este apartado se evalua lo necesario de cada columna

X = conjunto.drop(columns=['ataque_normal', 'ataque_attack', 'ip.src', 'ip.dst'])
y = conjunto['ataque_attack']

sel = SelectKBest(chi2, k=2)
sel.fit(X, y)

scores = sel.scores_

puntuaje = pan.DataFrame({'Columna': X.columns, 'Puntaje': scores})

puntaje_ordenado = puntuaje.sort_values(by='Puntaje', ascending=False)
selected_columns = puntaje_ordenado[puntaje_ordenado['Puntaje'] >= 10000]['Columna'].tolist()

selected_columns += ['ataque_normal', 'ataque_attack']
selected_columns += ['ip.src', 'ip.dst']

conjunto = conjunto[selected_columns]

conjunto.to_csv('C:/Users/drngb/Desktop/tfg/prueba/puntaje_mayor_igual_a_10_con_ataque.csv', index=False)

print(puntaje_ordenado)
puntaje_ordenado.to_csv('C:/Users/drngb/Desktop/tfg/prueba/puntaje.csv',index=False)
conjunto.to_csv('C:/Users/drngb/Desktop/tfg/prueba/puntaje_mayor_igual_a_10000.csv', index=False)
num_columnas = conjunto.shape[1]
print("Número de columnas en el archivo CSV:", num_columnas)
#m=  (RandomForestClassifier(), scoring='accuracy')
#m.fit(X,y)

#m.score(X,y)


