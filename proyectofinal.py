from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
from keras.models import load_model
import numpy as np
import os
from flask import Flask, render_template_string

os.environ['PYSHARK_NO_WIRESHARK'] = 'True'

app = Flask(__name__)

model = load_model('modelo_red_neuronal.keras')

last_time = None
packet_data = []
attack_detected = False

def packet_handler(packet):
    global last_time, attack_detected
    time_delta = 0

    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        tcp_layer = packet.getlayer(TCP)
        udp_layer = packet.getlayer(UDP)

        ack = src_port = dst_port = window_size = tcp_len = ip_len = frame_protocols_4_ftp_data = ip_ttl = frame_protocols_4_http = frame_protocols_4_data = push_flag = fin_flag = syn_flag = normal_attack_flag = frame_protocols_3_udp = 0

        if tcp_layer is not None:
            ack = tcp_layer.ack if hasattr(tcp_layer, 'ack') else 0
            src_port = tcp_layer.sport if hasattr(tcp_layer, 'sport') else 0
            dst_port = tcp_layer.dport if hasattr(tcp_layer, 'dport') else 0
            window_size = tcp_layer.window if hasattr(tcp_layer, 'window') else 0
            tcp_len = len(tcp_layer.payload) if hasattr(tcp_layer, 'payload') else 0
            push_flag = tcp_layer.flags.PSH if hasattr(tcp_layer, 'flags') and hasattr(tcp_layer.flags, 'PSH') else 0
            fin_flag = tcp_layer.flags.F if hasattr(tcp_layer, 'flags') and hasattr(tcp_layer.flags, 'F') else 0
            syn_flag = tcp_layer.flags.S if hasattr(tcp_layer, 'flags') and hasattr(tcp_layer.flags, 'S') else 0

            current_time = packet.time
            time_delta = current_time - last_time if last_time else 0
            last_time = current_time

        if udp_layer is not None:
            frame_protocols_3_udp = 1

        ip_len = packet[IP].len if hasattr(packet[IP], 'len') else 0
        frame_protocols_4_ftp_data = 1 if 'FTP' in packet else 0
        ip_ttl = packet[IP].ttl if hasattr(packet[IP], 'ttl') else 0
        frame_protocols_4_http = 1 if 'HTTP' in packet else 0
        frame_protocols_4_data = 1 if 'DATA' in packet else 0
        normal_attack_flag = 1 if 'Attack' in packet else 0

        protocol = packet[IP].proto

        packet_data.append([ack, time_delta, src_port, dst_port, window_size, tcp_len, ip_len,
                            frame_protocols_4_ftp_data, ip_ttl, frame_protocols_4_http,
                            frame_protocols_4_data, push_flag, frame_protocols_3_udp, fin_flag, syn_flag,
                            src_ip, dst_ip, protocol])

        df = pd.DataFrame(packet_data, columns=['tcp.ack', 'tcp.time_delta', 'tcp.srcport', 'tcp.dstport', 'tcp.window_size',
                                                'tcp.len', 'ip.len', 'frame.protocols_4_ftp-data', 'ip.ttl',
                                                'frame.protocols_4_http', 'frame.protocols_4_data', 'tcp.flags.push',
                                                'frame.protocols_3_udp', 'tcp.flags.fin', 'tcp.flags.syn',
                                                'ip.src','ip.dst', 'protocol'])
        df.drop(columns=['ip.src', 'ip.dst'], inplace=True)
        df = df.astype(float)

        if len(packet_data) >= 1500:
            df_last_100 = df[-1500:]
            y_pred_proba = model.predict(df_last_100)
            proba = np.mean(y_pred_proba)

            if proba > 0.9:
                attack_detected = True
                print("¡Ataque DDoS detectado!")

@app.route('/')
def index():
    global attack_detected
    return render_template_string('''
        <!doctype html>
        <html>
        <head>
            <title>Detección de DDoS</title>
        </head>
        <body>
            <h1>Estado del Sistema</h1>
            {% if attack_detected %}
                <p style="color: red;">¡Ataque DDoS detectado!</p>
            {% else %}
                <p style="color: green;">No se han detectado ataques.</p>
            {% endif %}
        </body>
        </html>
    ''', attack_detected=attack_detected)

def start_sniffing():
    sniff(prn=packet_handler, store=False)

if __name__ == '__main__':
    import threading
    sniff_thread = threading.Thread(target=start_sniffing)
    sniff_thread.start()
    app.run(port=5000)
