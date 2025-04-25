from scapy.all import sniff, conf, ARP, IP
import subprocess

# Approved device MAC addresses
approved_devices = {
    "00:11:22:33:44:55",
    "AA:BB:CC:DD:EE:FF",
}

def block_device(mac):
    rule_name = f"Block_{mac}"
    print(f"Blocking device with MAC: {mac}")
    # Example: Replace <device_ip> with the actual IP if available.
    # subprocess.run(["netsh", "advfirewall", "firewall", "add", "rule",
    #                 f"name={rule_name}", "dir=in", "action=block", "remoteip=<device_ip>"], check=True)

def handle_packet(packet):
    if packet.haslayer(ARP):
        mac = packet[ARP].hwsrc
        if mac not in approved_devices:
            print(f"Unauthorized ARP device detected: {mac}")
            block_device(mac)
    elif packet.haslayer(IP):
        print(f"IP packet from: {packet[IP].src}")

if __name__ == '__main__':
    print("Starting packet sniffing...")
    try:
        # Attempt layer 2 sniffing (requires npcap/WinPcap)
        sniff(prn=handle_packet, store=False)
    except RuntimeError:
        # Fallback to layer 3 sniffing (won't capture ARP)
        print("Layer 2 sniffing unavailable; switching to layer 3...")
        sniff(socket=conf.L3socket, prn=handle_packet, store=False)