
import dpkt
import pandas as pd
import numpy as np

from pathlib import Path
from collections import Counter


# Trained model path (kept unchanged)
MODEL_PATH = Path(
    "data/supervised/file_based_mitm60k/results/"
    "hybrid_dt_lr_mitm60k_no_https.joblib"
)

# Exact feature order expected by the trained model
FEATURE_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "DNS",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "IAT",
    "Number",
    "Variance",
]


def get_packet_info(buf):
    """Extract packet information from an Ethernet frame."""

    try:
        ethernet = dpkt.ethernet.Ethernet(buf)
        ip = ethernet.data

        # ARP packet
        if isinstance(ip, dpkt.arp.ARP):
            return {
                "protocol": 0,
                "ttl": 0,
                "header_length": 0,
                "packet_size": len(buf),
                "src_port": 0,
                "dst_port": 0,
                "tcp_flags": 0,
                "is_ipv4": 0,
                "is_ipv6": 0,
                "is_arp": 1,
            }

        # IPv4 packet
        if isinstance(ip, dpkt.ip.IP):
            protocol = ip.p
            ttl = ip.ttl
            is_ipv4 = 1
            is_ipv6 = 0

        # IPv6 packet
        elif isinstance(ip, dpkt.ip6.IP6):
            protocol = ip.nxt
            ttl = ip.hlim
            is_ipv4 = 0
            is_ipv6 = 1

        else:
            return None

        src_port = 0
        dst_port = 0
        tcp_flags = 0
        header_length = 0

        # TCP header length
        if isinstance(ip.data, dpkt.tcp.TCP):
            tcp = ip.data

            src_port = tcp.sport
            dst_port = tcp.dport
            tcp_flags = tcp.flags

            header_length = tcp.off * 4

        # UDP header length
        elif isinstance(ip.data, dpkt.udp.UDP):
            udp = ip.data

            src_port = udp.sport
            dst_port = udp.dport

            header_length = 8

        return {
            "protocol": protocol,
            "ttl": ttl,
            "header_length": header_length,
            "packet_size": len(buf),
            "src_port": src_port,
            "dst_port": dst_port,
            "tcp_flags": tcp_flags,
            "is_ipv4": is_ipv4,
            "is_ipv6": is_ipv6,
            "is_arp": 0,
        }

    except (dpkt.UnpackError, AttributeError, ValueError):
        return None


def extract_protocol_features(packet):
    """Extract protocol indicators and TCP flags."""

    src_port = packet["src_port"]
    dst_port = packet["dst_port"]
    protocol = packet["protocol"]
    flags = packet["tcp_flags"]

    return {
        "fin_flag_number": int(bool(flags & dpkt.tcp.TH_FIN)),
        "syn_flag_number": int(bool(flags & dpkt.tcp.TH_SYN)),
        "rst_flag_number": int(bool(flags & dpkt.tcp.TH_RST)),
        "psh_flag_number": int(bool(flags & dpkt.tcp.TH_PUSH)),
        "ack_flag_number": int(bool(flags & dpkt.tcp.TH_ACK)),
        "ece_flag_number": int(bool(flags & dpkt.tcp.TH_ECE)),
        "cwr_flag_number": int(bool(flags & dpkt.tcp.TH_CWR)),

        "TCP": int(protocol == 6),
        "UDP": int(protocol == 17),

        "DHCP": int(
            src_port in (67, 68) or dst_port in (67, 68)
        ),

        "HTTP": int(src_port == 80 or dst_port == 80),
        "DNS": int(src_port == 53 or dst_port == 53),
        "SSH": int(src_port == 22 or dst_port == 22),
        "IRC": int(src_port == 21 or dst_port == 21),

        "ARP": packet["is_arp"],
        "ICMP": int(protocol == 1),
        "IGMP": int(protocol == 2),
        "IPv": packet["is_ipv4"],
    }


def calculate_window_features(packets, timestamps, iats):
    """Calculate statistical features for one packet window."""

    if not packets:
        return {}

    sizes = np.array(
        [packet["packet_size"] for packet in packets],
        dtype=float
    )

    duration = max(timestamps) - min(timestamps)

    rate = len(sizes) / duration if duration > 0 else 0

    return {
        "Tot sum": float(np.sum(sizes)),
        "Min": float(np.min(sizes)),
        "Max": float(np.max(sizes)),
        "AVG": float(np.mean(sizes)),
        "Std": float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0,
        "Rate": float(rate),
        "Number": int(len(sizes)),
        "Variance": float(np.var(sizes, ddof=1)) if len(sizes) > 1 else 0,
        "IAT": float(np.mean(iats)) if iats else 0,
    }


def build_feature_row(packets, timestamps, iats):
    """Build one row containing the 34 model features."""

    if not packets:
        return None

    protocol_features = [
        extract_protocol_features(packet)
        for packet in packets
    ]

    window = calculate_window_features(
        packets,
        timestamps,
        iats
    )

    protocols = [packet["protocol"] for packet in packets]
    protocol_type = Counter(protocols).most_common(1)[0][0]

    row = {
        "Header_Length": float(
            np.mean([packet["header_length"] for packet in packets])
        ),

        "Protocol Type": int(protocol_type),

        "Time_To_Live": float(
            np.mean([packet["ttl"] for packet in packets])
        ),

        **window,
    }

    mean_features = [
        "fin_flag_number",
        "syn_flag_number",
        "rst_flag_number",
        "psh_flag_number",
        "ack_flag_number",
        "ece_flag_number",
        "cwr_flag_number",
        "HTTP",
        "DNS",
        "SSH",
        "IRC",
        "TCP",
        "UDP",
        "DHCP",
        "ARP",
        "ICMP",
        "IGMP",
        "IPv",
    ]

    for name in mean_features:
        row[name] = float(
            np.mean([packet[name] for packet in protocol_features])
        )

    # TCP flag counts
    row["ack_count"] = int(
        sum(packet["ack_flag_number"] for packet in protocol_features)
    )

    row["syn_count"] = int(
        sum(packet["syn_flag_number"] for packet in protocol_features)
    )

    row["fin_count"] = int(
        sum(packet["fin_flag_number"] for packet in protocol_features)
    )

    row["rst_count"] = int(
        sum(packet["rst_flag_number"] for packet in protocol_features)
    )

    return pd.DataFrame(
        [[row[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )


def extract_pcap_windows(pcap_path, window_size=10):
    """Read PCAPNG traffic and extract feature windows."""

    rows = []
    packets = []
    timestamps = []
    iats = []

    # Keep the previous timestamp across window boundaries
    last_packet_time = None

    with open(pcap_path, "rb") as capture_file:
        capture = dpkt.pcapng.Reader(capture_file)

        for timestamp, buf in capture:

            packet = get_packet_info(buf)

            if packet is None:
                continue

            # Calculate packet inter-arrival time
            if last_packet_time is None:
                packet_iat = 0.0
            else:
                packet_iat = timestamp - last_packet_time

            last_packet_time = timestamp

            packets.append(packet)
            timestamps.append(timestamp)
            iats.append(packet_iat)

            if len(packets) == window_size:

                row = build_feature_row(
                    packets,
                    timestamps,
                    iats
                )

                if row is not None:
                    rows.append(row)

                packets = []
                timestamps = []
                iats = []

    # Process final incomplete window
    if packets:

        row = build_feature_row(
            packets,
            timestamps,
            iats
        )

        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":

    capture_path = Path("captures/iot_mqtt.pcapng")

    if not capture_path.exists():

        print("Capture file not found:", capture_path)

    else:

        features = extract_pcap_windows(capture_path)

        print("CAPTURE:", capture_path)
        print("WINDOWS EXTRACTED:", len(features))
        print("FEATURE COUNT:", len(features.columns))

        print("\nEXTRACTED FEATURES:")
        print(features.head().to_string(index=False))

        print("\nFEATURE COLUMN ORDER:")
        print(list(features.columns))

        print("\nFEATURE VALIDATION:")

        print(
            "Correct feature count:",
            len(features.columns) == 34
        )

        print(
            "Correct feature order:",
            list(features.columns) == FEATURE_COLUMNS
        )

        print(
            "Missing values:",
            int(features.isnull().sum().sum())
        )
