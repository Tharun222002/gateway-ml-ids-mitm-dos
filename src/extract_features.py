#!/usr/bin/env python3

"""
extract_features.py

Extract window-based network behaviour features from PCAP/PCAPNG files
for a Machine Learning-based IDS.

Classes currently planned:
    NORMAL
    RECON
    MITM
    ARP_DOS

Each non-empty time window becomes one row in the CSV dataset.

Important:
    - The PCAP itself is NOT given directly to the ML model.
    - Network behaviour is converted into numerical features.
    - Each capture has one scenario label.
    - Source filename is retained so train/test splitting can later
      be performed at capture level rather than randomly by row.

Fix applied (vs earlier draft):
    - new_ip_mac_mapping_count now counts DISTINCT new (ip, mac) pairs
      per window, not once per packet. Previously a single new binding
      that appeared in 300 packets within one window was counted 300
      times, making the feature collinear with raw packet/ARP-reply
      volume instead of measuring distinctness of new bindings.

Usage:

    python3 extract_features.py \
        --pcap normal.pcapng \
        --label NORMAL \
        --window 5 \
        --out dataset.csv

    python3 extract_features.py \
        --pcap recon.pcapng \
        --label RECON \
        --window 5 \
        --out dataset.csv

    python3 extract_features.py \
        --pcap mitmattack.pcapng \
        --label MITM \
        --window 5 \
        --out dataset.csv

    python3 extract_features.py \
        --pcap ban.pcapng \
        --label ARP_DOS \
        --window 5 \
        --out dataset.csv

Requires:
    pip install scapy
"""

import argparse
import csv
import os
import statistics
import ipaddress
from collections import defaultdict, Counter

from scapy.all import PcapReader
from scapy.layers.l2 import ARP, Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP


BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"


def is_local_ip(ip_str, local_network=None):
    """
    True only for IPs that are actually ARP-reachable on the local subnet.

    IMPORTANT: on large institutional/campus networks, huge swathes of the
    address space (e.g. a whole 10.0.0.0/8) are RFC1918 "private" even
    though most of those hosts are many routed hops away and are NOT
    ARP-neighbors -- a generic is_private() check does nothing useful there,
    since nearly everything passes it. If --local-subnet is given, membership
    in that specific CIDR is checked instead, which is what actually
    determines ARP-reachability. Without it, falls back to the (weaker)
    generic RFC1918 check.
    """

    try:

        ip_obj = ipaddress.ip_address(ip_str)

    except ValueError:

        return False

    if local_network is not None:

        return ip_obj in local_network

    return ip_obj.is_private


FIELDNAMES = [

    # Dataset identification
    "source_file",
    "label",
    "window_start",
    "window_index",

    # General traffic
    "total_packets",
    "packet_rate",
    "avg_packet_size",
    "std_packet_size",

    # ARP
    "arp_requests",
    "arp_replies",
    "arp_request_rate",
    "arp_reply_rate",
    "arp_reply_request_ratio",
    "unique_arp_sources",
    "unique_arp_targets",
    "gratuitous_arp_count",

    # ARP/IP-MAC behaviour
    "unique_src_ip",
    "unique_dst_ip",
    "unique_src_mac",
    "unique_dst_mac",
    "duplicate_ip_mac_count",
    "new_ip_mac_mapping_count",

    # Broadcast
    "broadcast_count",
    "broadcast_rate",
    "broadcast_ratio",

    # Protocols
    "tcp_count",
    "udp_count",
    "icmp_count",
    "other_count",

    # Protocol rates
    "tcp_rate",
    "udp_rate",
    "icmp_rate",

    # TCP behaviour
    "tcp_syn_count",
    "tcp_syn_rate",
    "tcp_syn_ratio",
    "tcp_synack_count",
    "tcp_half_open_ratio",

    # ICMP behaviour
    "icmp_echo_request_count",
    "icmp_echo_reply_count",

    # Scan / fan-out behaviour (port & host sweep signatures)
    "unique_dst_ports",
    "max_dst_ports_per_src",
    "max_dst_ips_per_src",

]


class WindowStats:
    """
    Stores statistics for one time window.
    """

    def __init__(self):

        self.total_packets = 0

        self.packet_sizes = []

        # ARP
        self.arp_requests = 0
        self.arp_replies = 0
        self.arp_sources = set()
        self.arp_targets = set()
        self.gratuitous_arp_count = 0

        # IP / MAC
        self.src_ips = set()
        self.dst_ips = set()

        self.src_macs = set()
        self.dst_macs = set()

        # IP -> MAC mappings observed in this window
        self.ip_to_macs = defaultdict(set)

        # Number of DISTINCT new (ip, mac) mappings compared with
        # previous windows -- deduplicated within this window (see fix note above)
        self.new_ip_mac_mapping_count = 0
        self._new_pairs_seen_this_window = set()

        # Broadcast
        self.broadcast_count = 0

        # Protocols
        self.tcp_count = 0
        self.udp_count = 0
        self.icmp_count = 0
        self.other_count = 0

        # TCP
        self.tcp_syn_count = 0
        self.tcp_synack_count = 0

        # ICMP
        self.icmp_echo_request_count = 0
        self.icmp_echo_reply_count = 0

        # Scan / fan-out tracking
        self.dst_ports = set()
        self.src_to_dst_ports = defaultdict(set)
        self.src_to_dst_ips = defaultdict(set)

    def _note_mapping(self, ip, mac, historical_ip_mac):
        """
        Record an observed (ip, mac) binding. Increments
        new_ip_mac_mapping_count at most ONCE per distinct (ip, mac)
        pair per window, regardless of how many packets carry it.
        """

        pair = (ip, mac)

        if (
            mac not in historical_ip_mac.get(ip, set())
            and pair not in self._new_pairs_seen_this_window
        ):

            self.new_ip_mac_mapping_count += 1
            self._new_pairs_seen_this_window.add(pair)

    def add_packet(self, pkt, historical_ip_mac, local_network=None):

        self.total_packets += 1

        self.packet_sizes.append(len(pkt))

        # --------------------------------------------------
        # Ethernet
        # --------------------------------------------------

        if Ether in pkt:

            src_mac = pkt[Ether].src.lower()
            dst_mac = pkt[Ether].dst.lower()

            self.src_macs.add(src_mac)
            self.dst_macs.add(dst_mac)

            if dst_mac == BROADCAST_MAC:
                self.broadcast_count += 1

        # --------------------------------------------------
        # ARP
        # --------------------------------------------------

        if ARP in pkt:

            arp = pkt[ARP]

            sender_ip = arp.psrc
            target_ip = arp.pdst

            sender_mac = arp.hwsrc.lower() if arp.hwsrc else None

            if arp.op == 1:

                self.arp_requests += 1

            elif arp.op == 2:

                self.arp_replies += 1

            if sender_ip:
                self.arp_sources.add(sender_ip)

            if target_ip:
                self.arp_targets.add(target_ip)

            # ----------------------------------------------
            # Detect gratuitous ARP
            # ----------------------------------------------

            if sender_ip and target_ip:

                if sender_ip == target_ip:

                    self.gratuitous_arp_count += 1

            # ----------------------------------------------
            # Track IP -> MAC mapping (deduplicated per window)
            # ----------------------------------------------

            if sender_ip and sender_mac:

                if is_local_ip(sender_ip, local_network):

                    self._note_mapping(sender_ip, sender_mac, historical_ip_mac)

                    self.ip_to_macs[sender_ip].add(sender_mac)

        # --------------------------------------------------
        # IP
        # --------------------------------------------------

        if IP in pkt:

            ip = pkt[IP]

            self.src_ips.add(ip.src)
            self.dst_ips.add(ip.dst)

            # Fan-out: how many distinct destination IPs has THIS source
            # contacted this window -- a host-sweep signature.
            self.src_to_dst_ips[ip.src].add(ip.dst)

            # ----------------------------------------------
            # Track IP -> Ethernet MAC mapping (deduplicated per window)
            # ----------------------------------------------

            if Ether in pkt:

                src_mac = pkt[Ether].src.lower()

                if is_local_ip(ip.src, local_network):

                    self._note_mapping(ip.src, src_mac, historical_ip_mac)

                    self.ip_to_macs[ip.src].add(src_mac)

            # ----------------------------------------------
            # TCP
            # ----------------------------------------------

            if TCP in pkt:

                self.tcp_count += 1

                dport = int(pkt[TCP].dport)
                self.dst_ports.add(dport)
                self.src_to_dst_ports[ip.src].add(dport)

                flags = int(pkt[TCP].flags)

                syn = bool(flags & 0x02)
                ack = bool(flags & 0x10)

                if syn and not ack:

                    self.tcp_syn_count += 1

                elif syn and ack:

                    self.tcp_synack_count += 1

            # ----------------------------------------------
            # UDP
            # ----------------------------------------------

            elif UDP in pkt:

                self.udp_count += 1

                dport = int(pkt[UDP].dport)
                self.dst_ports.add(dport)
                self.src_to_dst_ports[ip.src].add(dport)

            # ----------------------------------------------
            # ICMP
            # ----------------------------------------------

            elif ICMP in pkt:

                self.icmp_count += 1

                icmp_type = int(pkt[ICMP].type)

                # Echo request
                if icmp_type == 8:

                    self.icmp_echo_request_count += 1

                # Echo reply
                elif icmp_type == 0:

                    self.icmp_echo_reply_count += 1

            else:

                self.other_count += 1

        else:

            if ARP not in pkt:

                self.other_count += 1

    def duplicate_ip_mac_count(self):

        """
        Counts IP addresses associated with multiple MAC addresses
        inside this time window.
        """

        return sum(
            1
            for macs in self.ip_to_macs.values()
            if len(macs) > 1
        )

    def to_row(
        self,
        source_file,
        label,
        window_start,
        window_index,
        window_size
    ):

        n = max(self.total_packets, 1)

        # --------------------------------------------------
        # Packet size
        # --------------------------------------------------

        if self.packet_sizes:

            avg_size = statistics.mean(self.packet_sizes)

            if len(self.packet_sizes) > 1:

                std_size = statistics.pstdev(self.packet_sizes)

            else:

                std_size = 0.0

        else:

            avg_size = 0.0
            std_size = 0.0

        # --------------------------------------------------
        # Rates
        # --------------------------------------------------

        packet_rate = self.total_packets / window_size

        arp_request_rate = self.arp_requests / window_size

        arp_reply_rate = self.arp_replies / window_size

        broadcast_rate = self.broadcast_count / window_size

        tcp_rate = self.tcp_count / window_size

        udp_rate = self.udp_count / window_size

        icmp_rate = self.icmp_count / window_size

        tcp_syn_rate = self.tcp_syn_count / window_size

        # --------------------------------------------------
        # Ratios
        # --------------------------------------------------

        arp_reply_request_ratio = (
            self.arp_replies / self.arp_requests
            if self.arp_requests > 0
            else 0.0
        )

        broadcast_ratio = (
            self.broadcast_count / n
        )

        tcp_syn_ratio = (
            self.tcp_syn_count / self.tcp_count
            if self.tcp_count > 0
            else 0.0
        )

        # Fraction of SYNs that never got a SYN-ACK back this window.
        # 1.0 = every SYN went unanswered (strong scan/flood signature).
        # 0.0 = every SYN was answered (typical of normal connections).
        total_syn_attempts = self.tcp_syn_count + self.tcp_synack_count
        tcp_half_open_ratio = (
            self.tcp_syn_count / total_syn_attempts
            if total_syn_attempts > 0
            else 0.0
        )

        max_dst_ports_per_src = (
            max((len(p) for p in self.src_to_dst_ports.values()), default=0)
        )

        max_dst_ips_per_src = (
            max((len(p) for p in self.src_to_dst_ips.values()), default=0)
        )

        return {

            "source_file": source_file,
            "label": label,
            "window_start": round(window_start, 3),
            "window_index": window_index,

            "total_packets": self.total_packets,
            "packet_rate": round(packet_rate, 3),
            "avg_packet_size": round(avg_size, 3),
            "std_packet_size": round(std_size, 3),

            "arp_requests": self.arp_requests,
            "arp_replies": self.arp_replies,

            "arp_request_rate":
                round(arp_request_rate, 3),

            "arp_reply_rate":
                round(arp_reply_rate, 3),

            "arp_reply_request_ratio":
                round(arp_reply_request_ratio, 3),

            "unique_arp_sources":
                len(self.arp_sources),

            "unique_arp_targets":
                len(self.arp_targets),

            "gratuitous_arp_count":
                self.gratuitous_arp_count,

            "unique_src_ip":
                len(self.src_ips),

            "unique_dst_ip":
                len(self.dst_ips),

            "unique_src_mac":
                len(self.src_macs),

            "unique_dst_mac":
                len(self.dst_macs),

            "duplicate_ip_mac_count":
                self.duplicate_ip_mac_count(),

            "new_ip_mac_mapping_count":
                self.new_ip_mac_mapping_count,

            "broadcast_count":
                self.broadcast_count,

            "broadcast_rate":
                round(broadcast_rate, 3),

            "broadcast_ratio":
                round(broadcast_ratio, 3),

            "tcp_count":
                self.tcp_count,

            "udp_count":
                self.udp_count,

            "icmp_count":
                self.icmp_count,

            "other_count":
                self.other_count,

            "tcp_rate":
                round(tcp_rate, 3),

            "udp_rate":
                round(udp_rate, 3),

            "icmp_rate":
                round(icmp_rate, 3),

            "tcp_syn_count":
                self.tcp_syn_count,

            "tcp_syn_rate":
                round(tcp_syn_rate, 3),

            "tcp_syn_ratio":
                round(tcp_syn_ratio, 3),

            "tcp_synack_count":
                self.tcp_synack_count,

            "tcp_half_open_ratio":
                round(tcp_half_open_ratio, 3),

            "icmp_echo_request_count":
                self.icmp_echo_request_count,

            "icmp_echo_reply_count":
                self.icmp_echo_reply_count,

            "unique_dst_ports":
                len(self.dst_ports),

            "max_dst_ports_per_src":
                max_dst_ports_per_src,

            "max_dst_ips_per_src":
                max_dst_ips_per_src,
        }


def extract(pcap_path, label, window_size, out_path=None, skip_seconds=0.0, local_network=None, quiet=False):
    """
    Extract windowed features from a PCAP.

    If out_path is given, rows are appended to that CSV (original
    offline-training behaviour). If out_path is None, no file is written --
    the rows are just returned in memory. This lets a live/real-time caller
    (e.g. live_ids.py) get feature rows for one rotated capture WITHOUT any
    CSV round-trip, and without paying Scapy's import cost more than once
    per process (call this function directly instead of shelling out to
    this script as a subprocess).

    Returns: list of row dicts (same shape as the CSV rows).
    """

    rows = []

    current = WindowStats()

    window_index = 0

    window_start_time = None

    first_ts = None

    cutoff_ts = None  # timestamp before which packets are discarded entirely

    # ------------------------------------------------------
    # Historical IP -> MAC information
    #
    # This persists between windows.
    # ------------------------------------------------------

    historical_ip_mac = defaultdict(set)

    packet_count = 0
    skipped_count = 0

    with PcapReader(pcap_path) as reader:

        for pkt in reader:

            try:

                ts = float(pkt.time)

            except Exception:

                continue

            # --------------------------------------------------
            # First packet -- establish absolute start time and,
            # if requested, the cutoff before which packets are
            # discarded (idle/setup time before the real scenario
            # actually started).
            # --------------------------------------------------

            if first_ts is None:

                first_ts = ts
                cutoff_ts = first_ts + skip_seconds

            if ts < cutoff_ts:

                skipped_count += 1
                continue

            if window_start_time is None:

                # First packet AFTER the skip cutoff starts window 0.
                window_start_time = ts

            packet_count += 1

            # --------------------------------------------------
            # Move to next time window
            # --------------------------------------------------

            while ts >= window_start_time + window_size:

                if current.total_packets > 0:

                    rows.append(
                        current.to_row(
                            os.path.basename(pcap_path),
                            label,
                            window_start_time - (first_ts + skip_seconds),
                            window_index,
                            window_size
                        )
                    )

                    # ------------------------------------------
                    # Preserve mappings across windows
                    # ------------------------------------------

                    for ip, macs in current.ip_to_macs.items():

                        historical_ip_mac[ip].update(macs)

                window_index += 1

                window_start_time += window_size

                current = WindowStats()

            # --------------------------------------------------
            # Add packet
            # --------------------------------------------------

            current.add_packet(
                pkt,
                historical_ip_mac,
                local_network
            )

    # ------------------------------------------------------
    # Final window
    # ------------------------------------------------------

    if current.total_packets > 0:

        rows.append(
            current.to_row(
                os.path.basename(pcap_path),
                label,
                window_start_time - (first_ts + skip_seconds),
                window_index,
                window_size
            )
        )

    # ------------------------------------------------------
    # Write CSV (only if a path was given -- see docstring)
    # ------------------------------------------------------

    if out_path is not None:

        file_exists = os.path.isfile(out_path)

        with open(
            out_path,
            "a",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=FIELDNAMES
            )

            if not file_exists:

                writer.writeheader()

            writer.writerows(rows)

    if not quiet:

        print()
        print("=" * 70)

        print(f"Capture       : {pcap_path}")
        print(f"Label         : {label}")
        print(f"Skip seconds  : {skip_seconds}")
        print(f"Packets skip  : {skipped_count}")
        print(f"Packets read  : {packet_count}")
        print(f"Windows       : {len(rows)}")
        print(f"Window size   : {window_size} seconds")
        print(f"Output        : {out_path if out_path else '(in-memory, no file written)'}")

        print("=" * 70)

    return rows


def main():

    parser = argparse.ArgumentParser(
        description="Extract ML features from PCAP/PCAPNG files."
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Input PCAP or PCAPNG file"
    )

    parser.add_argument(
        "--label",
        required=True,
        choices=[
            "NORMAL",
            "RECON",
            "MITM",
            "ARP_DOS"
        ],
        help="Scenario label"
    )

    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Window size in seconds (default: 5)"
    )

    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=0.0,
        help=(
            "Discard this many seconds from the START of the capture before "
            "extracting windows. Use this to trim idle/setup time before the "
            "actual attack or scan tool was launched, so that lead-in traffic "
            "isn't mislabeled as the attack class (default: 0, keep everything)."
        )
    )

    parser.add_argument(
        "--local-subnet",
        default=None,
        help=(
            "CIDR of YOUR local subnet, e.g. 10.50.108.0/22 (the range your "
            "test devices/VMs are actually on). Restricts IP-to-MAC mapping "
            "tracking (duplicate_ip_mac_count, new_ip_mac_mapping_count) to "
            "addresses that are genuinely ARP-reachable neighbors. Strongly "
            "recommended on large institutional networks, where a generic "
            "private-vs-public IP check is not enough -- huge swathes of "
            "campus address space are RFC1918 private despite being many "
            "routed hops away, not real ARP-neighbors. Without this, falls "
            "back to a weaker generic private-IP check."
        )
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    if args.window <= 0:

        raise ValueError(
            "Window size must be greater than zero."
        )

    if args.skip_seconds < 0:

        raise ValueError(
            "skip-seconds must be zero or greater."
        )

    local_network = None

    if args.local_subnet:

        try:

            local_network = ipaddress.ip_network(args.local_subnet, strict=False)

        except ValueError as e:

            raise ValueError(
                f"Invalid --local-subnet {args.local_subnet!r}: {e}"
            )

    if not os.path.isfile(args.pcap):

        raise FileNotFoundError(
            f"PCAP file not found: {args.pcap}"
        )

    extract(
        args.pcap,
        args.label,
        args.window,
        args.out,
        skip_seconds=args.skip_seconds,
        local_network=local_network
    )


if __name__ == "__main__":

    main()
