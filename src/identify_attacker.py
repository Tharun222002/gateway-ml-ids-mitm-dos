#!/usr/bin/env python3

"""
identify_attacker.py

Attacker/victim identification for a single flagged (confirmed-attack)
capture window. Called by live_ids.py ONLY when an attack has already been
confirmed by the ML model + temporal detector -- this module does not do
its own classification, it explains WHO is behind a window already
classified as MITM / RECON / ARP_DOS.

Two distinct identification methods, chosen automatically based on what
the window's ARP traffic actually shows:

1. ARP REPLY CONFLICT (used for MITM and ARP_DOS):
   Track which MAC(s) claim each IP via an ARP reply (is-at). When more
   than one MAC claims the same IP, that's the actual mechanism of ARP
   spoofing (impersonating another host's identity) and of bettercap's
   ban module (poisoning a target's cache) -- one of the claims is a lie.

   Resolving WHICH claimant is lying uses three methods, in priority order:

   a) TRUSTED BASELINE (primary, highest confidence): live_ids.py builds
      an IP-to-MAC table continuously from windows classified NORMAL,
      accumulated over the WHOLE live session BEFORE any attack happened,
      and passes it in as trusted_ip_mac. If the impersonated IP already
      has a known-trusted MAC, whichever claimant does NOT match it is
      unambiguously the attacker -- no per-window guessing needed. This is
      the only method with a genuine, positive ground truth to check
      against rather than an inference from the attack window alone.

   b) CROSS-REFERENCE (used when no trusted baseline exists for this IP):
      a MAC's own ordinary (non-ARP) IP traffic in the SAME capture
      reveals its true IP -- if a MAC's own traffic says it is IP X, but
      it separately claimed via ARP reply to BE IP Y, it's lying about Y.
      PARTIAL LIMITATION found in testing: this doesn't work well for
      ROUTERS specifically -- when a router forwards traffic, the IP
      source stays as the ORIGINAL sender's IP, not the router's own IP,
      so a router's own true IP rarely appears in what it forwards. This
      can leave cross-reference inconclusive for gateway impersonation
      even though it works well for host impersonation.

   c) FIRST-SEEN (last-resort fallback, used when neither of the above is
      available): assumes whichever MAC claimed the IP first is
      legitimate. PROVEN UNRELIABLE in testing -- see module history:
      bettercap's ban module was observed to have its forged reply appear
      BEFORE the real gateway's genuine reply within a window, reporting
      the roles backwards. Kept only as an absolute last resort, and the
      result always reports which method was used so a low-confidence
      fallback is never silently indistinguishable from a trusted or
      cross-referenced result.

     - attacker_mac  = the MAC identified as lying (by any method)
     - impersonated_ip = the IP whose identity was faked
     - victim_ip     = the ARP reply's destination (arp.pdst) -- who
                        actually received the poisoned entry

2. ARP REQUEST VOLUME (used for RECON, where there is no impersonation --
   just one host asking about many others):
     - attacker_ip = whichever IP sent the most ARP requests (who-has),
       since a legitimate host asks about a handful of neighbors, not
       dozens.
     - evidence = request count and number of distinct targets probed.
     - Only reported when evidence clears a minimum bar (see
       MIN_RECON_REQUESTS/MIN_RECON_TARGETS below) -- a single stray ARP
       request in an otherwise-quiet window is not meaningful evidence of
       scanning and previously produced misleadingly confident-looking
       "identifications" from noise.

If neither pattern is present in the window (e.g. the model's confidence
was based on non-ARP features), identification reports "insufficient ARP
evidence" rather than guessing.
"""

from collections import Counter, defaultdict

from scapy.all import PcapReader, ARP, IP, Ether


# Minimum evidence required before reporting an ARP-request-volume
# identification (RECON signature) -- see module docstring, method 2.
MIN_RECON_REQUESTS = 5
MIN_RECON_TARGETS = 3


def identify(pcap_paths, attack_class=None, trusted_ip_mac=None):
    """
    Analyse one OR MORE capture files together (see lookback window in
    live_ids.py -- an ongoing attack doesn't necessarily produce fresh ARP
    conflict evidence in every single 5-second slice, so identification on
    a CONFIRMED alert widens its evidence to the last few recent windows
    rather than just the single one that happened to trigger confirmation)
    and return a dict describing the suspected attacker and victim.

    pcap_paths: a single path (str) or a list of paths, read in order as
    one continuous stream.

    trusted_ip_mac: optional dict of {ip: mac}, either operator-supplied
    at startup (see --trust in live_ids.py) or learned by the caller (see
    learn_trusted_mappings() below) from genuinely clean traffic earlier in
    the session. When provided, this is checked FIRST and is the highest-
    confidence resolution method -- see module docstring.

    Always returns a dict with at least these keys populated (None/empty
    when not determinable), so callers can safely index into it without
    checking for missing keys:
        attack_class, packet_count, method, resolution,
        attacker_ip, attacker_mac, impersonated_ip, victim_ip,
        evidence (list, method-specific detail)

    'resolution' records HOW an arp_reply_conflict case was resolved
    ("trusted_baseline", "cross_reference", or "first_seen_fallback") so a
    lower-confidence fallback result is never silently indistinguishable
    from a higher-confidence one.
    """

    if isinstance(pcap_paths, str):
        pcap_paths = [pcap_paths]

    trusted_ip_mac = trusted_ip_mac or {}

    result = {
        "attack_class": attack_class,
        "packet_count": 0,
        "method": None,
        "resolution": None,
        "attacker_ip": None,
        "attacker_mac": None,
        "impersonated_ip": None,
        "victim_ip": None,
        "evidence": [],
    }

    # ip -> ordered list of macs seen claiming it via ARP reply (order of
    # first appearance, kept only for the first-seen fallback)
    ip_reply_macs_ordered = defaultdict(list)

    # ip -> set of macs seen claiming it (for quick membership checks)
    ip_reply_macs = defaultdict(set)

    # (ip, mac) -> set of pdst values -- who received that specific claim
    claim_targets = defaultdict(set)

    # mac -> Counter of IPs seen as that mac's OWN source IP in ORDINARY
    # (non-ARP) IP traffic -- this is what reveals a mac's true identity
    mac_true_ip_counts = defaultdict(Counter)

    arp_request_counts = Counter()   # requester ip -> number of ARP requests sent
    arp_targets = defaultdict(set)   # requester ip -> set of target ips probed

    for pcap_path in pcap_paths:

        try:

            with PcapReader(pcap_path) as reader:

                for pkt in reader:

                    result["packet_count"] += 1

                    if ARP in pkt:

                        arp = pkt[ARP]

                        sender_ip = arp.psrc
                        sender_mac = arp.hwsrc.lower() if arp.hwsrc else None
                        target_ip = arp.pdst

                        if arp.op == 1:  # who-has (request)

                            if sender_ip:

                                arp_request_counts[sender_ip] += 1

                                if target_ip:
                                    arp_targets[sender_ip].add(target_ip)

                        elif arp.op == 2:  # is-at (reply)

                            if sender_ip and sender_mac:

                                if sender_mac not in ip_reply_macs[sender_ip]:
                                    ip_reply_macs_ordered[sender_ip].append(sender_mac)

                                ip_reply_macs[sender_ip].add(sender_mac)
                                claim_targets[(sender_ip, sender_mac)].add(target_ip)

                    elif IP in pkt and Ether in pkt:

                        # Ordinary (non-ARP) IP traffic -- reveals this
                        # MAC's own true IP identity, used for
                        # cross-referencing.
                        mac_true_ip_counts[pkt[Ether].src.lower()][pkt[IP].src] += 1

        except Exception as e:

            # A single unreadable/missing file (e.g. it rolled out of the
            # lookback window and got deleted mid-analysis) shouldn't abort
            # identification across the rest of the lookback window --
            # skip it and continue with whatever other files are usable.
            continue

    # --------------------------------------------------------------
    # Method 1: ARP reply conflict (MITM / ARP_DOS signature)
    # --------------------------------------------------------------

    conflicting_ips = [
        ip for ip, macs in ip_reply_macs.items()
        if len(macs) > 1
    ]

    if conflicting_ips:

        resolved = []  # list of {impersonated_ip, attacker_mac, victim_ip, resolution}

        for ip in conflicting_ips:

            macs = list(ip_reply_macs[ip])

            # (a) Trusted baseline: do we already know the real MAC for
            # this IP from earlier clean traffic this session?
            trusted_mac = trusted_ip_mac.get(ip)

            if trusted_mac and trusted_mac in macs:

                impostors = [m for m in macs if m != trusted_mac]

                if len(impostors) == 1:

                    attacker_mac = impostors[0]
                    resolution = "trusted_baseline"

                else:

                    attacker_mac = None
                    resolution = None

            else:

                attacker_mac = None
                resolution = None

            # (b) Cross-reference: does any claimant's OWN traffic show a
            # different true IP than the one it's claiming here?
            if attacker_mac is None:

                liars = [
                    mac for mac in macs
                    if mac_true_ip_counts.get(mac)
                    and mac_true_ip_counts[mac].most_common(1)[0][0] != ip
                ]

                if len(liars) == 1:

                    attacker_mac = liars[0]
                    resolution = "cross_reference"

            # (c) Fallback: first-seen heuristic (see docstring -- known
            # unreliable, kept only as a last resort).
            if attacker_mac is None:

                first_mac = ip_reply_macs_ordered[ip][0]
                later_macs = [m for m in macs if m != first_mac]

                attacker_mac = later_macs[0] if later_macs else None
                resolution = "first_seen_fallback"

            if attacker_mac is None:
                continue

            victim_ips = sorted(
                v for v in claim_targets.get((ip, attacker_mac), set()) if v
            )

            # Attacker's own genuine IP, if their own ordinary traffic within
            # the lookback window reveals it. Only reported when real evidence
            # supports it -- never inferred or guessed.
            attacker_ip = None

            if mac_true_ip_counts.get(attacker_mac):
                attacker_ip = mac_true_ip_counts[attacker_mac].most_common(1)[0][0]

            resolved.append({
                "impersonated_ip": ip,
                "attacker_mac": attacker_mac,
                "attacker_ip": attacker_ip,
                "victim_ip": ", ".join(victim_ips) if victim_ips else None,
                "resolution": resolution,
            })

        if resolved:

            # If multiple conflicting IPs point to the same attacker MAC
            # (the common case -- one attacker impersonating one host to
            # reach one or more victims), report that MAC as the attacker.
            mac_counter = Counter(r["attacker_mac"] for r in resolved)
            top_attacker_mac, _ = mac_counter.most_common(1)[0]

            related = [r for r in resolved if r["attacker_mac"] == top_attacker_mac]

            impersonated_ips = sorted(set(r["impersonated_ip"] for r in related))
            victim_ips = sorted(set(
                v for r in related if r["victim_ip"] for v in r["victim_ip"].split(", ")
            ))

            # Attacker's own IP, only if at least one related conflict
            # actually resolved one from real traffic evidence.
            attacker_ips_found = set(
                r["attacker_ip"] for r in related if r.get("attacker_ip")
            )
            attacker_ip = sorted(attacker_ips_found)[0] if len(attacker_ips_found) == 1 else None

            # Report the STRONGEST resolution method used among the related
            # claims -- surfacing the highest-confidence tier reached
            # rather than diluting it with a weaker fallback.
            resolutions_used = set(r["resolution"] for r in related)

            if "trusted_baseline" in resolutions_used:
                overall_resolution = "trusted_baseline"
            elif "cross_reference" in resolutions_used:
                overall_resolution = "cross_reference"
            else:
                overall_resolution = "first_seen_fallback"

            result["method"] = "arp_reply_conflict"
            result["resolution"] = overall_resolution
            result["attacker_mac"] = top_attacker_mac
            result["attacker_ip"] = attacker_ip
            result["impersonated_ip"] = ", ".join(impersonated_ips)
            result["victim_ip"] = ", ".join(victim_ips) if victim_ips else None
            result["evidence"] = related

            return result

    # --------------------------------------------------------------
    # Method 2: ARP request volume (RECON signature)
    # --------------------------------------------------------------

    if arp_request_counts:

        top_ip, req_count = arp_request_counts.most_common(1)[0]

        targets = arp_targets.get(top_ip, set())

        if req_count >= MIN_RECON_REQUESTS or len(targets) >= MIN_RECON_TARGETS:

            result["method"] = "arp_request_volume"
            result["attacker_ip"] = top_ip
            result["evidence"] = [{
                "requests_sent": req_count,
                "distinct_targets_probed": len(targets),
            }]

            return result

        # Evidence too weak (e.g. a single stray request) -- fall through
        # to "insufficient_evidence" rather than reporting a low-value guess.

    # --------------------------------------------------------------
    # No usable ARP evidence in this window
    # --------------------------------------------------------------

    result["method"] = "insufficient_evidence"

    return result


def learn_trusted_mappings(pcap_path):
    """
    Extract (ip -> mac) pairs from ARP replies in a capture believed to be
    CLEAN (e.g. a window live_ids.py just classified NORMAL). Called by the
    live IDS to build up trusted_ip_mac over the course of a session, BEFORE
    any attack happens -- this is what identify() checks first and is the
    only method with genuine ground truth rather than an inference made
    from the attack window alone.

    Returns a dict {ip: mac}. If an IP shows more than one MAC even within
    a single "clean" window (shouldn't happen on genuinely normal traffic,
    but caution costs nothing), it is OMITTED rather than guessed at --
    the caller should not learn an ambiguous mapping as if it were trusted.
    """

    ip_macs = defaultdict(set)

    try:

        with PcapReader(pcap_path) as reader:

            for pkt in reader:

                if ARP not in pkt:
                    continue

                arp = pkt[ARP]

                if arp.op == 2 and arp.psrc and arp.hwsrc:  # is-at (reply)
                    ip_macs[arp.psrc].add(arp.hwsrc.lower())

    except Exception:

        return {}

    return {
        ip: next(iter(macs))
        for ip, macs in ip_macs.items()
        if len(macs) == 1
    }


def format_identification(result):
    """
    Human-readable block for the live IDS alert output.
    """

    lines = []

    lines.append("SOURCE / IDENTIFICATION ANALYSIS")
    lines.append("-" * 75)

    method = result.get("method")

    if method == "arp_reply_conflict":

        resolution = result.get("resolution")

        confidence_label = {
            "trusted_baseline": "HIGH",
            "cross_reference": "MODERATE",
            "first_seen_fallback": "LOW",
        }.get(resolution, "LOW")

        lines.append("Method                 : ARP reply conflict (spoofed identity claim)")
        lines.append("")

        if result.get("attacker_ip"):
            lines.append(f"Suspected attacker IP  : {result['attacker_ip']}")
        else:
            lines.append("Suspected attacker IP  : Not determinable from captured evidence")

        lines.append(f"Suspected attacker MAC : {result['attacker_mac']}")
        lines.append("")

        if result["victim_ip"]:
            lines.append(f"Victim IP              : {result['victim_ip']}")
        else:
            lines.append("Victim IP              : Not determinable from captured evidence")

        lines.append(f"Impersonated IP        : {result['impersonated_ip']}")
        lines.append("")

        lines.append(f"Conflicting claims seen: {len(result['evidence'])}")
        lines.append(f"Resolution             : {resolution}")
        lines.append(f"Confidence             : {confidence_label}")

        if resolution == "cross_reference":
            lines.append(
                "Note: cross-reference is unreliable for router/gateway impersonation "
                "specifically -- see module docstring."
            )
        elif resolution == "first_seen_fallback":
            lines.append(
                "Note: no trusted baseline or cross-reference was available; this "
                "result is a last-resort heuristic and should be treated cautiously."
            )

    elif method == "arp_request_volume":

        ev = result["evidence"][0]

        lines.append("Method                 : ARP request volume (host discovery)")
        lines.append(f"Suspected attacker IP  : {result['attacker_ip']}")
        lines.append(f"ARP requests sent      : {ev['requests_sent']}")
        lines.append(f"Distinct targets probed: {ev['distinct_targets_probed']}")

    elif method == "error":

        lines.append(f"Could not complete identification: {result['evidence'][0]}")

    else:

        lines.append("Insufficient ARP evidence in this window to identify a source.")
        lines.append("(Model's confidence likely came from non-ARP features.)")

    return "\n".join(lines)


# ============================================================
# STANDALONE TEST MODE
#
# live_ids.py calls identify() automatically -- this lets you test
# identification directly against any saved .pcapng WITHOUT running the
# whole live pipeline or waiting for a live CONFIRMED alert. Useful for
# debugging: e.g. does identify() correctly report the attacker for one
# of your existing mitmnew*.pcapng training captures?
#
# Usage:
#     python3 identify_attacker.py --pcap mitmnew2.pcapng --attack-class MITM
#     python3 identify_attacker.py --pcap ban2.pcapng --attack-class ARP_DOS
#     python3 identify_attacker.py --pcap recon2.pcapng --attack-class RECON
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Test attacker/victim identification directly on a saved PCAP file."
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Path to a .pcap/.pcapng file to analyse."
    )

    parser.add_argument(
        "--attack-class",
        default=None,
        help="Label to display in the result (e.g. MITM, ARP_DOS, RECON). "
             "Informational only -- does not change the detection logic."
    )

    parser.add_argument(
        "--trusted-pcap",
        default=None,
        help="Optional path to a KNOWN-CLEAN capture (e.g. one of your normal*.pcapng "
             "files) to build a trusted IP-MAC baseline from, for testing the "
             "highest-confidence resolution tier directly."
    )

    args = parser.parse_args()

    trusted_ip_mac = None

    if args.trusted_pcap:

        print(f"Learning trusted IP-MAC mappings from {args.trusted_pcap} ...")

        trusted_ip_mac = learn_trusted_mappings(args.trusted_pcap)

        print(f"Learned {len(trusted_ip_mac)} trusted mapping(s): {trusted_ip_mac}")
        print()

    print(f"Analysing {args.pcap} ...")
    print()

    result = identify(args.pcap, attack_class=args.attack_class, trusted_ip_mac=trusted_ip_mac)

    print(format_identification(result))
    print()
    print(f"(Packets read: {result['packet_count']})")
