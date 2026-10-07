================================================================================
Machine Learning-Based Detection, Identification, and Mitigation of MITM and
DoS Attacks in Network Environments
================================================================================

Student:      Tharun Reddy Adaboina
Student ID:   2467027
Module:       MOD002726 - Postgraduate Major Project (2025 TRI3)
Department:   M.Sc. Cybersecurity, Faculty of Science and Technology,
              Anglia Ruskin University


--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------

This artefact is a gateway-based, machine learning intrusion detection system
(IDS) that detects Man-in-the-Middle (MITM), ARP-based Denial-of-Service
(ARP_DOS), and reconnaissance (RECON) attacks in real time from live network
traffic, and identifies the specific attacker and victim once an attack is
confirmed. It requires no software installation on any protected or attacking
end-host -- it observes traffic passively from a single gateway/monitoring
position.

All experiments were conducted in a private, authorised laboratory
environment. This artefact is provided for academic assessment and
demonstration purposes only. It must not be run against any network the
person running it does not own or have explicit written permission to test.


--------------------------------------------------------------------------------
2. FOLDER STRUCTURE
--------------------------------------------------------------------------------

codes/
    extract_features.py    - Converts a packet capture (.pcap/.pcapng) into a
                              CSV of 38 behavioural features per 5-second
                              window. Used both for offline dataset building
                              and, as an importable module, by live_ids.py.

    train_model.py          - Trains and evaluates the Random Forest
                              classifier from the feature CSV, using a
                              capture-level train/test split. Prints the
                              classification report, confusion matrix, and
                              feature importances, and saves the trained
                              model.

    live_ids.py              - The real-time detection pipeline. Captures
                              live traffic in rotating 5-second windows,
                              classifies each window, applies temporal
                              confirmation, and calls identify_attacker.py
                              once an attack is confirmed. Logs all events
                              and confirmed alerts to CSV.

    identify_attacker.py     - Attacker/victim identification module. Given
                              recent capture files and an attack class,
                              resolves conflicting ARP claims to name the
                              likely attacker (IP/MAC), victim, and
                              impersonated identity, with a confidence label.
                              Can also be run standalone for testing.

datasets_and_models/
    dataset.csv               - The final labelled feature dataset (229
                              windows across 14 independent captures:
                              NORMAL, MITM, ARP_DOS, RECON).

    rf_model_200_corrected.pkl - The final trained Random Forest model
                              (200 trees), saved together with its expected
                              feature-column schema.

training_traffic_captures/
                              - Raw packet captures (.pcapng) used to build
                              the dataset above, organised by class
                              (e.g. normal*.pcapng, mitmnew*.pcapng,
                              ban*.pcapng, recon*.pcapng).

ids_logs/
    ids_events.csv            - Full log of every classification decision
                              made during live testing sessions.

    ids_alerts.csv            - Subset of the above containing only
                              CONFIRMED security alerts, with identification
                              results (attacker IP/MAC, victim IP) attached.


--------------------------------------------------------------------------------
3. REQUIREMENTS
--------------------------------------------------------------------------------

- Python 3.9 or later
- A Linux environment (developed and tested on Kali Linux) with root/sudo
  access for live packet capture
- Wireshark / tshark (provides dumpcap, used for live capture)

Python packages:

    pip install scapy scikit-learn pandas joblib numpy --break-system-packages

(Drop --break-system-packages if not needed on your system, e.g. inside a
virtual environment.)


--------------------------------------------------------------------------------
4. HOW TO RUN
--------------------------------------------------------------------------------

All commands below are run from inside the codes/ folder.

--- 4.1 Rebuild the dataset from raw captures (optional -- dataset.csv is
    already provided in datasets_and_models/) ---

    python3 extract_features.py \
        --pcap ../training_traffic_captures/normal.pcapng \
        --label NORMAL \
        --window 5 \
        --local-subnet <your_subnet>/24 \
        --out ../datasets_and_models/dataset.csv

    Repeat for each capture file, changing --label to MITM, ARP_DOS, or
    RECON as appropriate. Use --skip-seconds N to trim idle lead-in time
    from a capture if the attack tool was not started immediately.

--- 4.2 Retrain the model (optional -- the trained model is already
    provided) ---

    python3 train_model.py \
        --csv ../datasets_and_models/dataset.csv \
        --model ../datasets_and_models/rf_model_200_corrected.pkl

--- 4.3 Run the live IDS (the main demonstration) ---

    First, identify your monitoring interface and subnet:

        ip a
        ip route | grep default

    Then run (requires sudo for packet capture):

        sudo python3 live_ids.py \
            --interface eth0 \
            --model ../datasets_and_models/rf_model_200_corrected.pkl \
            --local-subnet <your_subnet>/24 \
            --trust <gateway_IP>=<gateway_MAC>

    Optional flags:
        --identify-lookback N      Number of recent capture files considered
                                  during identification (default: 3).
        --confirmation-span N      Number of recent windows considered for
                                  temporal confirmation (default: 4).
        --confidence N             Confidence threshold percentage
                                  (default: 70).

    The console will print live classification results, and a full
    identification breakdown (attacker/victim/method/confidence) whenever
    an attack is CONFIRMED. Automatic blocking is disabled by design --
    this artefact is detection/alert/identification only.

--- 4.4 Run the identification module standalone (for testing against a
    specific capture) ---

    python3 identify_attacker.py \
        --pcap ../training_traffic_captures/mitmnew.pcapng \
        --attack-class MITM \
        --trusted-pcap ../training_traffic_captures/normal.pcapng


--------------------------------------------------------------------------------
5. NOTES
--------------------------------------------------------------------------------

- --local-subnet must be set to the ACTUAL local subnet of whichever network
  interface is being monitored, or IP-MAC mapping tracking will be
  unreliable (see report, Section 5.1, for why this matters).
- --trust should be seeded with the real, verified gateway IP/MAC of the
  network being monitored. This value is protected internally and cannot
  be silently overwritten by the system's own passive learning.
- Live demonstration requires an attack to be generated on the same network
  segment (e.g. using bettercap's arp.spoof, ban, or net.probe modules) for
  the IDS to have traffic to detect.

================================================================================
END OF FILE
================================================================================
