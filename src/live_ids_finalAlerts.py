#!/usr/bin/env python3

"""
live_ids.py

Improved Live ML-Based Network Intrusion Detection System.

Pipeline:

    Network Interface
          |
          v
       dumpcap
          |
          v
    5-second PCAP
          |
          v
   Feature Extraction
          |
          v
    Random Forest ML
          |
          v
   Temporal Validation
          |
          v
  Attacker/Victim Identification (only on confirmed attacks)
          |
          v
      Alert + Logging


Detection classes:
    NORMAL
    MITM
    RECON
    ARP_DOS

This version is DETECTION + IDENTIFICATION.

It does NOT:
    - block IP addresses
    - modify firewall rules
    - modify routing
    - disconnect devices
    - inject packets

Intended for authorised/private lab use.

Changes vs the previous draft (see review):
    - analyse_sources() removed. It re-parsed the FULL pcap with rdpcap()
      on every single window (even NORMAL ones) via a naive "top source by
      packet volume" heuristic, which doesn't distinguish attacker from
      victim at all. Replaced by identify_attacker.identify(), called ONLY
      when an attack is CONFIRMED (not every window), using the actual ARP
      reply semantics (who claimed an IP, who received the poisoned entry)
      instead of raw packet counts.
    - Statistics no longer re-run model.predict() a second time; the
      predictions already computed for the alert are reused.
"""

import argparse
import csv
import glob
import ipaddress
import os
import subprocess
import sys
import time
from datetime import datetime

import joblib
import pandas as pd

try:
    from extract_features import extract as extract_features
except ImportError:
    print("[ERROR] Could not import extract_features.py.")
    print("        Make sure extract_features.py is in the same directory.")
    sys.exit(1)

try:
    from identify_attacker import identify, format_identification, learn_trusted_mappings
except ImportError:
    print("[ERROR] Could not import identify_attacker.py.")
    print("        Make sure identify_attacker.py is in the same directory.")
    sys.exit(1)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_INTERFACE = "eth0"
DEFAULT_WINDOW = 5.0

DEFAULT_CAPTURE_DIR = "live_captures"
DEFAULT_LOG_DIR = "ids_logs"

DEFAULT_MODEL = "rf_model_200_corrected.pkl"

DEFAULT_CONFIDENCE = 70.0

DEFAULT_CONFIRMATIONS = 2

DEFAULT_KEEP_CAPTURES = False

DEFAULT_MAX_HISTORY = 100


# ============================================================
# MODEL CLASSES
# ============================================================

ATTACK_CLASSES = {
    "MITM",
    "RECON",
    "ARP_DOS"
}

# MITM and ARP_DOS share the same underlying mechanism -- forged ARP
# replies -- and only differ in intent (redirect vs. disconnect), which
# raw packet features don't always cleanly reveal. Live testing showed the
# model is consistently confident traffic is "one of these two ARP-reply
# attacks" (combined probability often 90%+) while being genuinely torn on
# WHICH one (single-class confidence sometimes as low as 45-65%). Grouping
# them for confirmation purposes uses the combined probability instead of
# requiring one specific class alone to clear the threshold -- this
# reflects a real limitation (fine-grained MITM vs ARP_DOS separation is
# hard from ARP features alone) rather than papering over it: the alert
# still reports the model's specific top guess, just confirms based on the
# grouped signal.
ATTACK_GROUPS = {
    "MITM": "ARP_POISONING",
    "ARP_DOS": "ARP_POISONING",
    "RECON": "RECON",
}


# ============================================================
# BANNER
# ============================================================

def print_banner():

    print()
    print("=" * 75)
    print("             ML NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 75)
    print()
    print(" Detection       : Random Forest")
    print(" Identification  : ARP reply/claim analysis (on confirmed attacks)")
    print(" Feature count   : 38")
    print(" Window          : 5 seconds")
    print(" Blocking        : DISABLED")
    print()


# ============================================================
# COMMAND CHECK
# ============================================================

def check_command(command):

    result = subprocess.run(
        ["which", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directory(path):

    os.makedirs(path, exist_ok=True)


# ============================================================
# CAPTURE FILE DISCOVERY
# ============================================================

def get_existing_capture_files(capture_dir):

    patterns = [
        os.path.join(capture_dir, "*.pcap"),
        os.path.join(capture_dir, "*.pcapng")
    ]

    files = []

    for pattern in patterns:
        files.extend(glob.glob(pattern))

    return set(
        os.path.abspath(f)
        for f in files
    )


# ============================================================
# WAIT FOR FILE TO STOP CHANGING
# ============================================================

def wait_for_stable_file(
    path,
    checks=2,
    delay=0.5
):

    previous_size = -1
    stable_count = 0

    while stable_count < checks:

        if not os.path.exists(path):

            time.sleep(delay)
            continue

        try:

            current_size = os.path.getsize(path)

        except OSError:

            time.sleep(delay)
            continue

        if current_size == previous_size:

            stable_count += 1

        else:

            stable_count = 0

        previous_size = current_size

        time.sleep(delay)


# ============================================================
# FIND NEW CAPTURE
# ============================================================

def get_new_capture_file(
    capture_dir,
    known_files,
    timeout=30
):

    start = time.time()

    while time.time() - start < timeout:

        current_files = get_existing_capture_files(
            capture_dir
        )

        new_files = current_files - known_files

        if new_files:

            candidate = sorted(
                new_files,
                key=lambda f: os.path.getmtime(f)
            )[0]

            return candidate

        time.sleep(0.25)

    return None


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_in_memory(
    pcap_file,
    window_size,
    local_subnet_str=None
):

    local_network = None

    if local_subnet_str:

        try:

            local_network = ipaddress.ip_network(
                local_subnet_str,
                strict=False
            )

        except ValueError as e:

            print(
                f"[ERROR] Invalid local subnet: {e}"
            )

            return pd.DataFrame()

    try:

        rows = extract_features(
            pcap_path=pcap_file,
            label="NORMAL",
            window_size=window_size,
            out_path=None,
            local_network=local_network,
            quiet=True
        )

        return pd.DataFrame(rows)

    except Exception as e:

        print(
            f"[ERROR] Feature extraction failed: {e}"
        )

        return pd.DataFrame()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    package = joblib.load(model_path)

    if not isinstance(package, dict):

        raise ValueError(
            "Unexpected model format."
        )

    if "model" not in package:

        raise ValueError(
            "Saved model does not contain 'model'."
        )

    if "feature_cols" not in package:

        raise ValueError(
            "Saved model does not contain 'feature_cols'."
        )

    return (
        package["model"],
        package["feature_cols"]
    )


# ============================================================
# TEMPORAL DETECTOR
# ============================================================

class TemporalDetector:

    def __init__(
        self,
        confidence_threshold=70.0,
        confirmations=2,
        confirmation_span=4,
        max_history=100
    ):

        self.confidence_threshold = (
            confidence_threshold
        )

        self.confirmations_required = (
            confirmations
        )

        # How many of the MOST RECENT windows are considered when counting
        # toward confirmation (see evaluate() docstring for why this
        # replaced a strict "N consecutive windows" requirement).
        self.confirmation_span = confirmation_span

        self.max_history = max_history

        self.history = []

        # Recent (group, high_confidence) outcomes, most recent last,
        # capped at confirmation_span -- this is what confirmation counts
        # over now, instead of a strict consecutive streak.
        self.recent_groups = []

        self.attack_streak = 0

        self.last_attack = None

    def evaluate(
        self,
        prediction,
        confidence,
        probability_row=None,
        classes=None
    ):
        """
        Returns (confirmed: bool, effective_confidence: float, group: str|None).

        If prediction belongs to a multi-class group (see ATTACK_GROUPS) and
        the full probability_row/classes are provided, effective_confidence
        is the SUM of probabilities across every class in that group --
        e.g. MITM% + ARP_DOS% -- rather than just the top class's own
        confidence. This is what lets the detector confirm "this is some
        kind of ARP-reply attack" even when it can't confidently pick MITM
        vs ARP_DOS specifically.

        Confirmation counts high-confidence hits of the SAME group within
        the last confirmation_span windows, rather than requiring them to
        be strictly consecutive. This was changed after live testing showed
        some attack tools (e.g. bettercap's net.probe/net.recon) produce a
        genuinely intermittent, alternating pattern -- a high-confidence
        detection, then a window that looks normal, then another high-
        confidence detection -- which a strict consecutive-streak
        requirement never confirms even after dozens of correct individual
        detections, because no two ever land back-to-back. A sliding span
        tolerates that regular alternation while still requiring multiple
        independent high-confidence hits, not just one.
        """

        group = ATTACK_GROUPS.get(prediction)

        if group == "ARP_POISONING" and probability_row is not None and classes is not None:

            effective_confidence = sum(
                probability_row[i] * 100
                for i, c in enumerate(classes)
                if ATTACK_GROUPS.get(c) == "ARP_POISONING"
            )

        else:

            effective_confidence = confidence

        high_confidence = (
            effective_confidence >= self.confidence_threshold
        )

        # Record this window's outcome (group if it was a high-confidence
        # hit, else None) and keep only the most recent confirmation_span.
        self.recent_groups.append(group if high_confidence else None)

        if len(self.recent_groups) > self.confirmation_span:

            self.recent_groups.pop(0)

        if group is not None and high_confidence:

            count = sum(1 for g in self.recent_groups if g == group)
            self.attack_streak = count
            self.last_attack = group

        else:

            self.attack_streak = 0
            self.last_attack = None

        confirmed = (
            group is not None
            and high_confidence
            and self.attack_streak >= self.confirmations_required
        )

        self.history.append(
            {
                "prediction": prediction,
                "confidence": confidence,
                "effective_confidence": effective_confidence,
            }
        )

        if len(self.history) > self.max_history:

            self.history.pop(0)

        return confirmed, effective_confidence, group


# ============================================================
# PRINT DETECTION RESULT
# ============================================================

def print_detection_result(
    df,
    model,
    feature_cols,
    lookback_files,
    temporal_detector,
    window_number,
    logger,
    trusted_ip_mac
):
    """
    Returns (attack_confirmed, predictions_list) so the caller can reuse
    the predictions for statistics instead of calling model.predict() again.
    """

    if df.empty:

        print(
            "[INFO] No feature windows were produced."
        )

        return False, []

    missing = [
        col
        for col in feature_cols
        if col not in df.columns
    ]

    if missing:

        print("[ERROR] Missing model features:")

        for feature in missing:

            print(
                f"  - {feature}"
            )

        return False, []

    X = df[feature_cols]

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    attack_confirmed = False

    for index, prediction in enumerate(predictions):

        probability_row = probabilities[index]

        confidence = (
            max(probability_row) * 100
        )

        current_window = (
            window_number + index
        )

        print()
        print("=" * 75)
        print("                     ML DETECTION RESULT")
        print("=" * 75)

        print(
            f"Window       : {current_window}"
        )

        print(
            f"Prediction   : {prediction}"
        )

        print(
            f"Confidence   : {confidence:.2f}%"
        )

        print()
        print("Probabilities:")

        for class_name, probability in zip(
            classes,
            probability_row
        ):

            print(
                f"  {class_name:<10} "
                f"{probability * 100:6.2f}%"
            )

        confirmed, effective_confidence, group = temporal_detector.evaluate(
            prediction,
            confidence,
            probability_row=probability_row,
            classes=classes
        )

        is_grouped = (
            group == "ARP_POISONING"
            and effective_confidence != confidence
        )

        identification_result = None

        if prediction == "NORMAL":

            print()
            print(
                "Status       : NORMAL TRAFFIC"
            )

        elif (
            prediction in ATTACK_CLASSES
            and effective_confidence <
            temporal_detector.confidence_threshold
        ):

            print()
            print(
                "Status       : SUSPICIOUS / LOW CONFIDENCE"
            )

            print(
                f"Threshold    : "
                f"{temporal_detector.confidence_threshold:.2f}%"
            )

            if is_grouped:
                print(
                    f"Combined ARP_DOS+MITM confidence : {effective_confidence:.2f}%"
                )

            print(
                "Validation   : Waiting for confirmation"
            )

        elif not confirmed:

            print()
            print(
                "Status       : POTENTIAL ATTACK"
            )

            print(
                f"Attack class : {prediction}"
                + (" (grouped: ARP_POISONING)" if is_grouped else "")
            )

            print(
                f"Confidence   : {confidence:.2f}%"
                + (f"  (combined MITM+ARP_DOS: {effective_confidence:.2f}%)" if is_grouped else "")
            )

            print(
                f"Confirmation : "
                f"{temporal_detector.attack_streak}/"
                f"{temporal_detector.confirmations_required}"
            )

            print(
                "Action       : Monitoring"
            )

        else:

            attack_confirmed = True

            print()
            print(
                "!!! CONFIRMED SECURITY ALERT !!!"
            )

            print(
                f"Attack class : {prediction}"
                + (" (grouped: ARP_POISONING -- see note below)" if is_grouped else "")
            )

            print(
                f"Confidence   : {confidence:.2f}%"
                + (f"  (combined MITM+ARP_DOS: {effective_confidence:.2f}%)" if is_grouped else "")
            )

            print(
                f"Confirmed by : "
                f"{temporal_detector.attack_streak} high-confidence windows "
                f"within the last {temporal_detector.confirmation_span}"
            )

            if is_grouped:
                print(
                    "Note         : MITM and ARP_DOS share the same forged-ARP-reply "
                    "mechanism; confirmed as an ARP poisoning event. The specific "
                    f"subtype ({prediction}) is the model's best guess, not a certainty."
                )

            # ------------------------------------------------
            # Identification -- ONLY runs here, on a confirmed
            # attack, not on every window (see module docstring)
            # ------------------------------------------------

            print()
            print("[*] Running attacker/victim identification...")

            identification_result = identify(lookback_files, attack_class=prediction, trusted_ip_mac=trusted_ip_mac)

            print(format_identification(identification_result))

            print(
                "Action         : Detection/alert only"
            )

            print(
                "Blocking       : DISABLED"
            )

        timestamp = datetime.now().isoformat(
            timespec="seconds"
        )

        event = {
            "timestamp": timestamp,
            "window": current_window,
            "prediction": str(prediction),
            "confidence": round(confidence, 2),
            "confirmed_attack": confirmed,
            "suspected_attacker_ip":
                identification_result["attacker_ip"] if identification_result else "",
            "suspected_attacker_mac":
                identification_result["attacker_mac"] if identification_result else "",
            "victim_ip":
                identification_result["victim_ip"] if identification_result else "",
        }

        logger.write(event)

    print()
    print("=" * 75)

    return attack_confirmed, list(predictions)


# ============================================================
# CSV LOGGER
# ============================================================

class EventLogger:

    FIELDNAMES = [
        "timestamp",
        "window",
        "prediction",
        "confidence",
        "confirmed_attack",
        "suspected_attacker_ip",
        "suspected_attacker_mac",
        "victim_ip",
    ]

    def __init__(self, log_dir):

        ensure_directory(log_dir)

        self.path = os.path.join(
            log_dir,
            "ids_events.csv"
        )

        self.alert_path = os.path.join(
            log_dir,
            "ids_alerts.csv"
        )

        file_exists = os.path.isfile(
            self.path
        )

        alert_file_exists = os.path.isfile(
            self.alert_path
        )

        self.file = open(
            self.path,
            "a",
            newline="",
            buffering=1
        )

        self.alert_file = open(
            self.alert_path,
            "a",
            newline="",
            buffering=1
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.FIELDNAMES
        )

        self.alert_writer = csv.DictWriter(
            self.alert_file,
            fieldnames=self.FIELDNAMES
        )

        if not file_exists:

            self.writer.writeheader()

        if not alert_file_exists:

            self.alert_writer.writeheader()

    def write(self, event):

        # Keep the complete event history as before.
        self.writer.writerow(event)

        # Keep a separate file containing ONLY confirmed security alerts.
        if event.get("confirmed_attack"):
            self.alert_writer.writerow(event)

    def close(self):

        try:

            self.file.close()

        except Exception:
            pass

        try:

            self.alert_file.close()

        except Exception:
            pass


# ============================================================
# LIVE SUMMARY
# ============================================================

class LiveStatistics:

    def __init__(self):

        self.windows = 0
        self.normal = 0
        self.mitm = 0
        self.recon = 0
        self.arp_dos = 0
        self.alerts = 0

    def update(
        self,
        prediction,
        confirmed
    ):

        self.windows += 1

        if prediction == "NORMAL":
            self.normal += 1
        elif prediction == "MITM":
            self.mitm += 1
        elif prediction == "RECON":
            self.recon += 1
        elif prediction == "ARP_DOS":
            self.arp_dos += 1

        if confirmed:
            self.alerts += 1

    def print_summary(self):

        print()
        print("-" * 75)
        print("                         LIVE IDS SUMMARY")
        print("-" * 75)

        print(f"Windows processed : {self.windows}")
        print(f"NORMAL            : {self.normal}")
        print(f"MITM              : {self.mitm}")
        print(f"RECON             : {self.recon}")
        print(f"ARP_DOS           : {self.arp_dos}")
        print(f"Confirmed alerts  : {self.alerts}")

        print("-" * 75)


# ============================================================
# START / STOP DUMPCAP
# ============================================================

def start_packet_capture(
    interface,
    capture_dir,
    window_size
):

    ensure_directory(capture_dir)

    timestamp_pattern = os.path.join(
        capture_dir,
        "live-%Y%m%d-%H%M%S.pcapng"
    )

    command = [
        "dumpcap",
        "-i",
        interface,
        "-q",
        "-b",
        f"duration:{window_size}",
        "-w",
        timestamp_pattern
    ]

    print()
    print("[*] Starting packet capture...")
    print(f"[*] Interface : {interface}")
    print(f"[*] Window    : {window_size} seconds")
    print(f"[*] Directory : {capture_dir}")
    print()
    print("[*] Capture command:")
    print("    " + " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    return process


def cleanup_capture_process(process):

    if process is None:
        return

    if process.poll() is None:

        print()
        print("[*] Stopping packet capture...")

        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Improved live ML-based Network Intrusion Detection System"
    )

    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--local-subnet", default=None)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--confirmations", type=int, default=DEFAULT_CONFIRMATIONS)
    parser.add_argument(
        "--confirmation-span",
        type=int,
        default=4,
        help="Number of MOST RECENT windows considered when counting toward confirmation "
             "(default: 4). Confirmation now counts high-confidence hits within this recent "
             "span rather than requiring them strictly consecutive -- some attack tools "
             "(e.g. bettercap's net.probe) produce a genuinely intermittent, alternating "
             "pattern that a strict consecutive-streak rule never confirms."
    )
    parser.add_argument("--keep-captures", action="store_true", default=DEFAULT_KEEP_CAPTURES)
    parser.add_argument(
        "--trust",
        action="append",
        default=[],
        metavar="IP=MAC",
        help="Manually seed a known-good IP-MAC mapping (e.g. --trust 10.50.108.1=00:08:e3:ff:fc:28). "
             "Repeatable. Use this for the gateway or other critical hosts whose real MAC you already "
             "know -- guarantees correct trusted-baseline attacker identification from window 1, "
             "instead of depending on passively observing a genuine ARP reply before any attack starts."
    )
    parser.add_argument(
        "--identify-lookback",
        type=int,
        default=3,
        help="Number of recent rotated capture files (including the current one) to analyse together "
             "when an attack is CONFIRMED (default: 3, i.e. up to ~15s of context at the default 5s "
             "window). An ongoing attack doesn't necessarily produce fresh ARP conflict evidence in "
             "every single window, so widening the evidence window improves identification's chances."
    )

    args = parser.parse_args()

    print_banner()

    if args.window <= 0:
        raise ValueError("Window must be greater than zero.")

    if args.confidence < 0 or args.confidence > 100:
        raise ValueError("Confidence must be between 0 and 100.")

    if args.confirmations < 1:
        raise ValueError("Confirmations must be at least 1.")

    if not check_command("dumpcap"):
        print("[ERROR] dumpcap was not found.")
        print("Install Wireshark/dumpcap first.")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    ensure_directory(args.capture_dir)
    ensure_directory(args.log_dir)

    print("[*] Loading trained model...")
    model, feature_cols = load_model(args.model)

    print(f"[*] Model type : {type(model).__name__}")
    print(f"[*] Features   : {len(feature_cols)}")
    print(f"[*] Classes    : {list(model.classes_)}")
    print(f"[*] Confidence : {args.confidence:.2f}%")
    print(f"[*] Confirmation windows : {args.confirmations} (within last {args.confirmation_span} windows)")
    print(f"[*] Event log : {os.path.join(args.log_dir, 'ids_events.csv')}")
    print(f"[*] Alert log : {os.path.join(args.log_dir, 'ids_alerts.csv')}")

    temporal_detector = TemporalDetector(
        confidence_threshold=args.confidence,
        confirmations=args.confirmations,
        confirmation_span=args.confirmation_span,
        max_history=DEFAULT_MAX_HISTORY
    )

    statistics = LiveStatistics()
    logger = EventLogger(args.log_dir)

    # Trusted IP-MAC baseline, built continuously from NORMAL-classified
    # windows over the course of the session. This is what gives
    # identify() a genuine ground truth to check attack windows against,
    # rather than inferring everything from the attack window alone (see
    # identify_attacker.py docstring for why that was proven unreliable).
    # Seed with any operator-supplied --trust mappings first.
    trusted_ip_mac = {}

    # IPs seeded via --trust are operator-VERIFIED ground truth. They must
    # never be silently overwritten by passive learning below -- even a
    # window the classifier calls NORMAL can still contain an ongoing
    # attack's traffic (confidence can dip below threshold mid-attack,
    # especially on noisy networks with many other devices), and learning
    # from it would poison exactly the baseline this flag exists to fix.
    locked_ips = set()

    for entry in args.trust:

        if "=" not in entry:
            print(f"[WARNING] Ignoring malformed --trust value (expected IP=MAC): {entry}")
            continue

        ip, mac = entry.split("=", 1)
        ip = ip.strip()
        trusted_ip_mac[ip] = mac.strip().lower()
        locked_ips.add(ip)

    if trusted_ip_mac:
        print(f"[*] Seeded trusted IP-MAC mappings: {trusted_ip_mac}")

    # Rolling buffer of the most recent rotated capture files (including
    # the current one). When an attack CONFIRMS, identification analyses
    # ALL of these together rather than just the single triggering window
    # -- see --identify-lookback. Files are deleted only once they roll
    # out of this window (unless --keep-captures).
    recent_files = []

    known_files = get_existing_capture_files(args.capture_dir)
    capture_process = None
    window_number = 0

    try:

        capture_process = start_packet_capture(
            args.interface, args.capture_dir, args.window
        )

        print()
        print("[*] Live IDS is running.")
        print("[*] Detection is active.")
        print("[*] Identification runs only on CONFIRMED attacks.")
        print("[*] Automatic blocking is DISABLED.")
        print("[*] Press CTRL+C to stop.")
        print()

        while True:

            if capture_process.poll() is not None:

                print()
                print("[ERROR] dumpcap stopped unexpectedly.")

                stderr = capture_process.stderr.read()

                if stderr:
                    print(stderr)

                break

            new_file = get_new_capture_file(
                args.capture_dir, known_files, timeout=30
            )

            if new_file is None:

                if capture_process.poll() is not None:

                    print()
                    print("[ERROR] dumpcap stopped unexpectedly.")

                    stderr = capture_process.stderr.read()

                    if stderr:
                        print(stderr)

                    break

                continue

            known_files.add(os.path.abspath(new_file))
            window_number += 1

            print()
            print("=" * 75)
            print(f"[*] New capture: {new_file}")
            print(f"[*] Window: {window_number}")
            print(f"[*] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 75)

            wait_for_stable_file(new_file)

            print("[*] Extracting features...")

            df = extract_features_in_memory(
                new_file, args.window, args.local_subnet
            )

            # Add to the rolling lookback buffer BEFORE identification, so
            # identify() can see the current window plus recent history.
            recent_files.append(new_file)

            confirmed_attack, predictions = print_detection_result(
                df,
                model,
                feature_cols,
                list(recent_files),
                temporal_detector,
                window_number,
                logger,
                trusted_ip_mac
            )

            for prediction in predictions:
                statistics.update(prediction, confirmed_attack)

            # Learn trusted IP-MAC mappings from this window ONLY if it was
            # classified NORMAL -- never learn from a window that might
            # itself be under attack, or the baseline could be poisoned.
            # Critically: NEVER let a learned mapping overwrite an IP the
            # operator explicitly verified via --trust, even if this
            # window's own classification was NORMAL -- a window can score
            # NORMAL while an ongoing attack's traffic is still physically
            # present (confidence can dip mid-attack), and that traffic
            # must never be able to poison operator-supplied ground truth.
            if predictions and all(p == "NORMAL" for p in predictions):

                learned = learn_trusted_mappings(new_file)

                for ip, mac in learned.items():

                    if ip in locked_ips:

                        if trusted_ip_mac.get(ip) != mac:
                            print(
                                f"[WARNING] Ignoring learned mapping {ip}={mac} -- "
                                f"conflicts with operator-verified --trust value "
                                f"{ip}={trusted_ip_mac.get(ip)}. Keeping the verified one."
                            )

                        continue

                    trusted_ip_mac[ip] = mac

            # Evict and delete files that have rolled out of the lookback
            # window (rather than deleting every file immediately) so
            # identify() has access to recent history, not just the single
            # triggering window.
            while len(recent_files) > args.identify_lookback:

                evicted = recent_files.pop(0)

                if not args.keep_captures:

                    try:
                        os.remove(evicted)
                    except OSError:
                        pass

            statistics.print_summary()

            print()
            print("[*] Waiting for next capture...")

    except KeyboardInterrupt:

        print()
        print("[*] CTRL+C received.")

    finally:

        cleanup_capture_process(capture_process)

        if not args.keep_captures:

            for f in recent_files:
                try:
                    os.remove(f)
                except OSError:
                    pass

        logger.close()
        statistics.print_summary()

        print()
        print("[*] Live IDS stopped.")
        print(f"[*] Event log saved to: {os.path.join(args.log_dir, 'ids_events.csv')}")
        print(f"[*] Alert log saved to: {os.path.join(args.log_dir, 'ids_alerts.csv')}")


if __name__ == "__main__":
    main()
