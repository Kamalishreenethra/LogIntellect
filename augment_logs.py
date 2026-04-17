import random

# ─────────────────────────────────────────────
# LOAD ORIGINAL LOGS
# ─────────────────────────────────────────────
with open("HDFS_2k.log", "r", encoding="utf-8", errors="ignore") as f:
    logs = [l.strip() for l in f if l.strip()]

# ─────────────────────────────────────────────
# 1. REAL HDFS ANOMALIES
# ─────────────────────────────────────────────
real_anomalies = [
    "ERROR dfs.DataNode: Failed to write block blk_123456",
    "WARN dfs.DataNode: Connection reset by peer",
    "ERROR dfs.DataNode: writeBlock received exception java.io.IOException: Broken pipe",
    "ERROR dfs.FSNamesystem: Missing block detected blk_99999",
    "WARN dfs.DataNode: PacketResponder interrupted unexpectedly",
]

# ─────────────────────────────────────────────
# 2. SUBTLE ANOMALIES (INFO-LEVEL)
# ─────────────────────────────────────────────
subtle_anomalies = [
    "INFO unusual block delay detected in DataNode",
    "INFO repeated access from same IP 10.0.0.5",
    "INFO abnormal replication pattern detected",
    "INFO possible unauthorized pattern in file access",
    "INFO high latency observed in block transfer",
]

# ─────────────────────────────────────────────
# 3. SECURITY ATTACK LOGS
# ─────────────────────────────────────────────
security_logs = [
    "INFO Failed login attempt from 192.168.1.5",
    "INFO multiple authentication failures detected",
    "WARN unusual login pattern detected",
    "WARN brute force suspected from 10.0.0.3",
    "ERROR unauthorized access detected to /admin",
    "CRITICAL malware detected in upload",
    "ERROR SQL injection attempt detected",
    "WARN multiple authentication failures detected",
]

# ─────────────────────────────────────────────
# 4. GENERATE SYNTHETIC DATA
# ─────────────────────────────────────────────
def generate_logs(base_list, count):
    return [random.choice(base_list) for _ in range(count)]

# You can tune these numbers
real = generate_logs(real_anomalies, 120)
subtle = generate_logs(subtle_anomalies, 120)
security = generate_logs(security_logs, 120)

# ─────────────────────────────────────────────
# 5. COMBINE DATASET
# ─────────────────────────────────────────────
enhanced_logs = logs + real + subtle + security

# Shuffle to mix properly
random.shuffle(enhanced_logs)

# ─────────────────────────────────────────────
# 6. SAVE NEW DATASET
# ─────────────────────────────────────────────
with open("enhanced_HDFS.log", "w", encoding="utf-8") as f:
    for line in enhanced_logs:
        f.write(line + "\n")

print(f"✅ Enhanced dataset created: {len(enhanced_logs)} logs")