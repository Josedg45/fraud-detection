def fraud_decision(score: float, threshold: float = 0.2):
    if score >= threshold:
        return "BLOCK"
    elif score >= threshold * 0.7:
        return "CHALLENGE"
    return "ALLOW"
