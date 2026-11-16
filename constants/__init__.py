"""Official indoor volleyball court dimensions (meters)."""

# Full court
COURT_WIDTH = 9.0
COURT_LENGTH = 18.0
HALF_COURT_LENGTH = COURT_LENGTH / 2.0

# Attack / 3-meter line distance from the net
ATTACK_LINE_DISTANCE = 3.0

# Region between end line and attack line
NO_MANS_LAND_HEIGHT = HALF_COURT_LENGTH - ATTACK_LINE_DISTANCE  # 6.0 m

# Kept for backward compatibility with MiniCourt drawing
LINE_WIDTH = COURT_WIDTH
HALF_COURT_LINE_HEIGHT = HALF_COURT_LENGTH
LINE_HEIGHT = COURT_LENGTH
SERVICE_LINE_WIDTH = ATTACK_LINE_DISTANCE

# Approximate adult player height used for scale heuristics
PLAYER_HEIGHT = 1.75

# Detection / analysis defaults
DEFAULT_BALL_CONFIDENCE = 0.15
DEFAULT_PLAYER_CONFIDENCE = 0.35
MAX_PLAYERS_ON_COURT = 12
TRAJECTORY_HISTORY = 30
BOUNCE_VELOCITY_THRESHOLD = 2.5  # pixels / frame (screen space)
