from gym.envs.registration import register

# Gazebo
# ----------------------------------------
# AICar
register(
    id='AICar-v0',
    entry_point='gazeborlenv.envs:AICarEnv',
)
