from gymnasium.envs.registration import register

register(
    id='FactoredRiverSwim-v0',
    entry_point='mcts_forest.envs.jair_envs:FactoredRiverSwimEnv',
)

register(
    id='FourRooms-v0',
    entry_point='mcts_forest.envs.jair_envs:FourRoomsEnv',
)

register(
    id='PassengerGrid-v0',
    entry_point='mcts_forest.envs.jair_envs:PassengerGridEnv',
)

register(
    id='SysadminRing-v0',
    entry_point='mcts_forest.envs.jair_envs:SysadminRingEnv',
)
