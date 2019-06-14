from gym.envs.registration import register

register(
    id='{}-{}'.format('SlotMachines', 'v0'),
    entry_point='slot_machines.slot_machines:{}'.format('SlotMachines'),
    max_episode_steps=1,
    nondeterministic=True)