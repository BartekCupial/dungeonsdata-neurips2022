import numpy as np

from nle import nethack


class GoldScore:
    def __init__(self):
        self.score = 0

    def reward(self, env, last_observation, observation, end_status):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_gold = old_blstats[nethack.NLE_BL_GOLD]
        gold = blstats[nethack.NLE_BL_GOLD]

        reward = np.abs(gold - old_gold)
        self.score += reward


class EatingScore:
    def __init__(self):
        self.score = 0

    def reward(self, env, last_observation, observation, end_status):
        old_internal = last_observation[env._internal_index]
        internal = observation[env._internal_index]

        reward = max(0, internal[7] - old_internal[7])
        self.score += reward


class ScoutScore:
    def __init__(self):
        self.score = 0
        self.dungeon_explored = {}

    def reward(self, env, last_observation, observation, end_status):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        self.score += reward


class StaircaseScore:
    """
    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise.
    """

    def __init__(self):
        self.score = 0

    def reward(self, env, last_observation, observation, end_status):
        internal = observation[env._internal_index]
        stairs_down = internal[4]

        if stairs_down:
            self.score += 1


class StaircasePetScore:
    """
    This task requires the agent to get on top of a staircase down (>), while
    having their pet next to it. See `NetHackStaircase` for the reward function.
    """

    def __init__(self):
        self.score = 0

    def reward(self, env, last_observation, observation, end_status):
        internal = observation[env._internal_index]
        stairs_down = internal[4]

        if stairs_down:
            glyphs = observation[env._glyph_index]
            blstats = observation[env._blstats_index]
            x, y = blstats[:2]

            neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
            if np.any(nethack.glyph_is_pet(neighbors)):
                self.score += 1
