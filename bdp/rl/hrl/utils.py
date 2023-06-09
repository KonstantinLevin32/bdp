from typing import Tuple

from habitat.core.spaces import ActionSpace
from bdp.utils.common import get_num_actions


def find_action_range(
    action_space: ActionSpace, search_key: str
) -> Tuple[int, int]:
    start_idx = 0
    found = False
    end_idx = get_num_actions(action_space[search_key])
    for k in action_space:
        if k == search_key:
            found = True
            break
        start_idx += get_num_actions(action_space[k])
    if not found:
        raise ValueError(f"Could not find STOP action in {action_space}")
    return start_idx, end_idx
