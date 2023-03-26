from typing import Annotated
import numpy as np
from util import dtype_dataclass, dtype_array

U8 = np.dtype("u1")
U16 = np.dtype("<u2")
U32 = np.dtype("<u4")
F32 = np.dtype("<f4")


# unfinished
@dtype_dataclass
class PA8:
    dtype: np.dtype

    encryption_constant: U32
    sanity: U16
    checksum: U16
    species: U16
    held_item: U16
    tid: U16
    sid: U16
    exp: U32
    ability: U16
    _16: U8
    _17: U8
    mark_value: U16
    _1A: U8
    _1B: U8
    pid: U32
    nature: U8
    stat_nature: U8
    _22: U8
    _23: U8
    form: U16
    evs: Annotated[list[U8], dtype_array(U8, 6)]
    contest_stats: Annotated[list[U8], dtype_array(U8, 6)]
    pokerus: U8  # 0x32
    _33: U8  # 0x33
    ribbon_0: U32  # 0x34
    ribbon_1: U32  # 0x38
    _3C: U8
    _3D: U8
    alpha_move: U16  # 0x3C
    ribbon_2: U32
    ribbon_3: U32
    sociability: U32
    _4C: U8
    _4D: U8
    _4E: U8
    _4F: U8
    height: U8
    weight: U8
    scale: U8
    _53: U8
    moves: Annotated[list[U16], dtype_array(U16, 4)]
    move_pp: Annotated[list[U8], dtype_array(U8, 4)]
    nickname: Annotated[list[U8], dtype_array(U8, 26)]
    _7A_85: Annotated[list[U8], dtype_array(U8, 12)]
    move_pp_ups: Annotated[list[U8], dtype_array(U8, 4)]
    relearn_moves: Annotated[list[U16], dtype_array(U16, 4)]
    current_hp: U16
    iv32: U32
    dynamax_level: U8
    _99: U8
    _9A: U8
    _9B: U8
    status_condition: U32
    _A0: U32
    gvs: Annotated[list[U8], dtype_array(U8, 6)]
    _AA: U8
    _AB: U8
    height_absolute: F32
    weight_absolute: F32
    _B4: U8
    _B5: U8
    _B6: U8
    _B7: U8
    ht_name: Annotated[list[U8], dtype_array(U8, 26)]
    ht_gender: U8
    ht_language: U8
    current_handler: U8
    _D5: U8
    ht_tid: U16
    ht_friendship: U8
    ht_intensity: U8
    ht_memory: U8
    ht_feeling: U8
    gt_text_var: U16
    _DE: U8
    _DF: U8
    _E0: U8
    _E1: U8
    _E2: U8
    _E3: U8
    _E4: U8
    _E5: U8
    _E6: U8
    _E7: U8
    _E8: U8
    _E9: U8
    _EA: U8
    _EB: U8
    fullness: U8
    enjoyment: U8
    version: U8
    battle_version: U8
    _F0: U8
    _F1: U8
    language: U8
    _F3: U8
    form_argument: U32
    affixed_ribbon: U8
    _F9_10A: Annotated[list[U8], dtype_array(U8, 17)]
    ot_name: Annotated[list[U8], dtype_array(U8, 26)]
    _12A_177: Annotated[list[U8], dtype_array(U8, 84)]
